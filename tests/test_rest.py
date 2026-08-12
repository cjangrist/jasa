"""REST routes: auth, /search, /fetch, /researcher, body cap, errors."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
import respx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.testclient import TestClient

from jasa.auth import is_authorized
from jasa.config import load_config
from jasa.rest import register_provider_resources
from jasa.server import build_composition
from omnifetch.fetch.shared.types import ErrorType, ProviderError
from omnifetch.schemas import FetchResponse


def _server() -> Any:
    """Return a composed server for REST testing."""
    return build_composition(load_config()).server


class _ResourceServer:
    def __init__(self) -> None:
        self.resources: dict[str, Any] = {}

    def resource(self, uri: str) -> Any:
        def register(function: Any) -> Any:
            self.resources[uri] = function
            return function

        return register


def test_search_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    composition = build_composition(load_config())
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "T",
                            "url": "https://x.com",
                            "content": "c" * 320,
                            "score": 0.9,
                        }
                    ]
                },
            )
        )
        with TestClient(composition.server.http_app()) as client:
            response = client.post("/search", json={"query": "t", "count": 1})
    assert response.status_code == 200
    assert response.json()[0]["link"] == "https://x.com"


def test_search_omitted_count_defaults_to_twenty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    composition = build_composition(load_config())
    provider_results = [
        {
            "title": f"T{index}",
            "url": f"https://host{index}.example/result",
            "content": "c" * 320,
            "score": 1 - index / 100,
        }
        for index in range(25)
    ]
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json={"results": provider_results})
        )
        with TestClient(composition.server.http_app()) as client:
            response = client.post("/search", json={"query": "t"})
    assert response.status_code == 200
    assert len(response.json()) == 20


def test_search_no_providers_returns_503() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post("/search", json={"query": "test"})
    assert response.status_code == 503


def test_auth_rejects_wrong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_API_KEY", "secret")
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/search",
            json={"query": "test"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert response.status_code == 401


def test_auth_rejects_non_ascii_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_API_KEY", "secret")
    request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", b"Bearer caf\xe9")],
            "query_string": b"",
        }
    )
    assert is_authorized(request) is False


def test_body_cap_returns_413() -> None:
    big_body = '{"query": "' + "x" * 70000 + '"}'
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/search",
            content=big_body,
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 413


def test_bad_query_returns_400() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post("/search", json={"query": ""})
    assert response.status_code == 400


def test_invalid_json_returns_400() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/search",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400


def test_non_dict_body_returns_400() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post("/search", json=[1, 2, 3])
    assert response.status_code == 400


@pytest.mark.parametrize("count", [None, "abc"])
def test_search_invalid_count_defaults_to_twenty(
    monkeypatch: pytest.MonkeyPatch, count: object
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    composition = build_composition(load_config())
    provider_results = [
        {
            "title": f"T{index}",
            "url": f"https://host{index}.example/result",
            "content": "c" * 320,
            "score": 1 - index / 100,
        }
        for index in range(25)
    ]
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json={"results": provider_results})
        )
        with TestClient(composition.server.http_app()) as client:
            response = client.post(
                "/search", json={"query": "test", "count": count}
            )
    assert response.status_code == 200
    assert len(response.json()) == 20


def test_fetch_route_returns_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_skip: object = None

    async def succeed(
        engine: object,
        url: str,
        *,
        skip_providers: object = None,
    ) -> FetchResponse:
        nonlocal captured_skip
        captured_skip = skip_providers
        return FetchResponse(
            url=url,
            title="Example",
            content="Fetched content",
            source_provider="fake",
            total_duration_ms=1,
        )

    monkeypatch.setattr("omnifetch.tools.fetch.execute_web_fetch", succeed)
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/fetch",
            json={
                "url": "https://x.com",
                "skip_providers": ["scrapfly"],
            },
        )
    assert response.status_code == 200
    assert response.json()["content"] == "Fetched content"
    assert captured_skip == ["scrapfly"]


def test_fetch_missing_url_returns_400() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post("/fetch", json={})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "skip_providers",
    [1, {"provider": "tavily"}, ["tavily", 1]],
)
def test_fetch_invalid_skip_providers_returns_400(
    skip_providers: object,
) -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/fetch",
            json={
                "url": "https://x.com",
                "skip_providers": skip_providers,
            },
        )
    assert response.status_code == 400
    assert response.json() == {
        "error": "skip_providers must be a string or list of strings"
    }


def test_fetch_string_skip_provider_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_skip: object = None

    async def succeed(
        engine: object,
        url: str,
        *,
        skip_providers: object = None,
    ) -> FetchResponse:
        nonlocal captured_skip
        captured_skip = skip_providers
        return FetchResponse(
            url=url,
            title="Example",
            content="Fetched content",
            source_provider="fake",
            total_duration_ms=1,
        )

    monkeypatch.setattr("omnifetch.tools.fetch.execute_web_fetch", succeed)
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/fetch",
            json={
                "url": "https://x.com",
                "skip_providers": "scrapfly",
            },
        )
    assert response.status_code == 200
    assert captured_skip == "scrapfly"


def test_fetch_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_API_KEY", "secret")
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/fetch",
            json={"url": "https://x.com"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert response.status_code == 401


def test_fetch_invalid_json_returns_400() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/fetch",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400


def test_fetch_timeout_returns_504(monkeypatch: pytest.MonkeyPatch) -> None:
    async def timeout(
        engine: object,
        url: str,
        *,
        skip_providers: object = None,
    ) -> None:
        raise TimeoutError

    monkeypatch.setattr("omnifetch.tools.fetch.execute_web_fetch", timeout)
    with TestClient(_server().http_app()) as client:
        response = client.post("/fetch", json={"url": "https://x.com"})
    assert response.status_code == 504


@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [
        (ErrorType.INVALID_INPUT, 400),
        (ErrorType.RATE_LIMIT, 429),
        (ErrorType.NOT_FOUND, 404),
        (ErrorType.API_ERROR, 502),
    ],
)
def test_fetch_provider_error_status_mapping(
    monkeypatch: pytest.MonkeyPatch,
    error_type: ErrorType,
    expected_status: int,
) -> None:
    async def fail(
        engine: object,
        url: str,
        *,
        skip_providers: object = None,
    ) -> None:
        raise ProviderError(error_type, "failed", "fake")

    monkeypatch.setattr("omnifetch.tools.fetch.execute_web_fetch", fail)
    with TestClient(_server().http_app()) as client:
        response = client.post("/fetch", json={"url": "https://x.com"})
    assert response.status_code == expected_status


def test_researcher_get_no_providers_503() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.get("/researcher?query=test")
    assert response.status_code == 503


def test_researcher_post_no_providers_503() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post("/researcher", json={"query": "test"})
    assert response.status_code == 503


def test_researcher_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_API_KEY", "secret")
    with TestClient(_server().http_app()) as client:
        response = client.get(
            "/researcher?query=test",
            headers={"Authorization": "Bearer wrong"},
        )
    assert response.status_code == 401


def test_researcher_post_invalid_json_returns_400() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.post(
            "/researcher",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400


def test_researcher_missing_query_returns_400() -> None:
    with TestClient(_server().http_app()) as client:
        response = client.get("/researcher")
    assert response.status_code == 400


def test_auth_query_param_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_API_KEY", "secret")
    with TestClient(_server().http_app()) as client:
        response = client.get("/researcher?query=test&key=secret")
    assert response.status_code == 503


def test_researcher_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    composition = build_composition(load_config())
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "T",
                            "url": "https://x.com",
                            "content": "c" * 320,
                        }
                    ]
                },
            )
        )
        with TestClient(composition.server.http_app()) as client:
            response = client.get("/researcher?query=test")
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) >= 1
    assert entries[0]["href"] == "https://x.com"


async def test_provider_resources_status_and_info() -> None:
    server = _ResourceServer()
    register_provider_resources(
        cast("FastMCP[Any]", server), ["tavily"], ["jina"], load_config()
    )
    status = json.loads(await server.resources["jasa://providers/status"]())
    search_info = json.loads(
        await server.resources["jasa://providers/{provider}/info"]("tavily")
    )
    fetch_info = json.loads(
        await server.resources["jasa://providers/{provider}/info"]("jina")
    )
    assert status["search"]["providers"] == ["tavily"]
    assert search_info["family"] == "search"
    assert search_info["capabilities"] == ["web_search"]
    assert fetch_info["family"] == "fetch"
    assert fetch_info["capabilities"] == ["fetch"]
    with pytest.raises(ValueError, match="unknown provider"):
        await server.resources["jasa://providers/{provider}/info"]("missing")

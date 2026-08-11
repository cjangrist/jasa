"""Tavily provider: exact outbound-request + full error-mapping parity.

The full §8 matrix for one adapter: exact outbound request, happy path, empty
result, malformed body, 401/403/429/5xx status mapping, timeout, oversized
response, missing key, operator-domain merge, and key-absent-from-logs. This is
the template the other nine adapters follow.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.tavily import TavilyProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

TAVILY_URL = "https://api.tavily.com/search"
_KEY = "tvly-test-key"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as async_client:
        yield async_client


def _provider(client: httpx.AsyncClient, key: str = _KEY) -> TavilyProvider:
    return TavilyProvider(key, client)


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"results": results})


async def test_exact_outbound_request_and_mapping(
    client: httpx.AsyncClient,
) -> None:
    provider = _provider(client)
    with respx.mock:
        route = respx.post(TAVILY_URL).mock(
            return_value=_ok(
                [
                    {
                        "title": "T",
                        "url": "https://x.com",
                        "content": "snip",
                        "score": 0.9,
                    }
                ]
            )
        )
        results = await provider.search(
            SearchRequest(
                query="hello world", limit=5, include_domains=("a.com",)
            )
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == TAVILY_URL
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    body = json.loads(request.content)
    assert body == {
        "query": "hello world",
        "max_results": 5,
        "include_domains": ["a.com"],
        "exclude_domains": [],
        "search_depth": "basic",
        "topic": "general",
    }
    assert results == [
        SearchResult(
            title="T",
            url="https://x.com",
            snippet="snip",
            source_provider="tavily",
            score=0.9,
        )
    ]


async def test_operator_domains_merged_into_structural_lists(
    client: httpx.AsyncClient,
) -> None:
    provider = _provider(client)
    with respx.mock:
        route = respx.post(TAVILY_URL).mock(return_value=_ok([]))
        await provider.search(
            SearchRequest(
                query="site:b.com foo -site:c.com", include_domains=("a.com",)
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body["query"] == "foo"
    assert body["include_domains"] == ["a.com", "b.com"]
    assert body["exclude_domains"] == ["c.com"]


async def test_empty_results_is_success(client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(TAVILY_URL).mock(return_value=_ok([]))
        results = await _provider(client).search(SearchRequest(query="q"))
    assert results == []


async def test_missing_results_key_is_success(
    client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={}))
        results = await _provider(client).search(SearchRequest(query="q"))
    assert results == []


async def test_malformed_body_raises_api_error(
    client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(
                200,
                text="<html>not json</html>",
                headers={"content-type": "text/html"},
            )
        )
        with pytest.raises(ProviderError) as exc:
            await _provider(client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == "Invalid JSON response from tavily"


async def test_401_raises_api_error(client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(401, json={}))
        with pytest.raises(ProviderError) as exc:
            await _provider(client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == "Invalid API key"


async def test_403_raises_api_error(client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(403, json={}))
        with pytest.raises(ProviderError) as exc:
            await _provider(client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.API_ERROR
    assert "does not have access" in str(exc.value)


async def test_429_raises_rate_limit(client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(429, json={}))
        with pytest.raises(ProviderError) as exc:
            await _provider(client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.RATE_LIMIT
    assert str(exc.value) == "Rate limit exceeded for tavily"


async def test_500_raises_provider_error(client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(500, json={"message": "boom"})
        )
        with pytest.raises(ProviderError) as exc:
            await _provider(client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.PROVIDER_ERROR
    assert "internal error (500)" in str(exc.value)


async def test_timeout_raises_provider_error(client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(TAVILY_URL).mock(
            side_effect=httpx.ReadTimeout("read timed out")
        )
        with pytest.raises(ProviderError) as exc:
            await _provider(client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.PROVIDER_ERROR


async def test_oversized_response_raises_api_error(
    client: httpx.AsyncClient,
) -> None:
    too_large = 6 * 1024 * 1024
    with respx.mock:
        respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(
                200,
                headers={"content-length": str(too_large)},
                json={},
            )
        )
        with pytest.raises(ProviderError) as exc:
            await _provider(client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == f"Response too large ({too_large} bytes)"


async def test_missing_key_raises_invalid_input(
    client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await _provider(client, key="").search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for tavily"


async def test_unexpected_error_wrapped_as_api_error(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("kaboom")

    monkeypatch.setattr("jasa.search.providers.base.http_json", boom)
    with pytest.raises(ProviderError) as exc:
        await _provider(client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.API_ERROR
    assert "Failed to fetch search results" in str(exc.value)


async def test_api_key_absent_from_logs(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with respx.mock:
        respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(500, json={"message": "boom"})
        )
        with pytest.raises(ProviderError):
            await _provider(client).search(SearchRequest(query="q"))
    assert _KEY not in caplog.text

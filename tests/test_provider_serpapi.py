"""Serpapi provider: key-in-URL request, mapping, and URL-param redaction."""

from __future__ import annotations

import logging

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.serpapi import SerpapiProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

SERPAPI_URL = "https://serpapi.com/search.json"
_KEY = "serp-test-key"


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"organic_results": results})


async def test_exact_outbound_request_key_in_url(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.get(SERPAPI_URL).mock(
            return_value=_ok(
                [{"title": "T", "link": "https://x.com", "snippet": "s"}]
            )
        )
        results = await SerpapiProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=8)
        )
        params = route.calls.last.request.url.params
    assert params["engine"] == "google_light"
    assert params["q"] == "q"
    assert params["api_key"] == _KEY
    assert params["num"] == "8"
    assert results == [
        SearchResult(
            title="T",
            url="https://x.com",
            snippet="s",
            source_provider="serpapi",
        )
    ]


async def test_empty_snippet_defaults_to_blank(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get(SERPAPI_URL).mock(
            return_value=_ok([{"title": "T", "link": "u"}])
        )
        results = await SerpapiProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results[0].snippet == ""


async def test_api_key_redacted_from_logs(
    http_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = logging.getLogger("jasa")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.DEBUG, logger="jasa")
    with respx.mock:
        respx.get(SERPAPI_URL).mock(
            return_value=httpx.Response(500, json={"message": "x"})
        )
        with pytest.raises(ProviderError) as exc:
            await SerpapiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert _KEY not in caplog.text
    assert _KEY not in str(exc.value)


async def test_api_key_redacted_from_transport_error(
    http_client: httpx.AsyncClient,
) -> None:
    request = httpx.Request(
        "GET", f"{SERPAPI_URL}?engine=google_light&api_key={_KEY}"
    )
    with respx.mock:
        respx.get(SERPAPI_URL).mock(
            side_effect=httpx.ConnectError(
                f"failed request {request.url}", request=request
            )
        )
        with pytest.raises(ProviderError) as exc:
            await SerpapiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert _KEY not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await SerpapiProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for serpapi"

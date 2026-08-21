"""Brave provider: exact request + operator re-render + error parity."""

from __future__ import annotations

import httpx
import pytest
import respx

from jasa.search.fanout import _PER_PROVIDER_LIMIT
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.brave import BraveProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_KEY = "brave-test-key"


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"web": {"results": results}})


async def test_exact_outbound_request_and_mapping(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.get(BRAVE_URL).mock(
            return_value=_ok(
                [{"title": "T", "url": "https://x.com", "description": "d"}]
            )
        )
        results = await BraveProvider(_KEY, http_client).search(
            SearchRequest(query="hello world", limit=7)
        )
        request = route.calls.last.request
    assert request.method == "GET"
    assert request.url.params["q"] == "hello world"
    assert request.url.params["count"] == "7"
    assert request.headers["x-subscription-token"] == _KEY
    assert results == [
        SearchResult(
            title="T",
            url="https://x.com",
            snippet="d",
            source_provider="brave",
        )
    ]


async def test_count_is_clamped_to_the_brave_maximum(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.get(BRAVE_URL).mock(return_value=_ok([]))
        await BraveProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=_PER_PROVIDER_LIMIT)
        )
        request = route.calls.last.request
    assert _PER_PROVIDER_LIMIT > 20
    assert request.url.params["count"] == "20"


async def test_operators_rerendered_into_query(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.get(BRAVE_URL).mock(return_value=_ok([]))
        await BraveProvider(_KEY, http_client).search(
            SearchRequest(query="site:a.com filetype:pdf foo")
        )
        rendered_q = route.calls.last.request.url.params["q"]
    assert "site:a.com" in rendered_q
    assert "filetype:pdf" in rendered_q


async def test_missing_web_is_success(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(200, json={}))
        results = await BraveProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_429_raises_rate_limit(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(429, json={}))
        with pytest.raises(ProviderError) as exc:
            await BraveProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.RATE_LIMIT


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await BraveProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for brave"

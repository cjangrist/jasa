"""Firecrawl provider: success-flag failure, empty-web success, fallbacks."""

from __future__ import annotations

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.firecrawl import FirecrawlProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

FIRECRAWL_URL = "https://api.firecrawl.dev/v2/search"
_KEY = "fc-test-key"


async def test_success_false_is_failure(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(FIRECRAWL_URL).mock(
            return_value=httpx.Response(200, json={"success": False})
        )
        with pytest.raises(ProviderError) as exc:
            await FirecrawlProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == (
        "Failed to fetch search results: Firecrawl API returned success: false"
    )


async def test_missing_web_is_success(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(FIRECRAWL_URL).mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        results = await FirecrawlProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_maps_with_fallbacks_and_url_filter(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(FIRECRAWL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "web": [
                            {"url": "u1", "title": "t", "description": "d"},
                            {"url": "u2"},
                            {"title": "no url"},
                        ]
                    },
                },
            )
        )
        results = await FirecrawlProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="t", url="u1", snippet="d", source_provider="firecrawl"
        ),
        SearchResult(
            title="Source", url="u2", snippet="", source_provider="firecrawl"
        ),
    ]


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await FirecrawlProvider("", http_client).search(
            SearchRequest(query="q")
        )
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for firecrawl"

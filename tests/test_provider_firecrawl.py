"""Firecrawl provider: collections, success flag, fallbacks, malformed data."""

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


async def test_missing_data_is_success(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(FIRECRAWL_URL).mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        results = await FirecrawlProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_maps_the_keyed_web_collection(
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
                            {
                                "url": "https://a.com",
                                "title": "A",
                                "description": "d",
                                "position": 1,
                            },
                            {"url": "https://b.com"},
                            {"title": "no url"},
                        ],
                        "news": [{"url": "https://news.com", "title": "N"}],
                    },
                },
            )
        )
        results = await FirecrawlProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="A",
            url="https://a.com",
            snippet="d",
            source_provider="firecrawl",
        ),
        SearchResult(
            title="Source",
            url="https://b.com",
            snippet="",
            source_provider="firecrawl",
        ),
    ]


async def test_missing_web_collection_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(FIRECRAWL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {"news": [{"url": "https://n"}]},
                },
            )
        )
        results = await FirecrawlProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_malformed_collections_are_ignored_not_raised(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(FIRECRAWL_URL)
        route.side_effect = [
            httpx.Response(
                200, json={"success": True, "data": {"web": "not a list"}}
            ),
            httpx.Response(200, json={"success": True, "data": "not a list"}),
            httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "web": [
                            "a bare string",
                            7,
                            {"url": ["unhashable"]},
                            {"url": 42},
                            {
                                "url": "https://a.com",
                                "title": ["not", "a", "string"],
                                "description": {"not": "a string"},
                            },
                        ]
                    },
                },
            ),
        ]
        provider = FirecrawlProvider(_KEY, http_client)
        assert await provider.search(SearchRequest(query="a")) == []
        assert await provider.search(SearchRequest(query="b")) == []
        results = await provider.search(SearchRequest(query="c"))
    assert results == [
        SearchResult(
            title="Source",
            url="https://a.com",
            snippet="",
            source_provider="firecrawl",
        )
    ]


async def test_legacy_flat_array_is_still_accepted(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(FIRECRAWL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "data": [
                        {"url": "u1", "title": "t", "description": "d"},
                        {"url": "u2"},
                        {"title": "no url"},
                    ],
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

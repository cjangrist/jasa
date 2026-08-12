"""You.com provider: POST JSON, nested web results, snippet fallbacks."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.you import YouProvider
from omnifetch.fetch.shared.types import ErrorType, ProviderError

YOU_URL = "https://ydc-index.io/v1/search"
_KEY = "you-test-key"


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"results": {"web": results}})


async def test_snippet_join_and_description_fallback(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(YOU_URL).mock(
            return_value=_ok(
                [
                    {"title": "a", "url": "u1", "snippets": ["one", "two"]},
                    {"title": "b", "url": "u2", "description": "d"},
                    {"title": "c", "url": "u3"},
                ]
            )
        )
        results = await YouProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [r.snippet for r in results] == ["one two", "d", ""]


async def test_exact_outbound_headers(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(YOU_URL).mock(return_value=_ok([]))
        await YouProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=4)
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert request.headers["x-api-key"] == _KEY
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {"query": "q", "count": 4}


async def test_missing_results_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(YOU_URL).mock(return_value=httpx.Response(200, json={}))
        results = await YouProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_news_results_are_ignored(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(YOU_URL).mock(
            return_value=httpx.Response(
                200,
                json={"results": {"news": [{"title": "News"}]}},
            )
        )
        results = await YouProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await YouProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for you"

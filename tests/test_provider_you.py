"""You.com provider: snippet join + description fallback, header parity."""

from __future__ import annotations

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.you import YouProvider
from omnifetch.fetch.shared.types import ErrorType, ProviderError

YOU_URL = "https://ydc-index.io/v1/search"
_KEY = "you-test-key"


def _ok(web: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"results": {"web": web}})


async def test_snippet_join_and_description_fallback(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.get(YOU_URL).mock(
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
        route = respx.get(YOU_URL).mock(return_value=_ok([]))
        await YouProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=4)
        )
        request = route.calls.last.request
    assert request.url.params["query"] == "q"
    assert request.url.params["count"] == "4"
    assert request.headers["x-api-key"] == _KEY


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await YouProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for you"

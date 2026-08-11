"""Linkup provider: text-result filter, structural domain filters, mapping."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.linkup import LinkupProvider
from omnifetch.fetch.shared.types import ErrorType, ProviderError

LINKUP_URL = "https://api.linkup.so/v1/search"
_KEY = "linkup-test-key"


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"results": results})


async def test_only_text_results_survive(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(LINKUP_URL).mock(
            return_value=_ok(
                [
                    {"type": "text", "name": "t", "url": "u", "content": "c"},
                    {
                        "type": "image",
                        "name": "img",
                        "url": "u2",
                        "content": "c2",
                    },
                ]
            )
        )
        results = await LinkupProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert len(results) == 1
    assert results[0].title == "t"
    assert results[0].snippet == "c"


async def test_domain_filters_attached(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(LINKUP_URL).mock(return_value=_ok([]))
        await LinkupProvider(_KEY, http_client).search(
            SearchRequest(
                query="q",
                include_domains=("a.com",),
                exclude_domains=("b.com",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body["includeDomains"] == ["a.com"]
    assert body["excludeDomains"] == ["b.com"]


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await LinkupProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for linkup"

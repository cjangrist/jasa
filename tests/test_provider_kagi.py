"""Kagi provider: JSON request, lens operators, and Bearer auth."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.kagi import KagiProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

KAGI_URL = "https://kagi.com/api/v1/search"
_KEY = "kagi-test-key"


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"data": {"search": results}})


async def test_exact_outbound_request_and_mapping(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KAGI_URL).mock(
            return_value=_ok(
                [
                    {
                        "title": "T",
                        "url": "https://x.com",
                        "snippet": "s",
                        "rank": 1,
                    }
                ]
            )
        )
        results = await KagiProvider(_KEY, http_client).search(
            SearchRequest(query="hello world", limit=9)
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {"query": "hello world", "limit": 9}
    assert results == [
        SearchResult(
            title="T",
            url="https://x.com",
            snippet="s",
            source_provider="kagi",
        )
    ]


async def test_operators_and_domains_become_lens_fields(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KAGI_URL).mock(return_value=_ok([]))
        await KagiProvider(_KEY, http_client).search(
            SearchRequest(
                query=(
                    "filetype:pdf before:2024-12-31 after:2023-01-01 "
                    "site:query.example -site:blocked.example foo"
                ),
                include_domains=("included.example",),
                exclude_domains=("excluded.example",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "foo",
        "limit": 20,
        "lens": {
            "sites_included": ["included.example", "query.example"],
            "sites_excluded": ["excluded.example", "blocked.example"],
            "file_type": "pdf",
            "time_after": "2023-01-01",
            "time_before": "2024-12-31",
        },
    }


async def test_time_range_before_only(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        route = respx.post(KAGI_URL).mock(return_value=_ok([]))
        await KagiProvider(_KEY, http_client).search(
            SearchRequest(query="before:2024-06-01 foo")
        )
        body = json.loads(route.calls.last.request.content)
    assert body["lens"] == {"time_before": "2024-06-01"}


async def test_time_range_after_only(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        route = respx.post(KAGI_URL).mock(return_value=_ok([]))
        await KagiProvider(_KEY, http_client).search(
            SearchRequest(query="after:2023-01-01 foo")
        )
        body = json.loads(route.calls.last.request.content)
    assert body["lens"] == {"time_after": "2023-01-01"}


async def test_missing_data_is_success(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(KAGI_URL).mock(return_value=httpx.Response(200, json={}))
        results = await KagiProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_401_raises_api_error(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(KAGI_URL).mock(return_value=httpx.Response(401, json={}))
        with pytest.raises(ProviderError) as exc:
            await KagiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == "Invalid API key"


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await KagiProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for kagi"

"""Parallel provider: advanced-mode body, excerpts join, title fallback."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.parallel import ParallelProvider
from omnifetch.fetch.shared.types import ErrorType, ProviderError

PARALLEL_URL = "https://api.parallel.ai/v1/search"
_KEY = "parallel-test-key"
_OBJECTIVE = (
    "Return the most relevant, recent, high-signal sources for this query."
)


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"results": results})


async def test_exact_outbound_advanced_body(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(PARALLEL_URL).mock(return_value=_ok([]))
        await ParallelProvider(_KEY, http_client).search(
            SearchRequest(
                query="q",
                limit=5,
                include_domains=("a.com",),
                exclude_domains=("b.com",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "objective": _OBJECTIVE,
        "search_queries": ["q"],
        "mode": "advanced",
        "advanced_settings": {
            "max_results": 5,
            "source_policy": {
                "include_domains": ["a.com"],
                "exclude_domains": ["b.com"],
            },
        },
    }


async def test_non_array_results_is_empty(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(PARALLEL_URL).mock(return_value=httpx.Response(200, json={}))
        results = await ParallelProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_excerpts_joined_and_title_fallback(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(PARALLEL_URL).mock(
            return_value=_ok(
                [
                    {"url": "u1", "excerpts": ["para one", "para two"]},
                    {"url": "u2", "title": "t"},
                ]
            )
        )
        results = await ParallelProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results[0].snippet == "para one\n\npara two"
    assert results[0].title == "u1"
    assert results[1].title == "t"


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await ParallelProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for parallel"

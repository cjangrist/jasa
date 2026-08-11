"""Perplexity provider: structured-results preference + citations fallback."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.perplexity import PerplexityProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
_KEY = "pplx-test-key"


async def test_structured_results_preferred(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(PERPLEXITY_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "search_results": [
                        {"url": "u", "title": "t", "snippet": "s"}
                    ],
                    "citations": ["ignored"],
                },
            )
        )
        results = await PerplexityProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="t", url="u", snippet="s", source_provider="perplexity"
        )
    ]


async def test_citations_fallback(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(PERPLEXITY_URL).mock(
            return_value=httpx.Response(200, json={"citations": ["u1", "u2"]})
        )
        results = await PerplexityProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=5)
        )
    assert results == [
        SearchResult(
            title="Source", url="u1", snippet="", source_provider="perplexity"
        ),
        SearchResult(
            title="Source", url="u2", snippet="", source_provider="perplexity"
        ),
    ]


async def test_empty_citations_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(PERPLEXITY_URL).mock(
            return_value=httpx.Response(200, json={"citations": []})
        )
        results = await PerplexityProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_exact_outbound_body(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        route = respx.post(PERPLEXITY_URL).mock(
            return_value=httpx.Response(200, json={"citations": []})
        )
        await PerplexityProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "model": "sonar",
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0.1,
        "max_tokens": 256,
        "web_search_options": {"search_context_size": "high"},
    }


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await PerplexityProvider("", http_client).search(
            SearchRequest(query="q")
        )
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for perplexity"

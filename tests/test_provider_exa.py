"""Exa provider: dual-header auth, contents body, snippet/score fallbacks."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.exa import ExaProvider
from omnifetch.fetch.shared.types import ErrorType, ProviderError

EXA_URL = "https://api.exa.ai/search"
_KEY = "exa-test-key"


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"results": results})


async def test_exact_outbound_request_both_headers(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(EXA_URL).mock(return_value=_ok([]))
        await ExaProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=3, include_domains=("a.com",))
        )
        request = route.calls.last.request
        body = json.loads(request.content)
    assert request.headers["x-api-key"] == _KEY
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert body == {
        "query": "q",
        "type": "auto",
        "numResults": 3,
        "useAutoprompt": True,
        "contents": {"text": {"maxCharacters": 1500}, "livecrawl": "fallback"},
        "includeDomains": ["a.com"],
    }


async def test_exclude_domains_attached(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        route = respx.post(EXA_URL).mock(return_value=_ok([]))
        await ExaProvider(_KEY, http_client).search(
            SearchRequest(query="q", exclude_domains=("b.com",))
        )
        body = json.loads(route.calls.last.request.content)
    assert body["excludeDomains"] == ["b.com"]
    assert "includeDomains" not in body


async def test_snippet_fallback_chain(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(EXA_URL).mock(
            return_value=_ok(
                [
                    {"url": "u1", "text": "t"},
                    {"url": "u2", "summary": "s"},
                    {"url": "u3"},
                ]
            )
        )
        results = await ExaProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [r.snippet for r in results] == ["t", "s", "No content available"]


async def test_falsy_score_becomes_zero(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(EXA_URL).mock(
            return_value=_ok(
                [{"url": "u1", "score": 0}, {"url": "u2", "score": 0.5}]
            )
        )
        results = await ExaProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results[0].score == 0
    assert results[1].score == 0.5


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await ExaProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for exa"

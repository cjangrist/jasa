"""Serper provider request, operator, mapping, and error behavior."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.serper import SerperProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

SERPER_URL = "https://google.serper.dev/search"
_KEY = "serper-test-key"


def _ok(results: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"organic": results})


async def test_exact_outbound_request_and_mapping(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(SERPER_URL).mock(
            return_value=_ok(
                [{"title": "T", "link": "https://x.com", "snippet": "s"}]
            )
        )
        results = await SerperProvider(_KEY, http_client).search(
            SearchRequest(query="hello world", limit=7)
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert request.headers["x-api-key"] == _KEY
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {"q": "hello world", "num": 7}
    assert results == [
        SearchResult(
            title="T",
            url="https://x.com",
            snippet="s",
            source_provider="serper",
        )
    ]


async def test_operators_and_domain_filters_are_preserved(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(SERPER_URL).mock(return_value=_ok([]))
        await SerperProvider(_KEY, http_client).search(
            SearchRequest(
                query="site:b.com filetype:pdf foo -site:c.com",
                include_domains=("a.com",),
                exclude_domains=("d.com",),
            )
        )
        query = json.loads(route.calls.last.request.content)["q"]
    assert query == (
        "foo site:a.com OR site:b.com -site:d.com -site:c.com filetype:pdf"
    )


async def test_missing_organic_and_snippet_are_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(SERPER_URL)
        route.side_effect = [
            httpx.Response(200, json={}),
            _ok([{"title": "T", "link": "https://x.com"}]),
        ]
        empty = await SerperProvider(_KEY, http_client).search(
            SearchRequest(query="empty")
        )
        fallback = await SerperProvider(_KEY, http_client).search(
            SearchRequest(query="fallback", limit=0)
        )
        second_body = json.loads(route.calls[1].request.content)
    assert empty == []
    assert fallback[0].snippet == ""
    assert second_body["num"] == 20


async def test_invalid_key_is_redacted(
    http_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with respx.mock:
        respx.post(SERPER_URL).mock(
            return_value=httpx.Response(
                401, json={"message": f"invalid key {_KEY}"}
            )
        )
        with pytest.raises(ProviderError) as exc:
            await SerperProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert _KEY not in str(exc.value)
    assert _KEY not in caplog.text


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await SerperProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for serper"

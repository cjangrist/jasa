"""FastCRW Cloud search request, mapping, and shared error taxonomy tests."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.fastcrw import FastcrwProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_URL = "https://api.fastcrw.com/v1/search"
_KEY = "fastcrw-test-key"


def _response(data: object, *, success: bool = True) -> httpx.Response:
    """Build one FastCRW search envelope."""
    return httpx.Response(200, json={"success": success, "data": data})


def _provider(client: httpx.AsyncClient, key: str = _KEY) -> FastcrwProvider:
    """Construct the provider under test."""
    return FastcrwProvider(key, client)


async def test_exact_request_mapping_and_limit(
    http_client: httpx.AsyncClient,
) -> None:
    """POST the documented Cloud request and retain its native score."""
    payload = [
        {
            "title": "First",
            "url": "https://one.example",
            "snippet": "First snippet",
            "score": 0.8,
        },
        {"url": "https://two.example", "description": "Second snippet"},
    ]
    with respx.mock:
        route = respx.post(_URL).mock(return_value=_response(payload))
        results = await _provider(http_client).search(
            SearchRequest(query="hello", limit=30)
        )
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert request.extensions["timeout"]["read"] == 20.0
    assert json.loads(request.content) == {"query": "hello", "limit": 20}
    assert results == [
        SearchResult(
            "First", "https://one.example", "First snippet", "fastcrw", 0.8
        ),
        SearchResult(
            "Source", "https://two.example", "Second snippet", "fastcrw"
        ),
    ]


async def test_operators_and_domains_are_rendered_into_query(
    http_client: httpx.AsyncClient,
) -> None:
    """Use query text because FastCRW exposes no structural domain filter."""
    with respx.mock:
        route = respx.post(_URL).mock(return_value=_response([]))
        await _provider(http_client).search(
            SearchRequest(
                query="site:b.com filetype:pdf foo -site:c.com after:2026",
                include_domains=("a.com",),
                exclude_domains=("d.com",),
            )
        )
    body = json.loads(route.calls.last.request.content)
    assert body["query"] == (
        "foo site:a.com OR site:b.com -site:d.com -site:c.com "
        "filetype:pdf after:2026"
    )
    assert body["limit"] == 20


@pytest.mark.parametrize(
    "data",
    [[], None, {}, {"web": []}, {"results": []}, {"results": {"web": []}}],
)
async def test_documented_empty_shapes_are_successful(
    http_client: httpx.AsyncClient, data: object
) -> None:
    """Missing or empty result collections are a normal no-results answer."""
    with respx.mock:
        respx.post(_URL).mock(return_value=_response(data))
        assert (
            await _provider(http_client).search(SearchRequest(query="q")) == []
        )


async def test_grouped_and_wrapped_results_are_mapped(
    http_client: httpx.AsyncClient,
) -> None:
    """Accept the documented hosted grouped and self-hosted wrapper forms."""
    grouped = {"web": [{"title": "Grouped", "url": "https://grouped.example"}]}
    wrapped = {"results": {"web": [{"url": "https://wrapped.example"}]}}
    with respx.mock:
        respx.post(_URL).mock(
            side_effect=[_response(grouped), _response(wrapped)]
        )
        grouped_result = await _provider(http_client).search(
            SearchRequest(query="q")
        )
        wrapped_result = await _provider(http_client).search(
            SearchRequest(query="q")
        )
    assert [row.url for row in grouped_result] == ["https://grouped.example"]
    assert [row.url for row in wrapped_result] == ["https://wrapped.example"]


async def test_malformed_rows_are_ignored_and_false_success_fails(
    http_client: httpx.AsyncClient,
) -> None:
    """Do not cache a body that explicitly reports a FastCRW failure."""
    malformed: list[object] = [
        None,
        "bad",
        {"url": []},
        {"url": "https://valid.example", "score": True},
    ]
    with respx.mock:
        respx.post(_URL).mock(
            side_effect=[_response(malformed), _response([], success=False)]
        )
        results = await _provider(http_client).search(SearchRequest(query="q"))
        with pytest.raises(ProviderError) as caught:
            await _provider(http_client).search(SearchRequest(query="q"))
    assert results == [
        SearchResult("Source", "https://valid.example", "", "fastcrw")
    ]
    assert caught.value.error_type is ErrorType.API_ERROR
    assert str(caught.value) == "FastCRW Cloud search returned success: false"


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, ErrorType.API_ERROR),
        (429, ErrorType.RATE_LIMIT),
        (500, ErrorType.PROVIDER_ERROR),
    ],
)
async def test_http_errors_follow_shared_taxonomy(
    http_client: httpx.AsyncClient, status_code: int, error_type: ErrorType
) -> None:
    """Delegate HTTP classification to the shared omnifetch client helper."""
    with respx.mock:
        respx.post(_URL).mock(return_value=httpx.Response(status_code, json={}))
        with pytest.raises(ProviderError) as caught:
            await _provider(http_client).search(SearchRequest(query="q"))
    assert caught.value.error_type is error_type


async def test_blank_key_fails_before_http(
    http_client: httpx.AsyncClient,
) -> None:
    """Reject empty credentials before an upstream request can occur."""
    with respx.mock as router, pytest.raises(ProviderError) as caught:
        await _provider(http_client, "").search(SearchRequest(query="q"))
    assert caught.value.error_type is ErrorType.INVALID_INPUT
    assert len(router.calls) == 0


async def test_key_is_redacted_from_unexpected_error(
    http_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep a credential out of error text and logs after a bad transport."""

    async def boom(*args: object, **kwargs: object) -> object:
        raise ValueError(f"failure {_KEY}")

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("jasa.search.providers.base.http_json", boom)
    with pytest.raises(ProviderError) as caught:
        await _provider(http_client).search(SearchRequest(query="q"))
    assert _KEY not in str(caught.value)
    assert _KEY not in caplog.text

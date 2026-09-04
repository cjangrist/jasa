"""Keenable provider request, mapping, filters, cap, and error behavior."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.keenable import KeenableProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

KEENABLE_URL = "https://api.keenable.ai/v1/search"
_KEY = "keen-test-key"


async def test_exact_request_maps_results_and_requests_fifty(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "query": "search",
                    "results": [
                        {
                            "title": "Keenable",
                            "url": "https://keenable.ai/",
                            "description": "Search infrastructure.",
                            "snippet": "Query-relevant search infrastructure.",
                            "published_at": "2026-01-15T10:30:00Z",
                        }
                    ],
                },
            )
        )
        results = await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="search", limit=2)
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == KEENABLE_URL
    assert request.headers["x-api-key"] == _KEY
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "query": "search",
        "max_results": 50,
    }
    assert results == [
        SearchResult(
            title="Keenable",
            url="https://keenable.ai/",
            snippet="Query-relevant search infrastructure.",
            source_provider="keenable",
        )
    ]


async def test_native_site_and_date_filters_preserve_other_operators(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(
                query=(
                    "site:docs.example.com -site:private.example.com "
                    'after:2025-01-01 before:2026-01-31 "exact phrase"'
                )
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": '-site:private.example.com "exact phrase"',
        "max_results": 50,
        "site": "docs.example.com",
        "published_after": "2025-01-01",
        "published_before": "2026-01-31",
    }


async def test_multiple_include_domains_remain_in_query(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(
                query="site:third.example query",
                include_domains=("first.example", "second.example"),
                exclude_domains=("private.example",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert "site" not in body
    assert body["query"] == (
        "query site:first.example OR site:second.example OR "
        "site:third.example -site:private.example"
    )


async def test_repeated_identical_include_domain_uses_native_site(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(
                query="site:example.com site:example.com query",
                include_domains=("example.com",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "query",
        "max_results": 50,
        "site": "example.com",
    }


async def test_single_request_domain_uses_native_site(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="query", include_domains=("example.com",))
        )
        body = json.loads(route.calls.last.request.content)
    assert body["query"] == "query"
    assert body["site"] == "example.com"


@pytest.mark.parametrize(
    ("query", "expected_filters"),
    [
        ("site:example.com", {"site": "example.com"}),
        (
            "site:example.com after:2025-01-01",
            {
                "site": "example.com",
                "published_after": "2025-01-01",
            },
        ),
        ("after:2025-01-01", {"published_after": "2025-01-01"}),
    ],
)
async def test_operator_only_query_keeps_native_filters_with_wildcard(
    http_client: httpx.AsyncClient,
    query: str,
    expected_filters: dict[str, str],
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "*",
        "max_results": 50,
        **expected_filters,
    }


async def test_extended_date_formats_are_preserved_natively(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(
                query=("query after:1d before:2026-09-03T12:00:00.500-05:00")
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "query",
        "max_results": 50,
        "published_after": "1d",
        "published_before": "2026-09-03T12:00:00.500-05:00",
    }


@pytest.mark.parametrize(
    ("query", "expected_body"),
    [
        (
            '"before:2026-09-03T12:00:00Z"',
            {
                "query": '"before:2026-09-03T12:00:00Z"',
                "max_results": 50,
            },
        ),
        (
            'history "after:1d" before:2026-09-04',
            {
                "query": 'history "after:1d"',
                "max_results": 50,
                "published_before": "2026-09-04",
            },
        ),
        (
            '"site:example.com"',
            {"query": '"site:example.com"', "max_results": 50},
        ),
    ],
)
async def test_operator_syntax_inside_exact_phrases_remains_literal(
    http_client: httpx.AsyncClient,
    query: str,
    expected_body: dict[str, object],
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == expected_body


@pytest.mark.parametrize(
    ("query", "expected_body"),
    [
        (
            'release notes inbody:"changelog" inpage:"archive" '
            'site:example.com after:2026-01 "known issues"',
            {
                "query": (
                    "release notes inbody:changelog inpage:archive "
                    '"known issues"'
                ),
                "max_results": 50,
                "site": "example.com",
                "published_after": "2026-01-01",
            },
        ),
        (
            'query site:"example.com" -site:"private.example" intitle:"guide"',
            {
                "query": "query -site:private.example intitle:guide",
                "max_results": 50,
                "site": "example.com",
            },
        ),
        (
            'query +"needle" -"noise"',
            {"query": "query +needle -noise", "max_results": 50},
        ),
        (
            'query after:"2025"',
            {
                "query": "query",
                "max_results": 50,
                "published_after": "2025-01-01",
            },
        ),
    ],
)
async def test_quoted_operator_operands_remain_structural(
    http_client: httpx.AsyncClient,
    query: str,
    expected_body: dict[str, object],
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == expected_body


@pytest.mark.parametrize(
    ("query", "expected_query", "expected_field", "expected_value"),
    [
        (
            "query (before:2025)",
            "query ()",
            "published_before",
            "2025-12-31",
        ),
        (
            "query,after:2024-02",
            "query,",
            "published_after",
            "2024-02-01",
        ),
        (
            "query+after:2026-09-03T12:00:00Z",
            "query+",
            "published_after",
            "2026-09-03T12:00:00Z",
        ),
        (
            "query(after:1d)",
            "query()",
            "published_after",
            "1d",
        ),
    ],
)
async def test_punctuation_adjacent_dates_remain_native(
    http_client: httpx.AsyncClient,
    query: str,
    expected_query: str,
    expected_field: str,
    expected_value: str,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": expected_query,
        "max_results": 50,
        expected_field: expected_value,
    }


@pytest.mark.parametrize(
    ("query", "expected_field", "expected_value"),
    [
        ("query after:5min", "published_after", "5min"),
        ("query after:12mo", "published_after", "12mo"),
        ("query before:365d", "published_before", "365d"),
        ("query after:1440min", "published_after", "1440min"),
    ],
)
async def test_relative_date_lengths_are_not_calendar_dates(
    http_client: httpx.AsyncClient,
    query: str,
    expected_field: str,
    expected_value: str,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "query",
        "max_results": 50,
        expected_field: expected_value,
    }


@pytest.mark.parametrize(
    ("query", "expected_field", "expected_value"),
    [
        ("query after:2025", "published_after", "2025-01-01"),
        ("query before:2025", "published_before", "2025-12-31"),
        ("query after:2024-02", "published_after", "2024-02-01"),
        ("query before:2024-02", "published_before", "2024-02-29"),
    ],
)
async def test_partial_dates_expand_to_valid_inclusive_bounds(
    http_client: httpx.AsyncClient,
    query: str,
    expected_field: str,
    expected_value: str,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "query",
        "max_results": 50,
        expected_field: expected_value,
    }


async def test_invalid_partial_month_remains_for_vendor_validation(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="query before:2025-99")
        )
        body = json.loads(route.calls.last.request.content)
    assert body["published_before"] == "2025-99"


async def test_snippet_fallback_and_malformed_items(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        "invalid",
                        {"title": "missing url"},
                        {"url": 7},
                        {
                            "title": ["invalid"],
                            "url": "https://description.example",
                            "snippet": "",
                            "description": "Description fallback.",
                        },
                        {
                            "title": "No text",
                            "url": "https://empty.example",
                            "snippet": {"invalid": True},
                            "description": ["invalid"],
                        },
                    ]
                },
            )
        )
        results = await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="https://description.example",
            url="https://description.example",
            snippet="Description fallback.",
            source_provider="keenable",
        ),
        SearchResult(
            title="No text",
            url="https://empty.example",
            snippet="",
            source_provider="keenable",
        ),
    ]


@pytest.mark.parametrize("payload", [{}, [], {"results": "invalid"}])
async def test_missing_or_malformed_result_collection_is_empty_success(
    http_client: httpx.AsyncClient,
    payload: object,
) -> None:
    with respx.mock:
        respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json=payload)
        )
        results = await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, ErrorType.API_ERROR),
        (402, ErrorType.API_ERROR),
        (429, ErrorType.RATE_LIMIT),
        (500, ErrorType.PROVIDER_ERROR),
    ],
)
async def test_http_errors_use_shared_taxonomy(
    http_client: httpx.AsyncClient,
    status: int,
    error_type: ErrorType,
) -> None:
    with respx.mock:
        respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(status, json={"error": "failure"})
        )
        with pytest.raises(ProviderError) as exc:
            await KeenableProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is error_type
    assert exc.value.provider == "keenable"
    assert _KEY not in str(exc.value)


@pytest.mark.parametrize("key", ["", "   ", "''", '""'])
async def test_blank_key_is_invalid_input(
    http_client: httpx.AsyncClient,
    key: str,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await KeenableProvider(key, http_client).search(
            SearchRequest(query="q")
        )
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for keenable"


async def test_unexpected_error_is_redacted(
    http_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail(*args: object, **kwargs: object) -> object:
        raise ValueError(f"credential={_KEY}")

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("jasa.search.providers.base.http_json", fail)
    with pytest.raises(ProviderError) as exc:
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert "[REDACTED]" in str(exc.value)
    assert _KEY not in str(exc.value)
    assert _KEY not in caplog.text

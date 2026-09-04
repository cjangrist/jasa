"""Keenable provider request, mapping, filters, cap, and error behavior."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.keenable import KeenableProvider
from jasa.search.providers.keenable_query import KEENABLE_MAX_RESULTS
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

KEENABLE_URL = "https://api.keenable.ai/v1/search"
_KEY = "keen-test-key"


async def test_exact_request_maps_results_and_requests_fifty(
    http_client: httpx.AsyncClient,
) -> None:
    assert KEENABLE_MAX_RESULTS == 50
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
        "max_results": KEENABLE_MAX_RESULTS,
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
        "max_results": KEENABLE_MAX_RESULTS,
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
        "query (site:first.example OR site:second.example OR "
        "site:third.example) -site:private.example"
    )


@pytest.mark.parametrize(
    ("query", "expected_body"),
    [
        (
            "q (site:a.com OR site:b.com)",
            {
                "query": "q (site:a.com OR site:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q(site:a.com OR site:b.com) after:2025",
            {
                "query": "q (site:a.com OR site:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "2025-01-01",
            },
        ),
        (
            "q (site:a.com OR site:b.com) site:c.com",
            {
                "query": "q (site:a.com OR site:b.com OR site:c.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q (site:"a.com" OR site:"b.com")',
            {
                "query": "q (site:a.com OR site:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q site:a.com OR site:b.com",
            {
                "query": "q (site:a.com OR site:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
    ],
)
async def test_grouped_site_alternatives_have_one_boolean_scaffold(
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
    "query",
    [
        "q (site:a.com OR site:b.com) OR x",
        "q x AND (site:a.com OR site:b.com)",
        "q ((site:a.com OR site:b.com) OR x)",
        "q site:a.com OR site:b.com OR website:c.com",
        "q (site:a.com OR site:b.com) (site:c.com OR site:d.com)",
        "q (after:2025) OR x",
        "q x OR (site:a.com)",
        "q [intitle:x] OR y",
        "q y AND [filetype:pdf]",
    ],
)
async def test_filters_in_larger_boolean_expressions_remain_literal(
    http_client: httpx.AsyncClient,
    query: str,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {"query": query, "max_results": KEENABLE_MAX_RESULTS}


async def test_boolean_word_as_operator_value_does_not_hide_native_date(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="q intitle:OR after:2025")
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "q intitle:OR",
        "max_results": KEENABLE_MAX_RESULTS,
        "published_after": "2025-01-01",
    }


@pytest.mark.parametrize(
    ("query", "expected_body"),
    [
        (
            "q +site:a.com x",
            {
                "query": "q x",
                "max_results": KEENABLE_MAX_RESULTS,
                "site": "a.com",
            },
        ),
        (
            "q +after:2025 x",
            {
                "query": "q x",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "2025-01-01",
            },
        ),
        (
            "q +before:2026 +site:a.com x",
            {
                "query": "q x",
                "max_results": KEENABLE_MAX_RESULTS,
                "site": "a.com",
                "published_before": "2026-12-31",
            },
        ),
    ],
)
async def test_unary_plus_wrappers_are_consumed_with_native_filters(
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
        "max_results": KEENABLE_MAX_RESULTS,
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


async def test_non_domain_request_filter_remains_in_query(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="query", include_domains=("example.com/path",))
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "query site:example.com/path",
        "max_results": KEENABLE_MAX_RESULTS,
    }


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
        ("+after:2025", {"published_after": "2025-01-01"}),
        ("+site:example.com", {"site": "example.com"}),
        (
            "+after:2025 +before:2026",
            {
                "published_after": "2025-01-01",
                "published_before": "2026-12-31",
            },
        ),
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
        "max_results": KEENABLE_MAX_RESULTS,
        **expected_filters,
    }


@pytest.mark.parametrize("query", ["(after:2025)", "after:2025,"])
async def test_punctuation_wrapped_filter_only_query_uses_wildcard(
    http_client: httpx.AsyncClient,
    query: str,
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
        "max_results": KEENABLE_MAX_RESULTS,
        "published_after": "2025-01-01",
    }


@pytest.mark.parametrize("query", ["(site:example.com)", "site:example.com,"])
async def test_punctuation_wrapped_site_uses_clean_native_domain(
    http_client: httpx.AsyncClient,
    query: str,
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
        "max_results": KEENABLE_MAX_RESULTS,
        "site": "example.com",
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
        "max_results": KEENABLE_MAX_RESULTS,
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
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'history "after:1d" before:2026-09-04',
            {
                "query": 'history "after:1d"',
                "max_results": KEENABLE_MAX_RESULTS,
                "published_before": "2026-09-04",
            },
        ),
        (
            '"site:example.com"',
            {
                "query": '"site:example.com"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
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
                    'release notes inbody:"changelog" inpage:"archive" '
                    '"known issues"'
                ),
                "max_results": KEENABLE_MAX_RESULTS,
                "site": "example.com",
                "published_after": "2026-01-01",
            },
        ),
        (
            'query site:"example.com" -site:"private.example" intitle:"guide"',
            {
                "query": 'query -site:private.example intitle:"guide"',
                "max_results": KEENABLE_MAX_RESULTS,
                "site": "example.com",
            },
        ),
        (
            'query +"needle" -"noise"',
            {
                "query": 'query +"needle" -"noise"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query after:"2025"',
            {
                "query": "query",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "2025-01-01",
            },
        ),
        (
            'query intitle:"release notes" loc:"new york"',
            {
                "query": 'query intitle:"release notes" loc:"new york"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query intitle:"release\tnotes"',
            {
                "query": 'query intitle:"release\tnotes"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query intitle:"x:y:z"',
            {
                "query": 'query intitle:"x:y:z"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query +"machine learning" -"noise pollution"',
            {
                "query": ('query +"machine learning" -"noise pollution"'),
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            '(+"needle") [-"noise pollution"]',
            {
                "query": '() [] +"needle" -"noise pollution"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'website:"example.com"',
            {
                "query": 'website:"example.com"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'my-site:"example.com"',
            {
                "query": 'my-site:"example.com"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query inurl:/after:2025",
            {
                "query": "query inurl:/after:2025",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query inurl:.after:2025",
            {
                "query": "query inurl:.after:2025",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query inurl:after:2025",
            {
                "query": "query inurl:after:2025",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            '"look at site:" and "another phrase"',
            {
                "query": 'and "look at site:" "another phrase"',
                "max_results": KEENABLE_MAX_RESULTS,
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
    ("query", "expected_body"),
    [
        (
            "website:example.com news",
            {
                "query": "website:example.com news",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "quarterly report -after:2025",
            {
                "query": "quarterly report -after:2025",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'incident review before:"2025-1"',
            {
                "query": 'incident review before:"2025-1"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query inurl:?after:2025",
            {
                "query": "query inurl:?after:2025",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query custom:(before:2025)",
            {
                "query": "query custom:(before:2025)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "https://example.test/?after:2025",
            {
                "query": "https://example.test/?after:2025",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "example.test/path?before:1d",
            {
                "query": "example.test/path?before:1d",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "page?after:2025",
            {
                "query": "page?after:2025",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "foo.after:2025",
            {
                "query": "foo.after:2025",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'inurl:?after:"2025"',
            {
                "query": 'inurl:?after:"2025"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'custom:(before:"2025")',
            {
                "query": 'custom:(before:"2025")',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'https://example.test/?after:"2025"',
            {
                "query": 'https://example.test/?after:"2025"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'https://example.test/?q="foo"after:2025',
            {
                "query": 'https://example.test/?q="foo"after:2025',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'https://example.test/?q="foo"(after:2025)',
            {
                "query": 'https://example.test/?q="foo"(after:2025)',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'custom:"foo"[site:example.com]',
            {
                "query": 'custom:"foo"[site:example.com]',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'custom:"foo"(after:"2025")',
            {
                "query": 'custom:"foo"(after:"2025")',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'custom:"foo"site:example.com',
            {
                "query": 'custom:"foo"site:example.com',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "https://x/(site:a.com)(after:2025)",
            {
                "query": "https://x/(site:a.com)(after:2025)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "https://x/(a)(after:2025)",
            {
                "query": "https://x/(a)(after:2025)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "https://x/?a=(b)(site:b.com)",
            {
                "query": "https://x/?a=(b)(site:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "custom:(a)(after:2025)",
            {
                "query": "custom:(a)(after:2025)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "custom:[a][site:b.com]",
            {
                "query": "custom:[a][site:b.com]",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "custom:(site:a.com)(after:2025)",
            {
                "query": "custom:(site:a.com)(after:2025)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "custom:(site:a.com OR site:b.com)",
            {
                "query": "custom:(site:a.com OR site:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q (site:a.com/path OR site:b.com)",
            {
                "query": "q (site:a.com/path OR site:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q (site:a.com OR website:b.com)",
            {
                "query": "q (site:a.com OR website:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q site:a.com OR website:b.com",
            {
                "query": "q site:a.com OR website:b.com",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q website:a.com OR site:b.com",
            {
                "query": "q website:a.com OR site:b.com",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q (site:a.com AND site:b.com)",
            {
                "query": "q (site:a.com AND site:b.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q "look site:a.com OR site:b.com"',
            {
                "query": 'q "look site:a.com OR site:b.com"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q (site:a.com OR site:b.com)tail",
            {
                "query": "q (site:a.com OR site:b.com)tail",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'https://x/(+"needle")(after:2025)',
            {
                "query": 'https://x/(+"needle")(after:2025)',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'https://x/(site:a.com)(+"needle")',
            {
                "query": 'https://x/(site:a.com)(+"needle")',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'custom:(site:a.com)(-"needle")',
            {
                "query": 'custom:(site:a.com)(-"needle")',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q +"foo"after:2025',
            {
                "query": 'q +"foo"after:2025',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q -"foo"site:a.com',
            {
                "query": 'q -"foo"site:a.com',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'foo+"needle"',
            {
                "query": 'foo+"needle"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            '++"needle"',
            {
                "query": '++"needle"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'foo."first"(+"second")',
            {
                "query": 'foo."first"(+"second")',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q after:2025"x',
            {
                "query": 'q after:2025"x',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q site:a.com"x',
            {
                "query": 'q site:a.com"x',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q (intitle:x OR intitle:y)",
            {
                "query": "q (intitle:x OR intitle:y)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q intitle:"x" OR intitle:"y"',
            {
                "query": 'q intitle:"x" OR intitle:"y"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q after:2025 OR before:2026",
            {
                "query": "q after:2025 OR before:2026",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "custom:(site:example.com)",
            {
                "query": "custom:(site:example.com)",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query site:"my\tdomain.com"',
            {
                "query": 'query site:"my\tdomain.com"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "custom:filetype:pdf q",
            {
                "query": "custom:filetype:pdf q",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "relocation:paris q",
            {
                "query": "relocation:paris q",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q -foo.site:x.com",
            {
                "query": "q -foo.site:x.com",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "report -notes.site:example.com",
            {
                "query": "report -notes.site:example.com",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "docs +v1.intitle:guide",
            {
                "query": "docs +v1.intitle:guide",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "site:exa[mple].com q",
            {
                "query": "site:exa[mple].com q",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "site:example.com/path q",
            {
                "query": "site:example.com/path q",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'site:"example.com/path" q',
            {
                "query": 'site:"example.com/path" q',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q intitle:val)ue",
            {
                "query": "q intitle:val)ue",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q -after:"2025"',
            {
                "query": 'q -after:"2025"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "q -filetype:pdf",
            {
                "query": "q -filetype:pdf",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'q -location:"new york"',
            {
                "query": 'q -location:"new york"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query before:"2025-01-01T12:00:00Z"suffix',
            {
                "query": 'query before:"2025-01-01T12:00:00Z"suffix',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query -site:"my domain.com"',
            {
                "query": 'query -site:"my domain.com"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query site:"my domain.com" site:other.com',
            {
                "query": 'query site:"my domain.com"',
                "max_results": KEENABLE_MAX_RESULTS,
                "site": "other.com",
            },
        ),
    ],
)
async def test_ambiguous_filter_syntax_remains_literal(
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
            'query after:2025 after:"2024"',
            {
                "query": "query",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "2024-01-01",
            },
        ),
        (
            'query after:"2024" after:2025',
            {
                "query": "query",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "2025-01-01",
            },
        ),
        (
            'query before:2025 before:"2024"',
            {
                "query": "query",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_before": "2024-12-31",
            },
        ),
        (
            'query before:"2024" before:2025',
            {
                "query": "query",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_before": "2025-12-31",
            },
        ),
        (
            'query intitle:"first value" intitle:second',
            {
                "query": "query intitle:second",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query intitle:first intitle:"second value"',
            {
                "query": 'query intitle:"second value"',
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query ext:"docx" filetype:pdf',
            {
                "query": "query filetype:pdf",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query ext:docx filetype:pdf",
            {
                "query": "query filetype:pdf",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query filetype:pdf ext:docx",
            {
                "query": "query filetype:docx",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query language:en lang:fr",
            {
                "query": "query lang:fr",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            "query location:paris loc:london",
            {
                "query": "query loc:london",
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
        (
            'query site:"first.example" site:second.example',
            {
                "query": ("query (site:first.example OR site:second.example)"),
                "max_results": KEENABLE_MAX_RESULTS,
            },
        ),
    ],
)
async def test_repeated_single_value_operators_keep_source_order(
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


async def test_consumed_operator_segment_does_not_add_duplicate_whitespace(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query='a "exact" +foo "other" b')
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": 'a b "exact" "other" +foo',
        "max_results": KEENABLE_MAX_RESULTS,
    }


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
        "max_results": KEENABLE_MAX_RESULTS,
        expected_field: expected_value,
    }


@pytest.mark.parametrize(
    ("query", "expected_body"),
    [
        (
            "query after:1d,before:2d",
            {
                "query": "query ,",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "1d",
                "published_before": "2d",
            },
        ),
        (
            ("query after:2026-01-01T00:00:00Z,before:2026-09-03T12:00:00Z"),
            {
                "query": "query ,",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "2026-01-01T00:00:00Z",
                "published_before": "2026-09-03T12:00:00Z",
            },
        ),
        (
            "query intitle:guide\nafter:7d",
            {
                "query": "query intitle:guide",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "7d",
            },
        ),
        (
            "q site:a.com after:2025 x",
            {
                "query": "q x",
                "max_results": KEENABLE_MAX_RESULTS,
                "site": "a.com",
                "published_after": "2025-01-01",
            },
        ),
    ],
)
async def test_adjacent_dates_are_partitioned_independently(
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
            "(after:1d)(before:2d)",
            {
                "query": "*",
                "max_results": KEENABLE_MAX_RESULTS,
                "published_after": "1d",
                "published_before": "2d",
            },
        ),
        (
            "(site:example.com)(after:2025)",
            {
                "query": "*",
                "max_results": KEENABLE_MAX_RESULTS,
                "site": "example.com",
                "published_after": "2025-01-01",
            },
        ),
    ],
)
async def test_glued_wrapped_filters_are_partitioned_independently(
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


async def test_native_date_does_not_consume_adjacent_operator_tail(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query="q intitle:guide,after:2025,notes")
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": "q ,,notes intitle:guide",
        "max_results": KEENABLE_MAX_RESULTS,
        "published_after": "2025-01-01",
    }


@pytest.mark.parametrize(
    "query",
    [
        "query after:2025-1",
        "query after:2025-01-01T12:00",
        "query after:2025-01-01T12:00:00.123invalid",
        "query xafter:2025",
        "query my-after:2025",
        "query custom:after:2025",
        "q after:2025?foo=bar",
        "q before:2025&next=value",
        'q "unterminated after:2025',
        'q "unterminated site:example.com before:2026',
    ],
)
async def test_invalid_or_nested_dates_remain_literal(
    http_client: httpx.AsyncClient,
    query: str,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {"query": query, "max_results": KEENABLE_MAX_RESULTS}


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
        "max_results": KEENABLE_MAX_RESULTS,
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
        "max_results": KEENABLE_MAX_RESULTS,
        expected_field: expected_value,
    }


@pytest.mark.parametrize(
    "query",
    [
        "query before:2025-99",
        "query after:2025-02-30",
        "query before:2021-02-29",
        "query after:0000",
        "query after:0d",
        "query after:0min",
        "query after:1969",
        "query before:2149-06",
        "query after:2149-06-06",
        "query after:2150",
        'query after:"2025-02-30"',
        "query after:2025-01-01T25:00:00Z",
        "query after:2025-01-01T12:00:00+24:00",
        "query after:2025-01-01T12:00:00+01:60",
        "query after:2025-01-01T12:00:00-00:60",
        "query after:1970-01-01T00:00:00+00:01",
        "query before:2149-06-05T23:59:59-00:01",
    ],
)
async def test_invalid_provider_date_bounds_remain_literal(
    http_client: httpx.AsyncClient,
    query: str,
) -> None:
    with respx.mock:
        route = respx.post(KEENABLE_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await KeenableProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {"query": query, "max_results": KEENABLE_MAX_RESULTS}


@pytest.mark.parametrize(
    ("query", "expected_field", "expected_value"),
    [
        ("query after:1970", "published_after", "1970-01-01"),
        ("query before:2149-06-05", "published_before", "2149-06-05"),
        (
            "query after:1970-01-01T00:00:00Z",
            "published_after",
            "1970-01-01T00:00:00Z",
        ),
        (
            "query after:2025-01-01T12:00:00",
            "published_after",
            "2025-01-01T12:00:00",
        ),
        (
            "query before:2149-06-05T23:59:59Z",
            "published_before",
            "2149-06-05T23:59:59Z",
        ),
    ],
)
async def test_provider_date_window_boundaries_remain_native(
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
        "max_results": KEENABLE_MAX_RESULTS,
        expected_field: expected_value,
    }


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

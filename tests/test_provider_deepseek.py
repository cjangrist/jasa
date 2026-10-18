"""DeepSeek native search request, mapping, limits, and error contracts."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from jasa.search.providers import load_search_providers
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.deepseek import DeepseekProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_URL = "https://api.deepseek.com/v1/tools/search"
_KEY = "deepseek-test-key"


async def test_exact_request_and_native_mapping(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(_URL).respond(
            200,
            json={
                "results": [
                    {
                        "title": "Title",
                        "url": "https://example.com/page",
                        "summary": "Long source summary",
                        "snippet": "Short snippet",
                        "content": "Full page must not be used",
                        "rerank_score": 2.4,
                    }
                ]
            },
        )
        results = await DeepseekProvider(f'"{_KEY}"', http_client).search(
            SearchRequest(
                query=(
                    "DeepSeek site:github.com OR site:huggingface.co lang:zh,en"
                ),
                limit=30,
            )
        )
    request = route.calls.last.request
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert request.extensions["timeout"]["read"] == 20.0
    assert json.loads(request.content) == {
        "query": "DeepSeek site:github.com OR site:huggingface.co lang:zh,en",
        "count": 30,
        "need_content": False,
    }
    assert results == [
        SearchResult(
            title="Title",
            url="https://example.com/page",
            snippet="Long source summary",
            source_provider="deepseek",
            score=2.4,
        )
    ]


@pytest.mark.parametrize(
    ("limit", "count"),
    [(-10, 1), (0, 1), (1, 1), (30, 30), (50, 50), (100, 50)],
)
async def test_count_clamped(
    http_client: httpx.AsyncClient, limit: int, count: int
) -> None:
    with respx.mock:
        route = respx.post(_URL).respond(200, json={"results": []})
        await DeepseekProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=limit)
        )
    assert json.loads(route.calls.last.request.content)["count"] == count


async def test_explicit_domains_append_without_rewriting_operators(
    http_client: httpx.AsyncClient,
) -> None:
    query = (
        '"exact phrase" -"bad phrase" -word lang:zh,en after:2025 before:2026'
    )
    with respx.mock:
        route = respx.post(_URL).respond(200, json={"results": []})
        await DeepseekProvider(_KEY, http_client).search(
            SearchRequest(
                query=query,
                include_domains=("a.test", "b.test"),
                exclude_domains=("c.test", "d.test"),
            )
        )
    assert json.loads(route.calls.last.request.content)["query"] == (
        query + " site:a.test OR site:b.test -site:c.test -site:d.test"
    )


@pytest.mark.parametrize("query", ["q", "中" * 128, "😀" * 128, "  q  "])
async def test_valid_query_preserved_including_unicode(
    http_client: httpx.AsyncClient, query: str
) -> None:
    with respx.mock:
        route = respx.post(_URL).respond(200, json={"results": []})
        await DeepseekProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
    assert json.loads(route.calls.last.request.content)["query"] == query


@pytest.mark.parametrize("query", ["", " \t\n", "x" * 129, "😀" * 129])
async def test_invalid_query_fails_before_http(
    http_client: httpx.AsyncClient, query: str
) -> None:
    with respx.mock as router, pytest.raises(ProviderError) as exc:
        await DeepseekProvider(_KEY, http_client).search(
            SearchRequest(query=query)
        )
    assert not router.calls
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert exc.value.provider == "deepseek"
    assert "128 Unicode" in str(exc.value)


async def test_length_includes_explicit_domain_filters(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock as router, pytest.raises(ProviderError) as exc:
        await DeepseekProvider(_KEY, http_client).search(
            SearchRequest(
                query="中" * 120,
                include_domains=("example.com",),
            )
        )
    assert not router.calls
    assert exc.value.error_type is ErrorType.INVALID_INPUT


@pytest.mark.parametrize(
    "payload", [{}, [], {"results": None}, {"results": "bad"}, {"results": []}]
)
async def test_empty_or_missing_collection(
    http_client: httpx.AsyncClient, payload: object
) -> None:
    with respx.mock:
        respx.post(_URL).respond(200, json=payload)
        assert (
            await DeepseekProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
            == []
        )


async def test_malformed_rows_and_field_fallbacks(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(_URL).respond(
            200,
            json={
                "results": [
                    None,
                    "bad",
                    {},
                    {"url": 7},
                    {"url": ""},
                    {
                        "url": "https://example.com/1",
                        "title": [],
                        "summary": None,
                        "snippet": "Fallback",
                    },
                    {
                        "url": "https://example.com/2",
                        "title": "Second",
                        "summary": "",
                        "snippet": "Second snippet",
                    },
                    {
                        "url": "https://example.com/3",
                        "summary": {},
                        "snippet": [],
                        "content": "Not a snippet",
                    },
                ]
            },
        )
        results = await DeepseekProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [(r.title, r.url, r.snippet, r.score) for r in results] == [
        ("https://example.com/1", "https://example.com/1", "Fallback", None),
        ("Second", "https://example.com/2", "Second snippet", None),
        ("https://example.com/3", "https://example.com/3", "", None),
    ]


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, 0.0),
        (3, 3.0),
        (1.5, 1.5),
        (True, None),
        (False, None),
        ("2.4", None),
        (None, None),
        (-1, None),
        (4, None),
        ([], None),
        (float("nan"), None),
        (float("inf"), None),
        (-float("inf"), None),
    ],
)
async def test_only_finite_native_scores(
    http_client: httpx.AsyncClient, score: object, expected: float | None
) -> None:
    # Use raw JSON because strict encoders reject non-finite JSON extensions.
    body = json.dumps(
        {"results": [{"url": "https://example.com", "rerank_score": score}]}
    )
    with respx.mock:
        respx.post(_URL).respond(
            200, text=body, headers={"content-type": "application/json"}
        )
        results = await DeepseekProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results[0].score == expected


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, ErrorType.API_ERROR),
        (401, ErrorType.API_ERROR),
        (402, ErrorType.API_ERROR),
        (429, ErrorType.RATE_LIMIT),
        (500, ErrorType.PROVIDER_ERROR),
        (504, ErrorType.PROVIDER_ERROR),
    ],
)
async def test_shared_error_taxonomy(
    http_client: httpx.AsyncClient, status: int, kind: ErrorType
) -> None:
    with respx.mock:
        respx.post(_URL).respond(status, json={"error": {"message": "failure"}})
        with pytest.raises(ProviderError) as exc:
            await DeepseekProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is kind
    assert exc.value.provider == "deepseek"


@pytest.mark.parametrize("key", ["", "   ", "''", '""'])
async def test_missing_key_before_http(
    http_client: httpx.AsyncClient, key: str
) -> None:
    with respx.mock as router, pytest.raises(ProviderError) as exc:
        await DeepseekProvider(key, http_client).search(
            SearchRequest(query="q")
        )
    assert not router.calls
    assert exc.value.error_type is ErrorType.INVALID_INPUT


async def test_credential_redaction(
    http_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail(*args: object, **kwargs: object) -> object:
        raise ValueError(f"credential={_KEY}")

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("jasa.search.providers.base.http_json", fail)
    with pytest.raises(ProviderError) as exc:
        await DeepseekProvider(f'"{_KEY}"', http_client).search(
            SearchRequest(query="q")
        )
    assert "[REDACTED]" in str(exc.value)
    assert _KEY not in str(exc.value)
    assert _KEY not in caplog.text


async def test_registry_loads_deepseek_only_with_its_key(
    http_client: httpx.AsyncClient,
) -> None:
    assert (
        load_search_providers(ProviderSecrets.from_env({}), http_client) == {}
    )
    active = load_search_providers(
        ProviderSecrets.from_env({"DEEPSEEK_API_KEY": _KEY}), http_client
    )
    assert list(active) == ["deepseek"]
    assert isinstance(active["deepseek"], DeepseekProvider)
    assert active["deepseek"].model_id() is None

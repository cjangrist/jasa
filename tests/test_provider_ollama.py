"""Ollama provider request, mapping, cap, operators, and error behavior."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.ollama import OllamaProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

OLLAMA_URL = "https://ollama.com/api/web_search"
_KEY = "ollama-test-key"


async def test_exact_request_maps_results(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Ollama",
                            "url": "https://ollama.com/",
                            "content": "Cloud models are available.",
                        }
                    ]
                },
            )
        )
        results = await OllamaProvider(_KEY, http_client).search(
            SearchRequest(query="what is ollama?", limit=2)
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == OLLAMA_URL
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "query": "what is ollama?",
        "max_results": 10,
    }
    assert results == [
        SearchResult(
            title="Ollama",
            url="https://ollama.com/",
            snippet="Cloud models are available.",
            source_provider="ollama",
        )
    ]


async def test_max_results_is_always_ten_and_operators_are_rendered(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await OllamaProvider(_KEY, http_client).search(
            SearchRequest(
                query='"exact phrase" after:2025-01-01',
                limit=50,
                include_domains=("docs.ollama.com",),
                exclude_domains=("example.com",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body == {
        "query": (
            " site:docs.ollama.com -site:example.com after:2025-01-01 "
            '"exact phrase"'
        ),
        "max_results": 10,
    }


async def test_zero_limit_still_requests_ten(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        await OllamaProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=0)
        )
    assert json.loads(route.calls.last.request.content)["max_results"] == 10


@pytest.mark.parametrize("payload", [{}, [], {"results": "invalid"}])
async def test_missing_or_malformed_result_collection_is_empty_success(
    http_client: httpx.AsyncClient,
    payload: object,
) -> None:
    with respx.mock:
        respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(200, json=payload)
        )
        results = await OllamaProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_malformed_items_and_fields_are_ignored_or_sanitized(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        "not an object",
                        {"url": 7},
                        {"title": "no url"},
                        {
                            "title": ["invalid"],
                            "url": "https://example.com",
                            "content": {"invalid": True},
                        },
                    ]
                },
            )
        )
        results = await OllamaProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="https://example.com",
            url="https://example.com",
            snippet="",
            source_provider="ollama",
        )
    ]


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, ErrorType.API_ERROR),
        (403, ErrorType.API_ERROR),
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
        respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(status, json={"error": "failure"})
        )
        with pytest.raises(ProviderError) as exc:
            await OllamaProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is error_type
    assert exc.value.provider == "ollama"
    assert _KEY not in str(exc.value)


@pytest.mark.parametrize("key", ["", "   ", "''", '""'])
async def test_blank_key_is_invalid_input(
    http_client: httpx.AsyncClient,
    key: str,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await OllamaProvider(key, http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for ollama"


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
        await OllamaProvider(_KEY, http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.API_ERROR
    assert "[REDACTED]" in str(exc.value)
    assert _KEY not in str(exc.value)
    assert _KEY not in caplog.text

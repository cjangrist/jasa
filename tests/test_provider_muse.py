"""Muse search requests, source mapping, and failure boundaries."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx
from fastmcp import Client

from jasa.search.providers import load_search_providers
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.muse import MuseProvider
from jasa.search.ranking import SearchResult
from jasa.server import build_composition_async
from omnifetch.fetch.shared.config import ProviderSecrets
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_URL = "https://api.meta.ai/v1/responses"
_KEY = "muse-test-key"


def _hit(
    url: object = "https://example.com", **fields: object
) -> dict[str, object]:
    return {
        "type": "text_result",
        "url": url,
        "title": "Source",
        "snippet": "Page excerpt",
        **fields,
    }


def _search(*hits: object, status: str = "completed") -> dict[str, object]:
    return {"type": "web_search_call", "status": status, "results": list(hits)}


def _message(*annotations: object) -> dict[str, object]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Model prose must not be a source excerpt",
                "annotations": list(annotations),
            }
        ],
    }


async def _run(
    http_client: httpx.AsyncClient, payload: object, limit: int = 30
) -> list[SearchResult]:
    with respx.mock:
        respx.post(_URL).respond(200, content=json.dumps(payload))
        return await MuseProvider(_KEY, http_client).search(
            SearchRequest("q", limit=limit)
        )


async def test_exact_request_and_source_snippet(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(_URL).respond(
            200,
            json={
                "status": "completed",
                "output": [{"type": "reasoning"}, _search(_hit())],
            },
        )
        results = await MuseProvider(_KEY, http_client).search(
            SearchRequest("hello world")
        )
    request = route.calls.last.request
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert request.extensions["timeout"]["read"] == 60.0
    assert json.loads(request.content) == {
        "model": "muse-spark-1.3-contributor",
        "input": "Use the web_search tool to search the web for: hello world",
        "tools": [{"type": "web_search", "search_context_size": "medium"}],
        "include": ["web_search_call.results"],
    }
    assert results == [
        SearchResult("Source", "https://example.com", "Page excerpt", "muse")
    ]


async def test_operators_and_domains_remain_in_query(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(_URL).respond(200, json={})
        await MuseProvider(_KEY, http_client).search(
            SearchRequest(
                query="site:b.com filetype:pdf foo -site:c.com after:2026",
                include_domains=("a.com",),
                exclude_domains=("d.com",),
            )
        )
    body = json.loads(route.calls.last.request.content)
    assert body["input"] == (
        "Use the web_search tool to search the web for: "
        "foo site:a.com OR site:b.com -site:d.com -site:c.com "
        "filetype:pdf after:2026"
    )
    assert body["tools"] == [
        {"type": "web_search", "search_context_size": "medium"}
    ]


async def test_registry_gates_secret_and_passes_settings(
    http_client: httpx.AsyncClient,
) -> None:
    settings = {
        "MUSE_BASE_URL": "https://gateway.example/v1/",
        "MUSE_SEARCH_MODEL": "available-model",
    }
    assert (
        load_search_providers(ProviderSecrets.from_env(settings), http_client)
        == {}
    )
    active = load_search_providers(
        ProviderSecrets.from_env({**settings, "MODEL_API_KEY": _KEY}),
        http_client,
    )
    assert list(active) == ["muse"]
    with respx.mock:
        route = respx.post("https://gateway.example/v1/responses").respond(
            200, json={}
        )
        await active["muse"].search(SearchRequest("q"))
    assert (
        json.loads(route.calls.last.request.content)["model"]
        == "available-model"
    )


async def test_raw_hits_precede_citations_and_deduplicate_across_searches(
    http_client: httpx.AsyncClient,
) -> None:
    citation = {
        "type": "url_citation",
        "url": "https://example.com",
        "title": "Model title",
        "start_index": 0,
        "end_index": 10,
    }
    results = await _run(
        http_client,
        {
            "output": [
                _message(citation, {**citation, "url": "https://citation.com"}),
                _search(_hit(), _hit("https://second.com", title="Second")),
                _search(
                    _hit(), _hit("https://third.com", title=None, snippet=None)
                ),
                _message(citation, {**citation, "url": "https://citation.com"}),
            ]
        },
    )
    assert [(row.title, row.url, row.snippet) for row in results] == [
        ("Source", "https://example.com", "Page excerpt"),
        ("Second", "https://second.com", "Page excerpt"),
        ("https://third.com", "https://third.com", ""),
        ("Model title", "https://citation.com", ""),
    ]
    assert all(
        row.source_provider == "muse" and row.score is None for row in results
    )


async def test_citation_fallback_ignores_spans_and_model_prose(
    http_client: httpx.AsyncClient,
) -> None:
    results = await _run(
        http_client,
        {
            "output": [
                _message(
                    {
                        "type": "url_citation",
                        "url": "https://a.com",
                        "start_index": 0,
                        "end_index": 10,
                    },
                    {"type": "url_citation", "url": "https://a.com"},
                    {"type": "file_citation", "url": "https://ignored.com"},
                )
            ]
        },
    )
    assert results == [
        SearchResult("https://a.com", "https://a.com", "", "muse")
    ]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "invalid",
        {},
        {"output": None},
        {"output": "invalid"},
        {"status": "completed", "error": "", "output": [_message()]},
        {"output": [_search()]},
    ],
)
async def test_empty_or_missing_results_contribute_no_rows(
    http_client: httpx.AsyncClient, payload: object
) -> None:
    assert await _run(http_client, payload) == []


async def test_malformed_nested_data_is_ignored(
    http_client: httpx.AsyncClient,
) -> None:
    output = [
        None,
        "bad",
        {"type": "reasoning", "results": [_hit()]},
        {"type": "web_search_call", "results": "invalid"},
        _search(
            None,
            3,
            _hit([], title=[]),
            _hit("", snippet=[]),
            _hit(type="image_result"),
            _hit("https://valid.com", title={}, snippet=[]),
        ),
        {
            "type": "message",
            "content": [
                None,
                {
                    "type": "refusal",
                    "annotations": [
                        {"type": "url_citation", "url": "https://ignored.com"}
                    ],
                },
            ],
        },
        {"type": "message", "content": "invalid"},
        {
            "type": "message",
            "content": [{"type": "output_text", "annotations": None}],
        },
        _message(
            None,
            {"type": "url_citation", "url": []},
            {"type": "url_citation", "url": "https://citation.com", "title": 7},
        ),
    ]
    assert await _run(http_client, {"status": [], "output": output}) == [
        SearchResult("https://valid.com", "https://valid.com", "", "muse"),
        SearchResult(
            "https://citation.com", "https://citation.com", "", "muse"
        ),
    ]


@pytest.mark.parametrize(
    ("limit", "expected_count"), [(0, 30), (1, 1), (40, 35)]
)
async def test_limit_applies_after_deduplication(
    http_client: httpx.AsyncClient, limit: int, expected_count: int
) -> None:
    results = await _run(
        http_client,
        {
            "output": [
                _search(
                    *(
                        _hit(f"https://example.com/{index}")
                        for index in range(35)
                    ),
                    _hit("https://example.com/0"),
                )
            ]
        },
        limit,
    )
    assert len(results) == expected_count
    assert results[-1].url == f"https://example.com/{expected_count - 1}"


@pytest.mark.parametrize("status", ["incomplete", "in_progress", "queued"])
async def test_unfinished_response_without_results_is_transient(
    http_client: httpx.AsyncClient, status: str
) -> None:
    with pytest.raises(ProviderError) as caught:
        await _run(http_client, {"status": status, "output": [_search()]})
    assert caught.value.error_type is ErrorType.PROVIDER_ERROR
    assert caught.value.provider == "muse"


async def test_failed_search_without_hits_is_transient(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as caught:
        await _run(
            http_client,
            {"status": "completed", "output": [_search(status="failed")]},
        )
    assert caught.value.error_type is ErrorType.PROVIDER_ERROR


async def test_completed_search_hits_survive_late_tool_failure_and_token_limit(
    http_client: httpx.AsyncClient,
) -> None:
    results = await _run(
        http_client,
        {
            "status": "incomplete",
            "output": [_search(_hit()), _search(status="failed")],
        },
    )
    assert [row.url for row in results] == ["https://example.com"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "failed"}, "Muse web search failed"),
        ({"status": "cancelled"}, "Muse web search failed"),
        ({"error": {}}, "Muse web search failed"),
        ({"error": "bad request"}, "bad request"),
        ({"error": {"message": 42}}, "Muse web search failed"),
        ({"error": {"message": "bad request", "code": []}}, "bad request"),
    ],
)
async def test_body_errors_win_over_results(
    http_client: httpx.AsyncClient, payload: dict[str, object], expected: str
) -> None:
    with pytest.raises(ProviderError) as caught:
        await _run(http_client, {**payload, "output": [_search(_hit())]})
    assert caught.value.error_type is ErrorType.API_ERROR
    assert str(caught.value) == expected
    assert caught.value.provider == "muse"


@pytest.mark.parametrize(
    ("field", "marker"),
    [("code", "rate_limit_exceeded"), ("type", "rate_limit_error")],
)
async def test_body_rate_limit_uses_shared_category(
    http_client: httpx.AsyncClient, field: str, marker: str
) -> None:
    with pytest.raises(ProviderError) as caught:
        await _run(
            http_client, {"error": {field: marker, "message": "slow down"}}
        )
    assert caught.value.error_type is ErrorType.RATE_LIMIT


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ErrorType.API_ERROR),
        (402, ErrorType.API_ERROR),
        (404, ErrorType.API_ERROR),
        (429, ErrorType.RATE_LIMIT),
        (500, ErrorType.PROVIDER_ERROR),
    ],
)
async def test_http_failures_use_shared_taxonomy(
    http_client: httpx.AsyncClient, status: int, expected: ErrorType
) -> None:
    with respx.mock:
        respx.post(_URL).respond(status, json={})
        with pytest.raises(ProviderError) as caught:
            await MuseProvider(_KEY, http_client).search(SearchRequest("q"))
    assert caught.value.error_type is expected
    assert caught.value.provider == "muse"


async def test_transport_timeout_is_attributed(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
        with pytest.raises(ProviderError) as caught:
            await MuseProvider(_KEY, http_client).search(SearchRequest("q"))
    assert caught.value.error_type is ErrorType.PROVIDER_ERROR
    assert caught.value.provider == "muse"


@pytest.mark.parametrize("key", ["", " ", '""', "''"])
async def test_blank_keys_fail_before_http(
    http_client: httpx.AsyncClient, key: str
) -> None:
    with respx.mock, pytest.raises(ProviderError) as caught:
        await MuseProvider(key, http_client).search(SearchRequest("q"))
    assert caught.value.error_type is ErrorType.INVALID_INPUT


async def test_quoted_key_is_normalized_and_redacted(
    http_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with respx.mock:
        route = respx.post(_URL).respond(
            200, json={"error": {"message": f"invalid {_KEY}"}}
        )
        with pytest.raises(ProviderError) as caught:
            await MuseProvider(f'"{_KEY}"', http_client).search(
                SearchRequest("q")
            )
    assert route.calls.last.request.headers["authorization"] == f"Bearer {_KEY}"
    assert _KEY not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert _KEY not in caplog.text


async def test_muse_results_flow_through_mcp_rest_and_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", _KEY)
    composition = await build_composition_async()
    snippet = (
        "A retrieved source excerpt long enough to survive "
        "Jasa's quality filter."
    )
    with respx.mock:
        route = respx.post(_URL).respond(
            200,
            json={
                "status": "completed",
                "output": [_search(_hit(snippet=snippet))],
            },
        )
        async with Client(composition.server) as client:
            result = await client.call_tool("web_search", {"query": "q"})
            payload = result.structured_content
            assert payload is not None
            assert payload["providers_failed"] == []
            assert payload["providers_succeeded"][0]["provider"] == "muse"
            assert payload["grounding"]["attempted"] == 0
            assert payload["web_results"][0]["snippet_source"] == "aggregated"
            assert payload["web_results"][0]["snippets"] == [snippet]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=composition.server.http_app()
                ),
                base_url="http://test",
            ) as rest:
                health = await rest.get("/health")
                assert health.json()["search"]["providers"] == ["muse"]
                response = await rest.post("/search", json={"query": "q"})
                assert response.status_code == 200
                assert response.json() == [
                    {
                        "title": "Source",
                        "link": "https://example.com",
                        "snippet": snippet,
                    }
                ]
        assert route.call_count == 1

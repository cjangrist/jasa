"""Claude provider request, operator, mapping, settings, and error behavior."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.claude import ClaudeProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

CLAUDE_URL = "https://ai.angrist.net/v1/messages"
VENDOR_URL = "https://api.anthropic.com/v1/messages"
GATEWAY_URL = "https://gateway.example/v1/messages"
_KEY = "claude-test-key"


def _result_block(results: list[Any]) -> dict[str, object]:
    return {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": results,
    }


def _hit(url: str, title: str = "T") -> dict[str, object]:
    return {
        "type": "web_search_result",
        "title": title,
        "url": url,
        "encrypted_content": "opaque",
        "page_age": "2 hours ago",
    }


def _citation(url: str, cited_text: str) -> dict[str, object]:
    return {
        "type": "web_search_result_location",
        "url": url,
        "title": "T",
        "cited_text": cited_text,
        "encrypted_index": "opaque",
    }


def _ok(content: list[Any]) -> httpx.Response:
    return httpx.Response(200, json={"type": "message", "content": content})


async def test_exact_outbound_request_and_mapping(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CLAUDE_URL).mock(
            return_value=_ok(
                [
                    {"type": "server_tool_use", "name": "web_search"},
                    _result_block([_hit("https://x.com")]),
                    {
                        "type": "text",
                        "text": "x says so",
                        "citations": [
                            _citation("https://x.com", "the cited excerpt")
                        ],
                    },
                ]
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="hello world", limit=7)
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert request.headers["x-api-key"] == _KEY
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(request.content)
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["max_tokens"] == 2048
    assert body["system"].startswith("You are a web-search tool.")
    assert body["messages"] == [
        {"role": "user", "content": "Search the web for: hello world"}
    ]
    assert body["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 1,
        }
    ]
    assert results == [
        SearchResult(
            title="T",
            url="https://x.com",
            snippet="the cited excerpt",
            source_provider="claude",
        )
    ]


async def test_settings_override_endpoint_and_model(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(GATEWAY_URL).mock(return_value=_ok([]))
        await ClaudeProvider(
            _KEY,
            http_client,
            {
                "ANTHROPIC_BASE_URL": "https://gateway.example/",
                "CLAUDE_SEARCH_MODEL": "gateway-model",
            },
        ).search(SearchRequest(query="q"))
        request = route.calls.last.request
    assert str(request.url) == GATEWAY_URL
    assert json.loads(request.content)["model"] == "gateway-model"


async def test_settings_retarget_the_vendor_endpoint(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(VENDOR_URL).mock(return_value=_ok([]))
        await ClaudeProvider(
            _KEY,
            http_client,
            {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
        ).search(SearchRequest(query="q"))
        request = route.calls.last.request
    assert str(request.url) == VENDOR_URL
    assert json.loads(request.content)["model"] == "claude-haiku-4-5-20251001"


async def test_blank_settings_fall_back_to_defaults(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CLAUDE_URL).mock(return_value=_ok([]))
        await ClaudeProvider(
            _KEY,
            http_client,
            {"ANTHROPIC_BASE_URL": "", "CLAUDE_SEARCH_MODEL": ""},
        ).search(SearchRequest(query="q"))
        request = route.calls.last.request
    assert str(request.url) == CLAUDE_URL
    assert json.loads(request.content)["model"] == "claude-haiku-4-5-20251001"


async def test_include_domains_become_allowed_and_exclusions_stay_in_query(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CLAUDE_URL).mock(return_value=_ok([]))
        await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(
                query="site:b.com filetype:pdf foo -site:c.com",
                include_domains=("a.com",),
                exclude_domains=("d.com",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["allowed_domains"] == ["a.com", "b.com"]
    assert "blocked_domains" not in body["tools"][0]
    assert body["messages"][0]["content"] == (
        "Search the web for: foo -site:d.com -site:c.com filetype:pdf"
    )


async def test_exclude_only_domains_become_blocked_list(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CLAUDE_URL).mock(return_value=_ok([]))
        await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="foo -site:c.com", exclude_domains=("d.com",))
        )
        body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["blocked_domains"] == ["d.com", "c.com"]
    assert "allowed_domains" not in body["tools"][0]
    assert body["messages"][0]["content"] == "Search the web for: foo"


async def test_duplicate_urls_and_excerpts_collapse_in_rank_order(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=_ok(
                [
                    _result_block(
                        [
                            _hit("https://a.com", "A"),
                            {"type": "web_search_result", "title": "no url"},
                            _hit("https://b.com", "B"),
                        ]
                    ),
                    _result_block([_hit("https://a.com", "A again")]),
                    {
                        "type": "text",
                        "text": "prose",
                        "citations": [
                            _citation("https://a.com", "first"),
                            _citation("https://a.com", "first"),
                            _citation("https://a.com", "second"),
                            {"type": "char_location", "url": "https://b.com"},
                            {"type": "web_search_result_location"},
                        ],
                    },
                    {"type": "text", "text": "uncited"},
                ]
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [(result.title, result.url) for result in results] == [
        ("A", "https://a.com"),
        ("B", "https://b.com"),
    ]
    assert results[0].snippet == "first second"
    assert results[1].snippet == ""


async def test_untitled_hit_falls_back_to_url_and_limit_truncates(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=_ok(
                [
                    _result_block(
                        [
                            {
                                "type": "web_search_result",
                                "url": "https://a.com",
                            },
                            _hit("https://b.com"),
                        ]
                    )
                ]
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=1)
        )
    assert results == [
        SearchResult(
            title="https://a.com",
            url="https://a.com",
            snippet="",
            source_provider="claude",
        )
    ]


async def test_missing_content_and_zero_limit_are_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CLAUDE_URL)
        route.side_effect = [
            httpx.Response(200, json={"type": "message"}),
            _ok([_result_block([_hit("https://a.com")])]),
        ]
        provider = ClaudeProvider(_KEY, http_client)
        empty = await provider.search(SearchRequest(query="empty"))
        fallback = await provider.search(
            SearchRequest(query="fallback", limit=0)
        )
    assert empty == []
    assert [result.url for result in fallback] == ["https://a.com"]


async def test_malformed_blocks_are_ignored_not_raised(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "type": "message",
                    "content": [
                        "a bare string block",
                        42,
                        {"type": "web_search_tool_result", "content": "text"},
                        _result_block(
                            ["a bare string hit", _hit("https://a.com", "A")]
                        ),
                        {"type": "text", "citations": "not a list"},
                        {
                            "type": "text",
                            "citations": [
                                "a bare string citation",
                                _citation("https://a.com", "excerpt"),
                            ],
                        },
                    ],
                },
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="A",
            url="https://a.com",
            snippet="excerpt",
            source_provider="claude",
        )
    ]


async def test_non_string_leaf_fields_are_ignored(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=_ok(
                [
                    _result_block(
                        [
                            {
                                "type": "web_search_result",
                                "url": ["https://unhashable.example"],
                            },
                            {"type": "web_search_result", "url": 42},
                            {
                                "type": "web_search_result",
                                "url": "https://a.com",
                                "title": ["not", "a", "string"],
                            },
                        ]
                    ),
                    {
                        "type": "text",
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "url": ["https://unhashable.example"],
                                "cited_text": "dropped",
                            },
                            {
                                "type": "web_search_result_location",
                                "url": "https://a.com",
                                "cited_text": {"not": "a string"},
                            },
                        ],
                    },
                ]
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="https://a.com",
            url="https://a.com",
            snippet="",
            source_provider="claude",
        )
    ]


async def test_paused_turn_without_results_is_transient(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "type": "message",
                    "stop_reason": "pause_turn",
                    "content": [
                        {"type": "server_tool_use", "name": "web_search"}
                    ],
                },
            )
        )
        with pytest.raises(ProviderError) as exc:
            await ClaudeProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.PROVIDER_ERROR
    assert str(exc.value) == (
        "Claude paused the search turn before returning a result"
    )
    assert exc.value.provider == "claude"


async def test_paused_turn_with_results_returns_them(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "type": "message",
                    "stop_reason": "pause_turn",
                    "content": [_result_block([_hit("https://a.com", "A")])],
                },
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [result.url for result in results] == ["https://a.com"]


async def test_tool_error_wins_over_a_paused_turn(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "type": "message",
                    "stop_reason": "pause_turn",
                    "content": [
                        {
                            "type": "web_search_tool_result",
                            "content": {
                                "type": "web_search_tool_result_error",
                                "error_code": "unavailable",
                            },
                        }
                    ],
                },
            )
        )
        with pytest.raises(ProviderError) as exc:
            await ClaudeProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == "Claude web search failed: unavailable"


async def test_non_list_content_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=httpx.Response(
                200, json={"type": "message", "content": "not a list"}
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_tool_error_without_results_raises_api_error(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=_ok(
                [
                    {
                        "type": "web_search_tool_result",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "query_too_long",
                        },
                    },
                    {
                        "type": "web_search_tool_result",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "unavailable",
                        },
                    },
                ]
            )
        )
        with pytest.raises(ProviderError) as exc:
            await ClaudeProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == "Claude web search failed: query_too_long"
    assert exc.value.provider == "claude"


async def test_tool_rate_limit_error_maps_to_rate_limit(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=_ok(
                [
                    {
                        "type": "web_search_tool_result",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "too_many_requests",
                        },
                    }
                ]
            )
        )
        with pytest.raises(ProviderError) as exc:
            await ClaudeProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.RATE_LIMIT


async def test_malformed_tool_error_without_results_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=_ok(
                [{"type": "web_search_tool_result", "content": {}}]
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_tool_error_after_results_keeps_the_results(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=_ok(
                [
                    _result_block([_hit("https://a.com")]),
                    {
                        "type": "web_search_tool_result",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "max_uses_exceeded",
                        },
                    },
                ]
            )
        )
        results = await ClaudeProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [result.url for result in results] == ["https://a.com"]


async def test_echoed_key_is_redacted_from_the_raised_error(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=httpx.Response(
                400, json={"error": f"invalid key {_KEY}"}
            )
        )
        with pytest.raises(ProviderError) as exc:
            await ClaudeProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert _KEY not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


async def test_401_raises_api_error(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(CLAUDE_URL).mock(return_value=httpx.Response(401, json={}))
        with pytest.raises(ProviderError) as exc:
            await ClaudeProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == "Invalid API key"


async def test_api_key_absent_from_logs(
    http_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with respx.mock:
        respx.post(CLAUDE_URL).mock(
            return_value=httpx.Response(500, json={"message": "boom"})
        )
        with pytest.raises(ProviderError) as exc:
            await ClaudeProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.PROVIDER_ERROR
    assert _KEY not in caplog.text


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await ClaudeProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for claude"

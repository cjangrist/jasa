"""Codex provider request, operator, mapping, settings, and error behavior."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.codex import CodexProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

CODEX_URL = "https://ai.angrist.net/v1/responses"
VENDOR_URL = "https://api.openai.com/v1/responses"
GATEWAY_URL = "https://gateway.example/v1/responses"
_KEY = "codex-test-key"


def _citation(url: str, title: str | None = "T") -> dict[str, object]:
    annotation: dict[str, object] = {"type": "url_citation", "url": url}
    if title is not None:
        annotation["title"] = title
    return annotation


def _message(annotations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "prose the model wrote",
                "annotations": annotations,
            }
        ],
    }


def _ok(output: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(
        200, json={"status": "completed", "error": None, "output": output}
    )


async def test_exact_outbound_request_and_mapping(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CODEX_URL).mock(
            return_value=_ok(
                [
                    {"type": "reasoning"},
                    {"type": "web_search_call", "status": "completed"},
                    _message([_citation("https://x.com")]),
                ]
            )
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="hello world", limit=7)
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "model": "gpt-5.6-luna",
        "input": ("Use the web_search tool to search the web for: hello world"),
        "tools": [{"type": "web_search", "search_context_size": "medium"}],
    }
    assert results == [
        SearchResult(
            title="T",
            url="https://x.com",
            snippet="",
            source_provider="codex",
        )
    ]


async def test_settings_override_endpoint_and_model(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(GATEWAY_URL).mock(return_value=_ok([]))
        await CodexProvider(
            _KEY,
            http_client,
            {
                "OPENAI_BASE_URL": "https://gateway.example/v1/",
                "CODEX_SEARCH_MODEL": "gateway-model",
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
        await CodexProvider(
            _KEY,
            http_client,
            {
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                "CODEX_SEARCH_MODEL": "gpt-5.6",
            },
        ).search(SearchRequest(query="q"))
        request = route.calls.last.request
    assert str(request.url) == VENDOR_URL
    assert json.loads(request.content)["model"] == "gpt-5.6"


async def test_blank_settings_fall_back_to_defaults(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CODEX_URL).mock(return_value=_ok([]))
        await CodexProvider(
            _KEY,
            http_client,
            {"OPENAI_BASE_URL": "", "CODEX_SEARCH_MODEL": ""},
        ).search(SearchRequest(query="q"))
        request = route.calls.last.request
    assert str(request.url) == CODEX_URL
    assert json.loads(request.content)["model"] == "gpt-5.6-luna"


async def test_domains_become_filters_and_other_operators_stay_in_query(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CODEX_URL).mock(return_value=_ok([]))
        await CodexProvider(_KEY, http_client).search(
            SearchRequest(
                query="site:b.com filetype:pdf foo -site:c.com",
                include_domains=("a.com",),
                exclude_domains=("d.com",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["filters"] == {
        "allowed_domains": ["a.com", "b.com"],
        "blocked_domains": ["d.com", "c.com"],
    }
    assert body["input"] == (
        "Use the web_search tool to search the web for: foo filetype:pdf"
    )


async def test_exclude_only_domains_send_one_filter(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(CODEX_URL).mock(return_value=_ok([]))
        await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="foo", exclude_domains=("d.com",))
        )
        body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["filters"] == {"blocked_domains": ["d.com"]}


async def test_tracking_params_stripped_and_citations_deduped(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=_ok(
                [
                    _message(
                        [
                            _citation("https://a.com/p?utm_source=openai"),
                            _citation("https://a.com/p", "duplicate"),
                            _citation(
                                "https://b.com/p?highlight=n&utm_medium=x",
                                "B",
                            ),
                            _citation("https://c.com/p?keep=1", "C"),
                            _citation("https://d.com/p", None),
                            {"type": "file_citation", "url": "https://e.com"},
                            {"type": "url_citation"},
                        ]
                    ),
                    _message([_citation("https://f.com", "F")]),
                ]
            )
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [(result.title, result.url) for result in results] == [
        ("T", "https://a.com/p"),
        ("B", "https://b.com/p?highlight=n"),
        ("C", "https://c.com/p?keep=1"),
        ("https://d.com/p", "https://d.com/p"),
        ("F", "https://f.com"),
    ]
    assert all(result.snippet == "" for result in results)


async def test_malformed_output_is_ignored_not_raised(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        "a bare string item",
                        7,
                        {"type": "message", "content": "not a list"},
                        {"type": "message", "content": ["a bare part", 9]},
                        {
                            "type": "message",
                            "content": [{"annotations": "not a list"}],
                        },
                        {
                            "type": "message",
                            "content": [
                                {
                                    "annotations": [
                                        "a bare annotation",
                                        {
                                            "type": "url_citation",
                                            "url": ["unhashable"],
                                        },
                                        {"type": "url_citation", "url": 42},
                                        {
                                            "type": "url_citation",
                                            "url": "https://a.com",
                                            "title": ["not", "a", "string"],
                                        },
                                    ]
                                }
                            ],
                        },
                    ],
                },
            )
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="https://a.com",
            url="https://a.com",
            snippet="",
            source_provider="codex",
        )
    ]


async def test_non_list_output_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(
                200, json={"status": "completed", "output": "not a list"}
            )
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_non_object_payload_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(200, json=["not", "a", "dict"])
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_missing_output_and_zero_limit_are_success(
    http_client: httpx.AsyncClient,
) -> None:
    many = [_citation(f"https://a{index}.com") for index in range(25)]
    with respx.mock:
        route = respx.post(CODEX_URL)
        route.side_effect = [
            httpx.Response(200, json={"status": "completed"}),
            _ok([_message(many)]),
        ]
        provider = CodexProvider(_KEY, http_client)
        empty = await provider.search(SearchRequest(query="empty"))
        fallback = await provider.search(
            SearchRequest(query="fallback", limit=0)
        )
    assert empty == []
    assert len(fallback) == 20
    assert fallback[-1].url == "https://a19.com"


async def test_empty_error_string_is_not_a_failure(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "completed",
                    "error": "",
                    "output": [_message([_citation("https://a.com", "A")])],
                },
            )
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [result.url for result in results] == ["https://a.com"]


async def test_failure_wins_over_returned_citations(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "failed",
                    "error": {"message": "aborted late"},
                    "output": [_message([_citation("https://a.com", "A")])],
                },
            )
        )
        with pytest.raises(ProviderError) as exc:
            await CodexProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert str(exc.value) == "aborted late"


async def test_empty_incomplete_response_is_transient(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [{"type": "web_search_call"}],
                },
            )
        )
        with pytest.raises(ProviderError) as exc:
            await CodexProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.PROVIDER_ERROR
    assert str(exc.value) == (
        "OpenAI ended the search turn early without a result"
    )
    assert exc.value.provider == "codex"


@pytest.mark.parametrize(
    ("marker", "value"),
    [
        ("code", "rate_limit_exceeded"),
        ("type", "rate_limit_error"),
        ("code", "rate_limit_error"),
        ("type", "rate_limit_exceeded"),
    ],
)
async def test_in_body_rate_limit_maps_to_rate_limit(
    http_client: httpx.AsyncClient, marker: str, value: str
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "failed",
                    "error": {marker: value, "message": "slow down"},
                },
            )
        )
        with pytest.raises(ProviderError) as exc:
            await CodexProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.RATE_LIMIT
    assert str(exc.value) == "slow down"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://a.com/p?utm_source=openai", "https://a.com/p"),
        (
            "https://a.com/p?a=1%20b&utm_source=openai",
            "https://a.com/p?a=1%20b",
        ),
        ("https://a.com/p?a=1;b=2&utm_medium=x", "https://a.com/p?a=1;b=2"),
        ("https://a.com/p?blank=&utm_x=1", "https://a.com/p?blank="),
        ("https://a.com/p?utm%5Fsource=openai&k=1", "https://a.com/p?k=1"),
        ("https://a.com/p?utmx=keep", "https://a.com/p?utmx=keep"),
    ],
)
async def test_surviving_query_parameters_are_byte_identical(
    http_client: httpx.AsyncClient, url: str, expected: str
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=_ok([_message([_citation(url, "T")])])
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [result.url for result in results] == [expected]


async def test_limit_truncates_citations(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=_ok(
                [
                    _message(
                        [
                            _citation("https://a.com", "A"),
                            _citation("https://b.com", "B"),
                        ]
                    )
                ]
            )
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=1)
        )
    assert [result.url for result in results] == ["https://a.com"]


async def test_incomplete_response_keeps_its_citations(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [_message([_citation("https://a.com", "A")])],
                },
            )
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [result.url for result in results] == ["https://a.com"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"status": "failed", "error": {"message": "server had a problem"}},
            "server had a problem",
        ),
        ({"status": "failed", "error": {}}, "OpenAI web search failed"),
        ({"status": "failed"}, "OpenAI web search failed"),
        ({"status": "completed", "error": "tool unavailable"}, "tool unavail"),
    ],
)
async def test_explicit_failure_raises_api_error(
    http_client: httpx.AsyncClient,
    payload: dict[str, object],
    expected: str,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(200, json=payload)
        )
        with pytest.raises(ProviderError) as exc:
            await CodexProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert expected in str(exc.value)
    assert exc.value.provider == "codex"


async def test_failure_message_redacts_an_echoed_key(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "failed",
                    "error": {"message": f"bad key {_KEY}"},
                },
            )
        )
        with pytest.raises(ProviderError) as exc:
            await CodexProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert _KEY not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


async def test_unparseable_citation_url_is_passed_through(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=_ok([_message([_citation("http://[oops?a=1", "X")])])
        )
        results = await CodexProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [result.url for result in results] == ["http://[oops?a=1"]


async def test_401_raises_api_error(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(return_value=httpx.Response(401, json={}))
        with pytest.raises(ProviderError) as exc:
            await CodexProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == "Invalid API key"


async def test_429_raises_rate_limit(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(CODEX_URL).mock(return_value=httpx.Response(429, json={}))
        with pytest.raises(ProviderError) as exc:
            await CodexProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.RATE_LIMIT


async def test_api_key_absent_from_logs(
    http_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with respx.mock:
        respx.post(CODEX_URL).mock(
            return_value=httpx.Response(500, json={"message": "boom"})
        )
        with pytest.raises(ProviderError) as exc:
            await CodexProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.PROVIDER_ERROR
    assert _KEY not in caplog.text


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await CodexProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert str(exc.value) == "API key not found for codex"

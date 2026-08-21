"""Z.AI provider: exact request, clamp, operator re-render, mapping, errors."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from jasa.search.fanout import _PER_PROVIDER_LIMIT
from jasa.search.providers.base import SearchRequest
from jasa.search.providers.zai import ZaiProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

ZAI_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
GATEWAY_URL = "https://gateway.example/v1/chat/completions"
_KEY = "zai-test-key"


def _hit(url: str, title: str = "T", content: str = "c") -> dict[str, Any]:
    return {
        "link": url,
        "title": title,
        "content": content,
        "refer": "ref_1",
        "publish_date": "",
    }


def _ok(hits: list[Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": ""}}],
            "web_search": hits,
        },
    )


async def test_exact_outbound_request_and_mapping(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(ZAI_URL).mock(
            return_value=_ok([_hit("https://x.com", "T", "the excerpt")])
        )
        results = await ZaiProvider(_KEY, http_client).search(
            SearchRequest(query="hello world", limit=7)
        )
        request = route.calls.last.request
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["content-type"] == "application/json"
    body = json.loads(request.content)
    assert body["model"] == "glm-4.6"
    assert body["max_tokens"] == 1
    assert body["messages"] == [{"role": "user", "content": "hello world"}]
    assert body["tools"] == [
        {
            "type": "web_search",
            "web_search": {
                "enable": True,
                "search_engine": "search-prime",
                "search_result": True,
                "count": 7,
            },
        }
    ]
    assert results == [
        SearchResult(
            title="T",
            url="https://x.com",
            snippet="the excerpt",
            source_provider="zai",
        )
    ]


async def test_count_is_clamped_to_the_upstream_maximum(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(ZAI_URL).mock(
            return_value=_ok([_hit(f"https://e{n}.com") for n in range(30)])
        )
        results = await ZaiProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=_PER_PROVIDER_LIMIT)
        )
        body = json.loads(route.calls.last.request.content)
    assert _PER_PROVIDER_LIMIT > 10
    assert body["tools"][0]["web_search"]["count"] == 10
    assert len(results) == 10


async def test_zero_limit_falls_back_to_the_default(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(ZAI_URL).mock(return_value=_ok([]))
        await ZaiProvider(_KEY, http_client).search(
            SearchRequest(query="q", limit=0)
        )
        body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["web_search"]["count"] == 10


async def test_operators_and_domains_rerendered_into_query(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(ZAI_URL).mock(return_value=_ok([]))
        await ZaiProvider(_KEY, http_client).search(
            SearchRequest(
                query="site:b.com filetype:pdf foo -site:c.com",
                include_domains=("a.com",),
                exclude_domains=("d.com",),
            )
        )
        body = json.loads(route.calls.last.request.content)
    rendered = body["messages"][0]["content"]
    for token in (
        "foo",
        "site:a.com",
        "site:b.com",
        "-site:c.com",
        "-site:d.com",
        "filetype:pdf",
    ):
        assert token in rendered


async def test_settings_override_endpoint_and_model(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(GATEWAY_URL).mock(return_value=_ok([]))
        await ZaiProvider(
            _KEY,
            http_client,
            {
                "Z_AI_BASE_URL": "https://gateway.example/v1",
                "ZAI_SEARCH_MODEL": "glm-5.3",
            },
        ).search(SearchRequest(query="q"))
        body = json.loads(route.calls.last.request.content)
    assert body["model"] == "glm-5.3"


async def test_blank_settings_fall_back_to_defaults(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(ZAI_URL).mock(return_value=_ok([]))
        await ZaiProvider(
            _KEY,
            http_client,
            {"Z_AI_BASE_URL": "", "ZAI_SEARCH_MODEL": ""},
        ).search(SearchRequest(query="q"))
        body = json.loads(route.calls.last.request.content)
    assert body["model"] == "glm-4.6"


async def test_trailing_slash_base_url_is_normalized(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(GATEWAY_URL).mock(return_value=_ok([]))
        await ZaiProvider(
            _KEY, http_client, {"Z_AI_BASE_URL": "https://gateway.example/v1/"}
        ).search(SearchRequest(query="q"))
    assert route.called


async def test_duplicate_and_untitled_hits_collapse_in_rank_order(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(ZAI_URL).mock(
            return_value=_ok(
                [
                    _hit("https://a.com", "A"),
                    {"title": "no link", "content": "x"},
                    _hit("https://b.com", ""),
                    _hit("https://a.com", "A again"),
                ]
            )
        )
        results = await ZaiProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert [(r.title, r.url) for r in results] == [
        ("A", "https://a.com"),
        ("https://b.com", "https://b.com"),
    ]


async def test_malformed_and_non_string_fields_are_ignored(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(ZAI_URL).mock(
            return_value=_ok(
                [
                    "not a mapping",
                    {"link": ["not", "a", "string"]},
                    {"link": "https://ok.com", "title": 7, "content": None},
                ]
            )
        )
        results = await ZaiProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == [
        SearchResult(
            title="https://ok.com",
            url="https://ok.com",
            snippet="",
            source_provider="zai",
        )
    ]


async def test_missing_and_non_list_web_search_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    for payload in ({}, {"web_search": None}, {"web_search": {"a": 1}}):
        with respx.mock:
            respx.post(ZAI_URL).mock(
                return_value=httpx.Response(200, json=payload)
            )
            results = await ZaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
        assert results == []


async def test_non_mapping_payload_is_success(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(ZAI_URL).mock(return_value=httpx.Response(200, json=[1, 2]))
        results = await ZaiProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert results == []


async def test_insufficient_balance_maps_to_rate_limit(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(ZAI_URL).mock(
            return_value=httpx.Response(
                429,
                json={
                    "error": {
                        "code": "1113",
                        "message": "Insufficient balance or no resource "
                        "package. Please recharge.",
                    }
                },
            )
        )
        with pytest.raises(ProviderError) as exc:
            await ZaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.RATE_LIMIT
    assert exc.value.provider == "zai"
    assert _KEY not in str(exc.value)


async def test_5xx_maps_to_transient_provider_error(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(ZAI_URL).mock(
            return_value=httpx.Response(503, json={"message": "upstream down"})
        )
        with pytest.raises(ProviderError) as exc:
            await ZaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.PROVIDER_ERROR
    assert exc.value.provider == "zai"
    assert "503" in str(exc.value)
    assert _KEY not in str(exc.value)


async def test_401_raises_api_error(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(ZAI_URL).mock(
            return_value=httpx.Response(401, json={"error": {"code": "1000"}})
        )
        with pytest.raises(ProviderError) as exc:
            await ZaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert str(exc.value) == "Invalid API key"
    assert exc.value.provider == "zai"


async def test_echoed_key_is_redacted_from_the_raised_error(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(ZAI_URL).mock(
            return_value=httpx.Response(
                400, json={"message": f"bad token {_KEY}"}
            )
        )
        with pytest.raises(ProviderError) as exc:
            await ZaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert exc.value.error_type is ErrorType.API_ERROR
    assert exc.value.provider == "zai"
    assert _KEY not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


async def test_missing_key_raises_invalid_input(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as exc:
        await ZaiProvider("", http_client).search(SearchRequest(query="q"))
    assert exc.value.error_type is ErrorType.INVALID_INPUT
    assert exc.value.provider == "zai"
    assert str(exc.value) == "API key not found for zai"

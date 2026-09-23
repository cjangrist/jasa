"""xAI/Grok Responses search payload, parsing, safety, and error behavior."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from jasa.search.providers.base import SearchRequest
from jasa.search.providers.xai import _safe_url, XaiProvider
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_URL = "https://ai.angrist.net/v1/responses"
_KEY = "xai-test-key"


def _reply(
    text: str = "",
    *,
    annotations: list[dict[str, object]] | None = None,
    citations: list[object] | None = None,
    searched: bool = True,
    status: str = "completed",
) -> httpx.Response:
    output: list[dict[str, object]] = []
    if searched:
        output.append({"type": "web_search_call", "status": "completed"})
    output.append(
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                    "annotations": annotations
                    if annotations is not None
                    else [],
                }
            ],
        }
    )
    payload: dict[str, object] = {"status": status, "output": output}
    if citations is not None:
        payload["citations"] = citations
    return httpx.Response(200, json=payload)


async def test_exact_request_and_model_generated_rows(
    http_client: httpx.AsyncClient,
) -> None:
    text = json.dumps(
        {
            "results": [
                "not a row",
                {
                    "title": "News",
                    "url": "https://example.com/a",
                    "description": (
                        "Current relevant evidence from the original page "
                        "and a summary of the source."
                    ),
                },
                {
                    "title": "Duplicate",
                    "url": "https://example.com/a",
                    "description": "Repeated source.",
                },
                {
                    "title": "Docs",
                    "url": "https://docs.example.com/",
                    "description": (
                        "Official documentation from the original site."
                    ),
                },
            ]
        }
    )
    with respx.mock:
        route = respx.post(_URL).mock(return_value=_reply(text))
        rows = await XaiProvider(_KEY, http_client).search(
            SearchRequest(query="news", limit=8)
        )
        request = route.calls.last.request
    body = json.loads(request.content)
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert body["model"] == "grok-4.7-build-fast"
    assert body["tools"] == [{"type": "web_search"}]
    assert body["include"] == ["no_inline_citations"]
    assert body["input"][0]["role"] == "user"
    assert "Return at most 8 results" in body["input"][0]["content"]
    assert "Query: news" in body["input"][0]["content"]
    assert rows == [
        SearchResult(
            "News",
            "https://example.com/a",
            "Current relevant evidence from the original page "
            "and a summary of the source.",
            "xai",
        ),
        SearchResult(
            "Docs",
            "https://docs.example.com/",
            "Official documentation from the original site.",
            "xai",
        ),
    ]


@pytest.mark.parametrize(
    ("query", "includes", "excludes", "expected_filters"),
    [
        (
            "site:docs.example.com filetype:pdf new",
            (),
            (),
            {"allowed_domains": ["docs.example.com"]},
        ),
        (
            "new",
            ("a.com", "b.com"),
            (),
            {"allowed_domains": ["a.com", "b.com"]},
        ),
        ("new", (), ("spam.com",), {"excluded_domains": ["spam.com"]}),
        ("site:a.com -site:spam.com new", (), (), None),
        ("new", ("a.com",), ("spam.com",), None),
        (
            "new",
            ("a.com", "b.com", "c.com", "d.com", "e.com", "f.com"),
            (),
            {"allowed_domains": ["a.com", "b.com", "c.com", "d.com", "e.com"]},
        ),
    ],
)
async def test_domain_filters_and_query_preservation(
    http_client: httpx.AsyncClient,
    query: str,
    includes: tuple[str, ...],
    excludes: tuple[str, ...],
    expected_filters: dict[str, list[str]] | None,
) -> None:
    with respx.mock:
        route = respx.post(_URL).mock(return_value=_reply())
        assert (
            await XaiProvider(_KEY, http_client).search(
                SearchRequest(
                    query=query,
                    include_domains=includes,
                    exclude_domains=excludes,
                )
            )
            == []
        )
        body = json.loads(route.calls.last.request.content)
    assert body["tools"][0].get("filters") == expected_filters
    if "site:" in query:
        assert "site:" in body["input"][0]["content"]
    if "f.com" in includes:
        assert "site:f.com" in body["input"][0]["content"]


async def test_override_and_prose_wrapped_json(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post("https://gateway.example/v2/responses").mock(
            return_value=_reply(
                'Here is the result: {"results": [{"url": "https://source.example/a"}]}'
            )
        )
        rows = await XaiProvider(
            _KEY,
            http_client,
            {
                "XAI_SEARCH_BASE_URL": "https://gateway.example/v2/",
                "XAI_SEARCH_MODEL": "custom-grok",
            },
        ).search(SearchRequest(query="q"))
        body = json.loads(route.calls.last.request.content)
    assert body["model"] == "custom-grok"
    assert rows == [
        SearchResult(
            "https://source.example/a", "https://source.example/a", "", "xai"
        )
    ]


async def test_annotations_then_citation_fallback(
    http_client: httpx.AsyncClient,
) -> None:
    annotations: list[dict[str, object]] = [
        {"type": "other", "url": "https://discard.example/"},
        {"type": "url_citation", "url": "https://example.com/a", "title": "1"},
        {"type": "url_citation", "url": "https://example.com/a"},
        {"type": "url_citation", "url": "http://localhost/"},
    ]
    with respx.mock:
        route = respx.post(_URL).mock(
            return_value=_reply(
                "Non-JSON prose",
                annotations=annotations,
                citations=["https://fallback.com/"],
            )
        )
        rows = await XaiProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
        route.mock(
            return_value=_reply(
                "Non-JSON",
                citations=[
                    "",
                    "https://fallback.com/a",
                    9,
                    "https://fallback.com/a",
                ],
            )
        )
        fallback = await XaiProvider(_KEY, http_client).search(
            SearchRequest(query="other")
        )
    assert rows == [
        SearchResult(
            "https://example.com/a", "https://example.com/a", "", "xai"
        )
    ]
    assert fallback == [
        SearchResult(
            "https://fallback.com/a", "https://fallback.com/a", "", "xai"
        )
    ]


async def test_invalid_json_or_results_use_citations(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(_URL).mock(
            return_value=_reply(
                '{"results": "bad"}', citations=["https://cite.example/"]
            )
        )
        rows = await XaiProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
        route.mock(
            return_value=_reply(
                '{"results": [{"url": "http://127.0.0.1/"}]}',
                annotations=[
                    {"type": "url_citation", "url": "https://safe.example/"}
                ],
            )
        )
        safe = await XaiProvider(_KEY, http_client).search(
            SearchRequest(query="q")
        )
    assert rows[0].url == "https://cite.example/"
    assert safe[0].url == "https://safe.example/"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://example.com/",
        "https://localhost/",
        "https://host.local/",
        "https://host.internal/",
        "https://host.test/",
        "https://host.invalid/",
        "https://127.0.0.1/",
        "https://192.168.1.1/",
        "https://[::1]/",
        "https://user:pass@example.com/",
        "https://single/",
        "https://[bad/",
        "https://example.com:abc/",
    ],
)
def test_rejects_model_generated_unsafe_urls(url: str) -> None:
    assert not _safe_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/a",
        "https://8.8.8.8/",
        "https://[2606:4700:4700::1111]/",
    ],
)
def test_accepts_public_https_urls(url: str) -> None:
    assert _safe_url(url)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://gateway.example/v1",
        "https://user@gateway.example/v1",
        "https://gateway.example/v1?token=bad",
        "https://gateway.example/v1#tag",
        "not-a-url",
        "https://gateway.example:bad/v1",
    ],
)
async def test_bad_endpoint_rejected_before_http(
    http_client: httpx.AsyncClient, endpoint: str
) -> None:
    with respx.mock, pytest.raises(ProviderError) as error:
        await XaiProvider(
            _KEY, http_client, {"XAI_SEARCH_BASE_URL": endpoint}
        ).search(SearchRequest(query="q"))
    assert error.value.error_type == ErrorType.INVALID_INPUT
    assert _KEY not in str(error.value)


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        (
            {"error": {"type": "rate_limit_error", "message": "limited"}},
            ErrorType.RATE_LIMIT,
        ),
        (
            {"error": {"code": "rate_limit_exceeded", "message": "limited"}},
            ErrorType.RATE_LIMIT,
        ),
        ({"error": {"type": "other"}}, ErrorType.API_ERROR),
        ({"error": "bad"}, ErrorType.API_ERROR),
        ({"status": "failed"}, ErrorType.API_ERROR),
        ({"status": "cancelled"}, ErrorType.API_ERROR),
    ],
)
async def test_in_body_failures(
    http_client: httpx.AsyncClient, payload: dict[str, object], kind: ErrorType
) -> None:
    with respx.mock:
        respx.post(_URL).respond(200, json=payload)
        with pytest.raises(ProviderError) as error:
            await XaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert error.value.error_type == kind


async def test_error_redacts_key(http_client: httpx.AsyncClient) -> None:
    with respx.mock:
        respx.post(_URL).respond(
            200, json={"error": {"message": f"Rejected {_KEY}"}}
        )
        with pytest.raises(ProviderError) as error:
            await XaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert _KEY not in str(error.value)
    assert "[REDACTED]" in str(error.value)


@pytest.mark.parametrize("status", ["incomplete", "in_progress", "queued"])
async def test_unfinished_without_rows_is_failure(
    http_client: httpx.AsyncClient, status: str
) -> None:
    with respx.mock:
        respx.post(_URL).mock(return_value=_reply(status=status))
        with pytest.raises(ProviderError) as error:
            await XaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert error.value.error_type == ErrorType.PROVIDER_ERROR


async def test_without_search_call_is_failure(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        respx.post(_URL).mock(
            return_value=_reply(
                '{"results": [{"url": "https://fake.example/"}]}',
                searched=False,
            )
        )
        with pytest.raises(ProviderError, match="did not execute"):
            await XaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )


async def test_empty_completed_search_and_nonmapping_response(
    http_client: httpx.AsyncClient,
) -> None:
    with respx.mock:
        route = respx.post(_URL).mock(return_value=_reply())
        assert (
            await XaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
            == []
        )
        route.mock(return_value=httpx.Response(200, json=[1, 2]))
        with pytest.raises(ProviderError, match="did not execute"):
            await XaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )


async def test_missing_key_and_http_error(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(ProviderError) as absent:
        await XaiProvider("", http_client).search(SearchRequest(query="q"))
    assert absent.value.error_type == ErrorType.INVALID_INPUT
    with respx.mock:
        respx.post(_URL).respond(429, json={"error": "limited"})
        with pytest.raises(ProviderError) as limited:
            await XaiProvider(_KEY, http_client).search(
                SearchRequest(query="q")
            )
    assert limited.value.error_type == ErrorType.RATE_LIMIT

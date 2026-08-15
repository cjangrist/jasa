"""Grounding service: mocked fetch + LLM, outcome classification."""

from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest
import respx

from jasa.config import GroundingSettings
from jasa.grounding.detectors import grounding_detector_semantics
from jasa.grounding.prompts import (
    GROUNDING_MAX_TOKENS,
    grounding_prompt_semantics,
    SNIPPET_MAX_CHARS,
)
from jasa.grounding.service import (
    FREQUENCY_PENALTY,
    ground_results,
    grounding_semantic_fingerprint,
    GROUNDING_SEMANTICS_VERSION,
    GroundingContext,
    MIN_CONTENT_CHARS,
    MIN_SNIPPET_CHARS,
    TEMPERATURE,
    TOP_P,
)
from jasa.search.ranking import RankedWebResult

_SETTINGS = GroundingSettings()
_KEY = "cerebras-test"
_LLM_URL = "https://api.cerebras.ai/v1/chat/completions"


class _FetchResult:
    def __init__(self, content: str, title: str = "") -> None:
        self.content = content
        self.title = title


def _result(url: str) -> RankedWebResult:
    return RankedWebResult("t", url, ["agg"], ["p"], 0.1)


def _ctx() -> tuple[GroundingContext, httpx.AsyncClient]:
    """Return a grounding context with a dummy engine + fresh client."""
    client = httpx.AsyncClient()
    return (
        GroundingContext(
            engine=object(),
            client=client,
            api_key=_KEY,
            config=_SETTINGS,
        ),
        client,
    )


def _llm_ok(text: str | None) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": text}}]}
    )


def test_grounding_semantic_fingerprint_covers_output_inputs() -> None:
    identity = {
        "detectors": grounding_detector_semantics(),
        "frequency_penalty": FREQUENCY_PENALTY,
        "llm_base_url": _SETTINGS.llm_base_url,
        "llm_model": _SETTINGS.llm_model,
        "max_content_chars": _SETTINGS.max_content_chars,
        "max_tokens": GROUNDING_MAX_TOKENS,
        "min_content_chars": MIN_CONTENT_CHARS,
        "min_snippet_chars": MIN_SNIPPET_CHARS,
        "prompts": grounding_prompt_semantics(),
        "semantics_version": GROUNDING_SEMANTICS_VERSION,
        "snippet_max_chars": SNIPPET_MAX_CHARS,
        "temperature": TEMPERATURE,
        "top_n": _SETTINGS.top_n,
        "top_p": TOP_P,
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert grounding_semantic_fingerprint(_SETTINGS) == expected
    variants = {
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"llm_base_url": "https://other"})
        ),
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"llm_model": "other"})
        ),
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"max_content_chars": 500})
        ),
        grounding_semantic_fingerprint(
            _SETTINGS.model_copy(update={"top_n": 3})
        ),
    }
    assert expected not in variants
    assert len(variants) == 4


async def test_grounded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20, "Title")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    with respx.mock:
        route = respx.post(_LLM_URL).mock(
            return_value=_llm_ok("Grounded snippet.")
        )
        pairs, stats = await ground_results(
            "q", [_result("https://x.com")], ctx
        )
    assert pairs[0][1] == "grounded"
    assert pairs[0][0].snippet_source == "grounded"
    assert stats.grounded_count == 1
    assert (
        json.loads(route.calls.last.request.content)["max_tokens"]
        == GROUNDING_MAX_TOKENS
    )
    await client.aclose()


async def test_fetch_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> None:
        raise RuntimeError("connect failed")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:fetch_exhausted"
    assert stats.transient_failures == 0
    await client.aclose()


async def test_fetch_too_short(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("short")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:fetch_too_short"
    await client.aclose()


async def test_fetch_junk(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("subscribe to continue reading" + "x" * 100)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:fetch_junk"
    await client.aclose()


async def test_llm_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok("[no usable content]"))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:llm_sentinel"
    await client.aclose()


async def test_overlong_snippet_repairs_fence_after_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    overlong = "x" * 1980 + "\n```python\n" + "y" * 100
    ctx, client = _ctx()
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok(overlong))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    snippet = pairs[0][0].snippets[0]
    assert len(snippet) == 2004
    assert snippet.endswith("\n```")
    assert snippet.count("```") == 2
    await client.aclose()


async def test_llm_error_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=httpx.Response(500))
        pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:llm_error"
    assert stats.transient_failures == 1
    await client.aclose()


async def test_pipeline_timeout_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        await asyncio.sleep(0.15)
        return _FetchResult("content")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    settings = GroundingSettings(per_url_deadline_ms=100)
    client = httpx.AsyncClient()
    ctx = GroundingContext(
        engine=object(), client=client, api_key=_KEY, config=settings
    )
    pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:pipeline_timeout"
    assert stats.transient_failures == 1
    await client.aclose()


async def test_llm_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok(""))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:llm_empty"
    await client.aclose()


async def test_llm_null_content_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Real content. " * 20)

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    with respx.mock:
        respx.post(_LLM_URL).mock(return_value=_llm_ok(None))
        pairs, _ = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:llm_empty"
    await client.aclose()


async def test_worker_rejection_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject(*_args: object) -> None:
        raise RuntimeError("worker failed")

    monkeypatch.setattr("jasa.grounding.service._fetch_and_ground", reject)
    ctx, client = _ctx()
    pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:worker_rejected"
    assert stats.transient_failures == 1
    await client.aclose()


async def test_fetch_error_not_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(engine: object, url: str) -> None:
        raise ValueError("unexpected")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    ctx, client = _ctx()
    pairs, stats = await ground_results("q", [_result("u")], ctx)
    assert pairs[0][1] == "fallback:fetch_exhausted"
    assert stats.transient_failures == 0
    await client.aclose()


async def test_total_urls_counts_only_processed_top_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_urls: list[str] = []

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        processed_urls.append(url)
        return _FetchResult("short")

    monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fake_fetch)
    settings = GroundingSettings(top_n=2)
    client = httpx.AsyncClient()
    ctx = GroundingContext(
        engine=object(), client=client, api_key=_KEY, config=settings
    )
    results = [_result(f"https://{index}.example") for index in range(4)]
    pairs, stats = await ground_results("q", results, ctx)
    assert len(pairs) == 2
    assert stats.total_urls == 2
    assert processed_urls == ["https://0.example", "https://1.example"]
    await client.aclose()

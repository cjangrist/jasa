"""Grounding service: mocked fetch + LLM, outcome classification."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from jasa.config import GroundingSettings
from jasa.grounding.prompts import GROUNDING_MAX_TOKENS
from jasa.grounding.service import ground_results, GroundingContext
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

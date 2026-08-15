"""Grounding flight deadlines, worker-slot scheduling, and log redaction."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import cast

import httpx
import pytest

import jasa.grounding.cache as cache_module
from jasa.cache.base import CacheBackend
from jasa.cache.memory import MemoryCache
from jasa.config import GroundingSettings
from jasa.grounding.cache import (
    grounding_cache_identity,
    make_grounding_cache_key,
)
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.prompts import build_grounded_user_message
from jasa.grounding.service import ground_results, GroundingContext
from jasa.search.ranking import RankedWebResult


class _FetchResult:
    def __init__(self, content: str, title: str = "Title") -> None:
        self.content = content
        self.title = title


def _result(url: str) -> RankedWebResult:
    return RankedWebResult("title", url, ["aggregate"], ["provider"], 0.1)


def _context(
    cache: CacheBackend,
    flights: GroundingFlightRegistry,
    *,
    settings: GroundingSettings,
) -> tuple[GroundingContext, httpx.AsyncClient]:
    client = httpx.AsyncClient()
    return (
        GroundingContext(
            engine=object(),
            client=client,
            cache=cache,
            cache_write_semaphore=asyncio.Semaphore(settings.concurrency),
            flights=flights,
            api_key="test-key",
            config=settings,
        ),
        client,
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


def _capture_events(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        cache_module,
        "emit_grounding_cache_metric",
        lambda **fields: events.append(fields),
    )
    return events


async def test_leader_timeout_releases_longer_budget_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_call_started = asyncio.Event()
    first_call_cancelled = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> str:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            first_call_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_call_cancelled.set()
                raise
        return "Recovered"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr("jasa.grounding.service._llm_call", fake_llm_call)
    events = _capture_events(monkeypatch)
    cache = MemoryCache()
    flights = GroundingFlightRegistry()
    leader_context, leader_client = _context(
        cache,
        flights,
        settings=GroundingSettings(per_url_deadline_ms=100),
    )
    waiter_context, waiter_client = _context(
        cache,
        flights,
        settings=GroundingSettings(per_url_deadline_ms=500),
    )
    leader = asyncio.create_task(
        ground_results("query", [_result("a")], leader_context)
    )
    await first_call_started.wait()
    waiter = asyncio.create_task(
        ground_results("query", [_result("a")], waiter_context)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    leader_result, waiter_result = await asyncio.gather(leader, waiter)

    assert first_call_cancelled.is_set()
    assert leader_result[0][0][1] == "fallback:pipeline_timeout"
    assert waiter_result[0][0][1] == "grounded"
    assert llm_calls == 2
    assert flights.active_count == 0
    await leader_client.aclose()
    await waiter_client.aclose()


async def test_waiter_timeout_does_not_cancel_longer_budget_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader_started = asyncio.Event()
    release_leader = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> str:
        nonlocal llm_calls
        llm_calls += 1
        leader_started.set()
        await release_leader.wait()
        return "Grounded"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr("jasa.grounding.service._llm_call", fake_llm_call)
    events = _capture_events(monkeypatch)
    cache = MemoryCache()
    flights = GroundingFlightRegistry()
    leader_context, leader_client = _context(
        cache,
        flights,
        settings=GroundingSettings(per_url_deadline_ms=1000),
    )
    waiter_context, waiter_client = _context(
        cache,
        flights,
        settings=GroundingSettings(per_url_deadline_ms=100),
    )
    leader = asyncio.create_task(
        ground_results("query", [_result("a")], leader_context)
    )
    await leader_started.wait()
    waiter = asyncio.create_task(
        ground_results("query", [_result("a")], waiter_context)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    waiter_result = await waiter

    assert waiter_result[0][0][1] == "fallback:pipeline_timeout"
    assert flights.active_count == 1
    release_leader.set()
    leader_result = await leader
    assert leader_result[0][0][1] == "grounded"
    assert llm_calls == 1
    assert flights.active_count == 0
    await leader_client.aclose()
    await waiter_client.aclose()


async def test_waiter_worker_reacquisition_respects_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = GroundingSettings(
        concurrency=1,
        per_url_deadline_ms=100,
        top_n=2,
    )
    waiting_content = "Waiting page content. " * 20
    blocker_started = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        content = (
            waiting_content if url == "waiting" else "Blocker content. " * 20
        )
        return _FetchResult(content)

    async def fake_llm_call(*args: object) -> str:
        nonlocal llm_calls
        llm_calls += 1
        blocker_started.set()
        await asyncio.Event().wait()
        return "unreachable"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr("jasa.grounding.service._llm_call", fake_llm_call)
    flights = GroundingFlightRegistry()
    waiting_message = build_grounded_user_message(
        "query",
        "Title",
        waiting_content,
        settings.max_content_chars,
    )
    waiting_key = make_grounding_cache_key(
        grounding_cache_identity(waiting_message, settings)
    )
    is_leader, completion = flights.claim(waiting_key)
    context, client = _context(
        MemoryCache(),
        flights,
        settings=settings,
    )
    started_at = asyncio.get_running_loop().time()
    task = asyncio.create_task(
        ground_results(
            "query",
            [_result("waiting"), _result("blocker")],
            context,
        )
    )
    await blocker_started.wait()
    flights.release(waiting_key, completion)
    pairs, stats = await task
    elapsed = asyncio.get_running_loop().time() - started_at

    assert is_leader is True
    assert [pair[1] for pair in pairs] == [
        "fallback:pipeline_timeout",
        "fallback:pipeline_timeout",
    ]
    assert stats.transient_failures == 2
    assert llm_calls == 1
    assert elapsed < 0.5
    assert flights.active_count == 0
    await client.aclose()


async def test_waiters_release_worker_slots_for_distinct_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_call_started = asyncio.Event()
    distinct_call_started = asyncio.Event()
    release_shared_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        content = (
            "Distinct page content. "
            if url == "distinct"
            else "Shared page content. "
        )
        return _FetchResult(content * 20)

    async def fake_llm_call(*args: object) -> str:
        nonlocal llm_calls
        llm_calls += 1
        user_message = cast(str, args[3])
        if "Distinct page content" in user_message:
            distinct_call_started.set()
            return "Distinct grounding"
        shared_call_started.set()
        await release_shared_call.wait()
        return "Shared grounding"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr("jasa.grounding.service._llm_call", fake_llm_call)
    events = _capture_events(monkeypatch)
    flights = GroundingFlightRegistry()
    context, client = _context(
        MemoryCache(),
        flights,
        settings=GroundingSettings(concurrency=2, top_n=3),
    )
    task = asyncio.create_task(
        ground_results(
            "query",
            [_result("shared-a"), _result("shared-b"), _result("distinct")],
            context,
        )
    )
    await shared_call_started.wait()
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    await distinct_call_started.wait()

    assert llm_calls == 2
    release_shared_call.set()
    pairs, stats = await task
    assert [pair[1] for pair in pairs] == ["grounded", "grounded", "grounded"]
    assert stats.grounded_count == 3
    assert flights.active_count == 0
    await client.aclose()


async def test_grounding_cache_logs_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    query = "private grounding query"
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Private fetched page content. " * 20)

    async def fake_llm_call(*args: object) -> str:
        first_call_started.set()
        await release_first_call.wait()
        return "Private grounded output"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr("jasa.grounding.service._llm_call", fake_llm_call)
    flights = GroundingFlightRegistry()
    context, client = _context(
        MemoryCache(),
        flights,
        settings=GroundingSettings(),
    )
    with caplog.at_level(logging.DEBUG, logger="jasa.grounding.cache"):
        leader = asyncio.create_task(
            ground_results(query, [_result("a")], context)
        )
        await first_call_started.wait()
        waiter = asyncio.create_task(
            ground_results(query, [_result("a")], context)
        )
        await _wait_until(
            lambda: any(
                "event=coalesced" in message for message in caplog.messages
            )
        )
        release_first_call.set()
        await asyncio.gather(leader, waiter)

    messages = "\n".join(caplog.messages)
    assert query not in messages
    assert "Private fetched page content" not in messages
    assert "Private grounded output" not in messages
    assert "jasa:grounding:v1:" not in messages
    assert "Grounding cache event=coalesced" in messages
    assert "Grounding cache event=write" in messages
    assert "Grounding cache event=hit" in messages
    await client.aclose()

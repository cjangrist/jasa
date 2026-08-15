"""Grounding flight cancellation, rejected writes, and stale releases."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

import httpx
import pytest

import jasa.grounding.cache as cache_module
from jasa.cache.base import CacheBackend
from jasa.cache.memory import MemoryCache
from jasa.config import GroundingSettings
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.service import ground_results, GroundingContext
from jasa.search.ranking import RankedWebResult


class _FetchResult:
    def __init__(self, content: str, title: str = "Title") -> None:
        self.content = content
        self.title = title


class _RejectingCache:
    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        return False

    async def close(self) -> None:
        return None


def _result(url: str) -> RankedWebResult:
    return RankedWebResult("title", url, ["aggregate"], ["provider"], 0.1)


def _context(
    cache: CacheBackend,
    flights: GroundingFlightRegistry,
) -> tuple[GroundingContext, httpx.AsyncClient]:
    settings = GroundingSettings()
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


async def test_leader_cancellation_releases_waiter(
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
    flights = GroundingFlightRegistry()
    context, client = _context(MemoryCache(), flights)
    leader = asyncio.create_task(
        ground_results("query", [_result("a")], context)
    )
    await first_call_started.wait()
    waiter = asyncio.create_task(
        ground_results("query", [_result("a")], context)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    leader.cancel()

    with pytest.raises(asyncio.CancelledError):
        await leader
    waiter_result = await waiter
    assert first_call_cancelled.is_set()
    assert waiter_result[0][0][1] == "grounded"
    assert llm_calls == 2
    assert flights.active_count == 0
    await client.aclose()


async def test_rejected_write_releases_waiter_to_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> str:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            first_call_started.set()
            await release_first_call.wait()
        return "Grounded"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr("jasa.grounding.service._llm_call", fake_llm_call)
    events = _capture_events(monkeypatch)
    flights = GroundingFlightRegistry()
    context, client = _context(
        cast(CacheBackend, _RejectingCache()),
        flights,
    )
    leader = asyncio.create_task(
        ground_results("query", [_result("a")], context)
    )
    await first_call_started.wait()
    waiter = asyncio.create_task(
        ground_results("query", [_result("a")], context)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    release_first_call.set()
    outcomes = await asyncio.gather(leader, waiter)

    assert all(outcome[0][0][1] == "grounded" for outcome in outcomes)
    assert llm_calls == 2
    assert flights.active_count == 0
    assert sum(event["event"] == "write_error" for event in events) == 2
    await client.aclose()


async def test_waiter_cancellation_does_not_cancel_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> str:
        nonlocal llm_calls
        llm_calls += 1
        first_call_started.set()
        await release_first_call.wait()
        return "Grounded"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr("jasa.grounding.service._llm_call", fake_llm_call)
    events = _capture_events(monkeypatch)
    flights = GroundingFlightRegistry()
    context, client = _context(MemoryCache(), flights)
    leader = asyncio.create_task(
        ground_results("query", [_result("a")], context)
    )
    await first_call_started.wait()
    waiter = asyncio.create_task(
        ground_results("query", [_result("a")], context)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert flights.active_count == 1
    release_first_call.set()
    leader_result = await leader
    assert leader_result[0][0][1] == "grounded"
    assert llm_calls == 1
    assert flights.active_count == 0
    await client.aclose()


async def test_stale_release_does_not_remove_new_flight() -> None:
    flights = GroundingFlightRegistry()
    first_leader, first = flights.claim("key")
    flights.release("key", first)
    second_leader, second = flights.claim("key")

    flights.release("key", first)

    assert first_leader is True
    assert second_leader is True
    assert flights.active_count == 1
    flights.release("key", second)
    assert flights.active_count == 0

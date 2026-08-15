"""Grounding miss coalescing and non-cacheable leader retries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

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


async def test_identical_concurrent_misses_call_llm_once(
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
        return "Grounded once"

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
    waiters = [
        asyncio.create_task(ground_results("query", [_result("a")], context))
        for _ in range(4)
    ]
    await _wait_until(
        lambda: sum(event["event"] == "coalesced" for event in events) == 4
    )

    assert flights.active_count == 1
    release_first_call.set()
    outcomes = await asyncio.gather(leader, *waiters)

    assert llm_calls == 1
    assert flights.active_count == 0
    assert all(outcome[0][0][1] == "grounded" for outcome in outcomes)
    assert all(
        outcome[0][0][0].snippets == ["Grounded once"] for outcome in outcomes
    )
    await client.aclose()


async def test_distinct_effective_inputs_do_not_coalesce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    both_calls_started = asyncio.Event()
    release_calls = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> _FetchResult:
        return _FetchResult(f"{url} distinct page content. " * 20)

    async def fake_llm_call(*args: object) -> str:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 2:
            both_calls_started.set()
        await release_calls.wait()
        return "Grounded"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr("jasa.grounding.service._llm_call", fake_llm_call)
    flights = GroundingFlightRegistry()
    context, client = _context(MemoryCache(), flights)
    first = asyncio.create_task(
        ground_results("query", [_result("a")], context)
    )
    second = asyncio.create_task(
        ground_results("query", [_result("b")], context)
    )
    await both_calls_started.wait()

    assert flights.active_count == 2
    release_calls.set()
    await asyncio.gather(first, second)
    assert llm_calls == 2
    assert flights.active_count == 0
    await client.aclose()


@pytest.mark.parametrize(
    ("leader_output", "leader_outcome"),
    [
        ("", "fallback:llm_empty"),
        ("[no usable content]", "fallback:llm_sentinel"),
    ],
)
async def test_noncacheable_leader_releases_waiter_to_retry(
    monkeypatch: pytest.MonkeyPatch,
    leader_output: str,
    leader_outcome: str,
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
            return leader_output
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
    release_first_call.set()
    leader_result, waiter_result = await asyncio.gather(leader, waiter)

    assert leader_result[0][0][1] == leader_outcome
    assert waiter_result[0][0][1] == "grounded"
    assert waiter_result[0][0][0].snippets == ["Recovered"]
    assert llm_calls == 2
    assert flights.active_count == 0
    await client.aclose()


async def test_llm_error_leader_releases_waiter_to_retry(
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
            raise RuntimeError("leader failed")
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
    release_first_call.set()
    leader_result, waiter_result = await asyncio.gather(leader, waiter)

    assert leader_result[0][0][1] == "fallback:llm_error"
    assert waiter_result[0][0][1] == "grounded"
    assert llm_calls == 2
    assert flights.active_count == 0
    await client.aclose()

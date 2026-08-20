"""Grounding miss coalescing and non-cacheable leader retries."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from jasa.cache.memory import MemoryCache
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.service import ground_results
from tests.conftest import GroundingFlightHarness


class _ParallelHitCache(MemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.block_reads = False
        self.read_count = 0
        self.both_reads_started = asyncio.Event()
        self.release_reads = asyncio.Event()

    async def get(self, key: str) -> str | None:
        if self.block_reads:
            self.read_count += 1
            if self.read_count == 2:
                self.both_reads_started.set()
            await self.release_reads.wait()
        return await super().get(key)


async def test_identical_concurrent_misses_call_llm_once(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

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
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(MemoryCache(), flights)
    leader = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    await grounding_flights.wait_for_event(first_call_started)
    waiters = [
        asyncio.create_task(
            ground_results("query", [grounding_flights.result("a")], context)
        )
        for _ in range(4)
    ]
    await grounding_flights.wait_until(
        lambda: (
            sum(
                event["event"] == "coalesced"
                for event in grounding_flights.events
            )
            == 4
        )
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


async def test_concurrent_cache_hits_do_not_join_a_flight(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> str:
        nonlocal llm_calls
        llm_calls += 1
        return "Cached grounding"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    cache = _ParallelHitCache()
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(cache, flights)
    await ground_results("query", [grounding_flights.result("a")], context)
    cache.block_reads = True

    first = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    second = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    await grounding_flights.wait_for_event(cache.both_reads_started)

    assert flights.active_count == 0
    assert not any(
        event["event"] == "coalesced" for event in grounding_flights.events
    )
    cache.release_reads.set()
    outcomes = await asyncio.gather(first, second)

    assert llm_calls == 1
    assert flights.active_count == 0
    assert all(outcome[0][0][1] == "grounded" for outcome in outcomes)


async def test_leader_reread_shares_the_initial_read_deadline(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    read_deadlines: list[float] = []

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

    async def capture_miss(*args: object) -> None:
        read_deadlines.append(cast(float, args[2]))

    async def fake_llm_call(*args: object) -> str:
        return "Grounded"

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._read_cached_grounding",
        capture_miss,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    context = grounding_flights.context(
        MemoryCache(), GroundingFlightRegistry()
    )

    pairs, _stats = await ground_results(
        "query", [grounding_flights.result("a")], context
    )

    assert pairs[0][1] == "grounded"
    assert len(read_deadlines) == 2
    assert read_deadlines[0] == read_deadlines[1]


async def test_distinct_effective_inputs_do_not_coalesce(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    both_calls_started = asyncio.Event()
    release_calls = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result(
            f"{url} distinct page content. " * 20
        )

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
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(MemoryCache(), flights)
    first = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    second = asyncio.create_task(
        ground_results("query", [grounding_flights.result("b")], context)
    )
    await grounding_flights.wait_for_event(both_calls_started)

    assert flights.active_count == 2
    release_calls.set()
    await asyncio.gather(first, second)
    assert llm_calls == 2
    assert flights.active_count == 0


@pytest.mark.parametrize(
    ("leader_output", "leader_outcome"),
    [
        ("", "fallback:llm_empty"),
        ("[no usable content]", "fallback:llm_sentinel"),
    ],
)
async def test_noncacheable_leader_releases_waiter_to_retry(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
    leader_output: str,
    leader_outcome: str,
) -> None:
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

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
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(MemoryCache(), flights)
    leader = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    await grounding_flights.wait_for_event(first_call_started)
    waiter = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    await grounding_flights.wait_until(
        lambda: any(
            event["event"] == "coalesced" for event in grounding_flights.events
        )
    )
    release_first_call.set()
    leader_result, waiter_result = await asyncio.gather(leader, waiter)

    assert leader_result[0][0][1] == leader_outcome
    assert waiter_result[0][0][1] == "grounded"
    assert waiter_result[0][0][0].snippets == ["Recovered"]
    assert llm_calls == 2
    assert flights.active_count == 0


async def test_llm_error_leader_releases_waiter_to_retry(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

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
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(MemoryCache(), flights)
    leader = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    await grounding_flights.wait_for_event(first_call_started)
    waiter = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    await grounding_flights.wait_until(
        lambda: any(
            event["event"] == "coalesced" for event in grounding_flights.events
        )
    )
    release_first_call.set()
    leader_result, waiter_result = await asyncio.gather(leader, waiter)

    assert leader_result[0][0][1] == "fallback:llm_error"
    assert waiter_result[0][0][1] == "grounded"
    assert llm_calls == 2
    assert flights.active_count == 0

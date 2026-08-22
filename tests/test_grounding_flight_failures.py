"""Grounding flight cancellation, rejected writes, and stale releases."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

import jasa.grounding.service as service_module
from jasa.cache.base import CacheBackend
from jasa.cache.memory import MemoryCache
from jasa.grounding.flights import (
    GroundingFlightOwnership,
    GroundingFlightRegistry,
    GroundingWait,
)
from jasa.grounding.service import _TierResponse, ground_results
from tests.conftest import GroundingFlightHarness, tier_answer


class _RejectingCache:
    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        return False

    async def close(self) -> None:
        return None


async def test_leader_cancellation_releases_waiter(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    first_call_started = asyncio.Event()
    first_call_cancelled = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> _TierResponse:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            first_call_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_call_cancelled.set()
                raise
        return tier_answer("Recovered")

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
    leader.cancel()

    with pytest.raises(asyncio.CancelledError):
        await leader
    waiter_result = await waiter
    assert first_call_cancelled.is_set()
    assert waiter_result[0][0][1] == "grounded"
    assert llm_calls == 2
    assert flights.active_count == 0


async def test_cancellation_during_leader_handoff_releases_waiter(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    handoff_reached = asyncio.Event()
    llm_calls = 0
    pause_next_leader = True
    original_worker = service_module._run_grounding_worker

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> _TierResponse:
        nonlocal llm_calls
        llm_calls += 1
        return tier_answer("Recovered")

    async def pause_leader_handoff(
        execution: service_module._GroundingExecution,
        deadline_at: float | None,
        prepared: service_module._GroundingInput | None,
        ownership: GroundingFlightOwnership,
    ) -> tuple[
        float,
        service_module._GroundingInput | None,
        service_module._GroundingAttempt
        | GroundingWait
        | service_module._GroundingLeader,
    ]:
        nonlocal pause_next_leader
        resolution = await original_worker(
            execution, deadline_at, prepared, ownership
        )
        if pause_next_leader and isinstance(
            resolution[2], service_module._GroundingLeader
        ):
            pause_next_leader = False
            handoff_reached.set()
            await asyncio.Event().wait()
        return resolution

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    monkeypatch.setattr(
        "jasa.grounding.service._run_grounding_worker",
        pause_leader_handoff,
    )
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(MemoryCache(), flights)
    leader = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    await grounding_flights.wait_for_event(handoff_reached)
    waiter = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], context)
    )
    await grounding_flights.wait_until(
        lambda: any(
            event["event"] == "coalesced" for event in grounding_flights.events
        )
    )
    leader.cancel()

    with pytest.raises(asyncio.CancelledError):
        await leader
    waiter_result = await waiter

    assert waiter_result[0][0][1] == "grounded"
    assert llm_calls == 2
    assert flights.active_count == 0


async def test_rejected_write_releases_waiter_to_retry(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> _TierResponse:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            first_call_started.set()
            await release_first_call.wait()
        return tier_answer("Grounded")

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(
        cast(CacheBackend, _RejectingCache()),
        flights,
    )
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
    outcomes = await asyncio.gather(leader, waiter)

    assert all(outcome[0][0][1] == "grounded" for outcome in outcomes)
    assert llm_calls == 2
    assert flights.active_count == 0
    assert (
        sum(
            event["event"] == "write_error"
            for event in grounding_flights.events
        )
        == 2
    )


async def test_waiter_cancellation_does_not_cancel_leader(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> _TierResponse:
        nonlocal llm_calls
        llm_calls += 1
        first_call_started.set()
        await release_first_call.wait()
        return tier_answer("Grounded")

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
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert flights.active_count == 1
    release_first_call.set()
    leader_result = await leader
    assert leader_result[0][0][1] == "grounded"
    assert llm_calls == 1
    assert flights.active_count == 0


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

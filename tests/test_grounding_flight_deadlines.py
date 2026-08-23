"""Grounding flight deadlines, worker-slot scheduling, and log redaction."""

from __future__ import annotations

import asyncio
import logging

import pytest

from jasa.cache.memory import MemoryCache
from jasa.config import GroundingSettings
from jasa.grounding.cache import (
    grounding_cache_identity,
    GROUNDING_CACHE_KEY_PREFIX,
    make_grounding_cache_key,
)
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.service import _TierResponse, ground_results
from jasa.server import _fetch_cache_identity
from tests.conftest import (
    GroundingFlightHarness,
    single_tier_waterfall,
    tier_answer,
)


async def test_leader_timeout_releases_longer_budget_waiter(
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
    cache = MemoryCache()
    flights = GroundingFlightRegistry()
    leader_context = grounding_flights.context(
        cache,
        flights,
        settings=GroundingSettings(per_url_deadline_ms=100),
    )
    waiter_context = grounding_flights.context(
        cache,
        flights,
        settings=GroundingSettings(per_url_deadline_ms=500),
    )
    leader = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], leader_context)
    )
    await grounding_flights.wait_for_event(first_call_started)
    waiter = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], waiter_context)
    )
    await grounding_flights.wait_until(
        lambda: any(
            event["event"] == "coalesced" for event in grounding_flights.events
        )
    )
    leader_result, waiter_result = await asyncio.gather(leader, waiter)

    assert first_call_cancelled.is_set()
    assert leader_result[0][0][1] == "fallback:pipeline_timeout"
    assert waiter_result[0][0][1] == "grounded"
    assert llm_calls == 2
    assert flights.active_count == 0


async def test_waiter_timeout_does_not_cancel_longer_budget_leader(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    leader_started = asyncio.Event()
    release_leader = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Shared page content. " * 20)

    async def fake_llm_call(*args: object) -> _TierResponse:
        nonlocal llm_calls
        llm_calls += 1
        leader_started.set()
        await release_leader.wait()
        return tier_answer("Grounded")

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    cache = MemoryCache()
    flights = GroundingFlightRegistry()
    leader_context = grounding_flights.context(
        cache,
        flights,
        settings=GroundingSettings(per_url_deadline_ms=1000),
    )
    waiter_context = grounding_flights.context(
        cache,
        flights,
        settings=GroundingSettings(per_url_deadline_ms=100),
    )
    leader = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], leader_context)
    )
    await grounding_flights.wait_for_event(leader_started)
    waiter = asyncio.create_task(
        ground_results("query", [grounding_flights.result("a")], waiter_context)
    )
    await grounding_flights.wait_until(
        lambda: any(
            event["event"] == "coalesced" for event in grounding_flights.events
        )
    )
    waiter_result = await waiter

    assert waiter_result[0][0][1] == "fallback:pipeline_timeout"
    assert flights.active_count == 1
    release_leader.set()
    leader_result = await leader
    assert leader_result[0][0][1] == "grounded"
    assert llm_calls == 1
    assert flights.active_count == 0


async def test_waiter_worker_reacquisition_respects_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    settings = GroundingSettings(
        concurrency=1,
        per_url_deadline_ms=100,
        top_n=2,
    )
    waiting_content = "Waiting page content. " * 20
    blocker_started = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        content = (
            waiting_content if url == "waiting" else "Blocker content. " * 20
        )
        return grounding_flights.fetch_result(content)

    async def fake_llm_call(*args: object) -> _TierResponse:
        nonlocal llm_calls
        llm_calls += 1
        blocker_started.set()
        await asyncio.Event().wait()
        return tier_answer("unreachable")

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    flights = GroundingFlightRegistry()
    waiting_key = make_grounding_cache_key(
        grounding_cache_identity(
            _fetch_cache_identity("waiting"),
            "query",
            settings.max_content_chars,
            single_tier_waterfall(settings).chain,
        )
    )
    is_leader, completion = flights.claim(waiting_key)
    context = grounding_flights.context(
        MemoryCache(),
        flights,
        settings=settings,
    )
    started_at = asyncio.get_running_loop().time()
    task = asyncio.create_task(
        ground_results(
            "query",
            [
                grounding_flights.result("waiting"),
                grounding_flights.result("blocker"),
            ],
            context,
        )
    )
    await grounding_flights.wait_for_event(blocker_started)
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


async def test_waiters_release_worker_slots_for_distinct_inputs(
    monkeypatch: pytest.MonkeyPatch,
    grounding_flights: GroundingFlightHarness,
) -> None:
    """Two requests coalesce when they are the same page, not the same bytes.

    Every URL here returns byte-identical content, so content is useless as a
    discriminator and only the identity can separate them. That is the point:
    the two `shared` entries must share a flight because they are one page,
    while `distinct` must run its own call despite being indistinguishable by
    content. A content-keyed identity collapses all three into one flight and
    reaches one LLM call, failing the count below.

    Giving `distinct` its own body would let a content-keyed implementation
    pass this test unchanged, which is exactly how the earlier version of it
    stopped discriminating.
    """
    shared_call_started = asyncio.Event()
    distinct_call_started = asyncio.Event()
    release_shared_call = asyncio.Event()
    llm_calls = 0

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result("Identical page content. " * 20)

    async def fake_llm_call(*args: object) -> _TierResponse:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            shared_call_started.set()
            await release_shared_call.wait()
            return tier_answer("Shared grounding")
        distinct_call_started.set()
        return tier_answer("Distinct grounding")

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(
        MemoryCache(),
        flights,
        settings=GroundingSettings(concurrency=2, top_n=3),
    )
    task = asyncio.create_task(
        ground_results(
            "query",
            [
                grounding_flights.result("shared"),
                grounding_flights.result("shared"),
                grounding_flights.result("distinct"),
            ],
            context,
        )
    )
    await grounding_flights.wait_for_event(shared_call_started)
    await grounding_flights.wait_until(
        lambda: any(
            event["event"] == "coalesced" for event in grounding_flights.events
        )
    )
    await grounding_flights.wait_for_event(distinct_call_started)

    assert llm_calls == 2
    release_shared_call.set()
    pairs, stats = await task
    assert [pair[1] for pair in pairs] == ["grounded", "grounded", "grounded"]
    assert stats.grounded_count == 3
    assert flights.active_count == 0


async def test_grounding_cache_logs_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    grounding_flights: GroundingFlightHarness,
) -> None:
    query = "private grounding query"
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()

    async def fake_fetch(engine: object, url: str) -> object:
        return grounding_flights.fetch_result(
            "Private fetched page content. " * 20
        )

    async def fake_llm_call(*args: object) -> _TierResponse:
        first_call_started.set()
        await release_first_call.wait()
        return tier_answer("Private grounded output")

    monkeypatch.setattr(
        "jasa.grounding.service.execute_web_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        "jasa.grounding.service._call_grounding_tier", fake_llm_call
    )
    flights = GroundingFlightRegistry()
    context = grounding_flights.context(
        MemoryCache(),
        flights,
        settings=GroundingSettings(),
    )
    with caplog.at_level(logging.DEBUG, logger="jasa.grounding.cache"):
        leader = asyncio.create_task(
            ground_results(query, [grounding_flights.result("a")], context)
        )
        await grounding_flights.wait_for_event(first_call_started)
        waiter = asyncio.create_task(
            ground_results(query, [grounding_flights.result("a")], context)
        )
        await grounding_flights.wait_until(
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
    assert GROUNDING_CACHE_KEY_PREFIX not in messages
    assert "Grounding cache event=coalesced" in messages
    assert "Grounding cache event=write" in messages
    assert "Grounding cache event=hit" in messages

"""Search miss coalescing, fail-open retries, and cache observability."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping

import pytest

import jasa.search.service as service_module
from jasa.cache.base import make_cache_key, SearchCacheIdentity
from jasa.cache.memory import MemoryCache
from jasa.search.fanout import _FanoutKnobs
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import rank_and_merge, RankedWebResult, SearchResult
from jasa.search.service import (
    run_search,
    SearchError,
    SearchFlightRegistry,
    SearchOptions,
)
from omnifetch.fetch.shared.types import ErrorType, ProviderError


async def _no_sleep(_seconds: float) -> None:
    return None


_KNOBS = _FanoutKnobs(retry_sleep=_no_sleep)


class _SequencedProvider(SearchProvider):
    name = "sequenced"
    secret_env = "SEQUENCED_API_KEY"
    base_url = ""
    default_timeout_s = 1.0

    def __init__(
        self,
        name: str,
        outcomes: list[list[SearchResult] | Exception],
        gates: list[asyncio.Event | None] | None = None,
    ) -> None:
        self.name = name
        self.outcomes = outcomes
        self.gates = gates or []
        self.calls = 0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        call_index = self.calls
        self.calls += 1
        gate = self.gates[call_index] if call_index < len(self.gates) else None
        if gate is not None:
            await gate.wait()
        outcome = self.outcomes[min(call_index, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return list(outcome)


class _BrokenGetCache:
    async def get(self, key: str) -> object | None:
        raise RuntimeError("read failed")

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool | None:
        return None

    async def close(self) -> None:
        return None


class _BrokenSetCache:
    def __init__(self) -> None:
        self.inner = MemoryCache()

    async def get(self, key: str) -> object | None:
        return await self.inner.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool | None:
        raise RuntimeError("write failed")

    async def close(self) -> None:
        return None


class _ObjectGetCache(_BrokenSetCache):
    async def get(self, key: str) -> object | None:
        return object()

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool | None:
        return None


class _RejectingSetCache(_BrokenSetCache):
    async def set(self, key: str, value: str, ttl_seconds: int) -> bool | None:
        return False


def _result(provider: str, suffix: str = "1") -> SearchResult:
    return SearchResult(
        title="title",
        url=f"https://{provider}.example/{suffix}",
        snippet="s" * 80,
        source_provider=provider,
    )


def _api_error(provider: str) -> ProviderError:
    return ProviderError(ErrorType.API_ERROR, "failed", provider)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


def _capture_events(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        service_module,
        "emit_search_cache_metric",
        lambda **fields: events.append(fields),
    )
    return events


async def test_identical_concurrent_misses_dispatch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate])
    flights = SearchFlightRegistry()
    options = SearchOptions(flights=flights)
    cache = MemoryCache()
    events = _capture_events(monkeypatch)

    leader = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(lambda: provider.calls == 1)
    waiters = [
        asyncio.create_task(
            run_search({"a": provider}, cache, "q", options=options)
        )
        for _ in range(4)
    ]
    await _wait_until(
        lambda: sum(event["event"] == "coalesced" for event in events) == 4
    )
    gate.set()

    outcomes = await asyncio.gather(leader, *waiters)

    assert provider.calls == 1
    assert flights.active_count == 0
    assert all(
        outcome.web_results == outcomes[0].web_results for outcome in outcomes
    )


async def test_distinct_identities_do_not_coalesce() -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate, gate])
    flights = SearchFlightRegistry()
    options = SearchOptions(flights=flights)
    cache = MemoryCache()

    first = asyncio.create_task(
        run_search({"a": provider}, cache, "first", options=options)
    )
    second = asyncio.create_task(
        run_search({"a": provider}, cache, "second", options=options)
    )
    await _wait_until(lambda: provider.calls == 2)

    assert flights.active_count == 2
    gate.set()
    await asyncio.gather(first, second)
    assert provider.calls == 2
    assert flights.active_count == 0


async def test_partial_leader_makes_waiter_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    good = _SequencedProvider("good", [[_result("good")]], [gate, None])
    bad = _SequencedProvider("bad", [_api_error("bad")])
    providers = {"good": good, "bad": bad}
    flights = SearchFlightRegistry()
    options = SearchOptions(flights=flights)
    cache = MemoryCache()
    events = _capture_events(monkeypatch)

    leader = asyncio.create_task(
        run_search(providers, cache, "q", options=options)
    )
    await _wait_until(lambda: good.calls == 1)
    waiter = asyncio.create_task(
        run_search(providers, cache, "q", options=options)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    gate.set()
    outcomes = await asyncio.gather(leader, waiter)

    assert good.calls == 2
    assert bad.calls == 2
    assert all(outcome.providers_failed for outcome in outcomes)
    assert flights.active_count == 0


async def test_write_failure_makes_waiter_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate, None])
    flights = SearchFlightRegistry()
    options = SearchOptions(flights=flights)
    cache = _BrokenSetCache()
    events = _capture_events(monkeypatch)

    leader = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(lambda: provider.calls == 1)
    waiter = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    gate.set()
    await asyncio.gather(leader, waiter)

    assert provider.calls == 2
    assert flights.active_count == 0
    assert sum(event["event"] == "write_error" for event in events) == 2


async def test_all_failed_leader_releases_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider(
        "a", [_api_error("a"), [_result("a")]], [gate, None]
    )
    flights = SearchFlightRegistry()
    options = SearchOptions(flights=flights)
    cache = MemoryCache()
    events = _capture_events(monkeypatch)

    leader = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(lambda: provider.calls == 1)
    waiter = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    gate.set()

    with pytest.raises(SearchError, match="All configured"):
        await leader
    outcome = await waiter
    assert outcome.providers_succeeded[0].provider == "a"
    assert provider.calls == 2
    assert flights.active_count == 0


async def test_leader_cancellation_releases_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate, None])
    flights = SearchFlightRegistry()
    options = SearchOptions(flights=flights)
    cache = MemoryCache()
    events = _capture_events(monkeypatch)

    leader = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(lambda: provider.calls == 1)
    waiter = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    leader.cancel()

    with pytest.raises(asyncio.CancelledError):
        await leader
    outcome = await waiter
    assert outcome.providers_succeeded[0].provider == "a"
    assert provider.calls == 2
    assert flights.active_count == 0


async def test_waiter_cancellation_does_not_cancel_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate])
    flights = SearchFlightRegistry()
    options = SearchOptions(flights=flights)
    cache = MemoryCache()
    events = _capture_events(monkeypatch)

    leader = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(lambda: provider.calls == 1)
    waiter = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    gate.set()

    outcome = await leader
    assert outcome.providers_succeeded[0].provider == "a"
    assert provider.calls == 1
    assert flights.active_count == 0


async def test_waiter_keeps_its_own_shorter_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate])
    flights = SearchFlightRegistry()
    cache = MemoryCache()
    events = _capture_events(monkeypatch)

    leader = asyncio.create_task(
        run_search(
            {"a": provider},
            cache,
            "q",
            options=SearchOptions(timeout_ms=1000, flights=flights),
        )
    )
    await _wait_until(lambda: provider.calls == 1)
    waiter = asyncio.create_task(
        run_search(
            {"a": provider},
            cache,
            "q",
            options=SearchOptions(timeout_ms=10, flights=flights),
        )
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )

    with pytest.raises(SearchError, match="All configured"):
        await waiter
    assert flights.active_count == 1
    gate.set()
    await leader
    assert provider.calls == 1
    assert flights.active_count == 0


async def test_waiter_retry_uses_remaining_original_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate])
    flights = SearchFlightRegistry()
    cache = MemoryCache()
    events = _capture_events(monkeypatch)
    ticks = [0.0]
    knobs = _FanoutKnobs(
        retry_sleep=_no_sleep,
        clock=lambda: ticks[0],
    )
    key = make_cache_key(SearchCacheIdentity("q", False, False, ("a",), None))
    is_leader, completion = flights.claim(key)

    waiter = asyncio.create_task(
        run_search(
            {"a": provider},
            cache,
            "q",
            options=SearchOptions(timeout_ms=100, flights=flights),
            knobs=knobs,
        )
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    ticks[0] = 0.09
    flights.release(key, completion)

    with pytest.raises(SearchError, match="All configured"):
        await waiter
    assert is_leader is True
    assert provider.calls == 1
    assert flights.active_count == 0


async def test_expired_budget_rejects_before_waiting() -> None:
    provider = _SequencedProvider("a", [[_result("a")]])
    flights = SearchFlightRegistry()
    cache = MemoryCache()
    key = make_cache_key(SearchCacheIdentity("q", False, False, ("a",), None))
    _is_leader, completion = flights.claim(key)
    ticks = iter([0.0, 0.02])
    knobs = _FanoutKnobs(
        retry_sleep=_no_sleep,
        clock=lambda: next(ticks),
    )

    with pytest.raises(SearchError, match="All configured"):
        await run_search(
            {"a": provider},
            cache,
            "q",
            options=SearchOptions(timeout_ms=10, flights=flights),
            knobs=knobs,
        )

    assert provider.calls == 0
    flights.release(key, completion)
    assert flights.active_count == 0


async def test_expired_budget_rejects_before_dispatch() -> None:
    provider = _SequencedProvider("a", [[_result("a")]])
    ticks = iter([0.0, 0.02])
    knobs = _FanoutKnobs(
        retry_sleep=_no_sleep,
        clock=lambda: next(ticks),
    )

    with pytest.raises(SearchError, match="All configured"):
        await run_search(
            {"a": provider},
            MemoryCache(),
            "q",
            options=SearchOptions(timeout_ms=10),
            knobs=knobs,
        )

    assert provider.calls == 0


async def test_leader_timeout_releases_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate, None])
    flights = SearchFlightRegistry()
    leader_options = SearchOptions(timeout_ms=10, flights=flights)
    waiter_options = SearchOptions(timeout_ms=1000, flights=flights)
    cache = MemoryCache()
    events = _capture_events(monkeypatch)

    leader = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=leader_options)
    )
    await _wait_until(lambda: provider.calls == 1)
    waiter = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=waiter_options)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )

    with pytest.raises(SearchError, match="All configured"):
        await leader
    outcome = await waiter
    assert outcome.providers_succeeded[0].provider == "a"
    assert provider.calls == 2
    assert flights.active_count == 0


async def test_unexpected_leader_error_releases_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    provider = _SequencedProvider("a", [[_result("a")]], [gate, None])
    flights = SearchFlightRegistry()
    options = SearchOptions(flights=flights)
    cache = MemoryCache()
    events = _capture_events(monkeypatch)
    original_rank = rank_and_merge
    rank_calls = 0

    def rank(
        results_by_provider: Mapping[str, list[SearchResult]],
        query: str,
        skip_quality_filter: bool = False,
    ) -> list[RankedWebResult]:
        nonlocal rank_calls
        rank_calls += 1
        if rank_calls == 1:
            raise RuntimeError("rank failed")
        return original_rank(results_by_provider, query, skip_quality_filter)

    monkeypatch.setattr(service_module, "rank_and_merge", rank)
    leader = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(lambda: provider.calls == 1)
    waiter = asyncio.create_task(
        run_search({"a": provider}, cache, "q", options=options)
    )
    await _wait_until(
        lambda: any(event["event"] == "coalesced" for event in events)
    )
    gate.set()

    with pytest.raises(RuntimeError, match="rank failed"):
        await leader
    outcome = await waiter
    assert outcome.providers_succeeded[0].provider == "a"
    assert provider.calls == 2
    assert flights.active_count == 0


async def test_backend_failures_emit_bounded_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _capture_events(monkeypatch)
    provider = _SequencedProvider("a", [[_result("a")]])

    await run_search({"a": provider}, _BrokenGetCache(), "read", knobs=_KNOBS)
    await run_search({"a": provider}, _ObjectGetCache(), "object", knobs=_KNOBS)
    await run_search({"a": provider}, _BrokenSetCache(), "write", knobs=_KNOBS)
    await run_search(
        {"a": provider}, _RejectingSetCache(), "reject", knobs=_KNOBS
    )

    assert {event["event"] for event in events} >= {
        "read_error",
        "miss",
        "write",
        "write_error",
    }
    assert {event.get("error_type") for event in events} >= {
        "RuntimeError",
        "BackendRejected",
    }


async def test_cache_logs_never_include_query_or_key_material(
    caplog: pytest.LogCaptureFixture,
) -> None:
    query = "private-query-material"
    provider = _SequencedProvider("a", [[_result("a")]])
    cache = MemoryCache()

    with caplog.at_level(logging.DEBUG):
        await run_search({"a": provider}, cache, query, knobs=_KNOBS)
        await run_search({"a": provider}, cache, query, knobs=_KNOBS)

    messages = "\n".join(caplog.messages)
    assert query not in messages
    assert "jasa:search" not in messages
    assert "Search cache event=miss" in messages
    assert "Search cache event=write" in messages
    assert "Search cache event=hit" in messages


async def test_stale_release_does_not_remove_new_flight() -> None:
    flights = SearchFlightRegistry()
    first_leader, first = flights.claim("key")
    flights.release("key", first)
    second_leader, second = flights.claim("key")

    flights.release("key", first)

    assert first_leader is True
    assert second_leader is True
    assert flights.active_count == 1
    flights.release("key", second)
    assert flights.active_count == 0

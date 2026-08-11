"""Search service: cache short-circuit, error taxonomy, complete-fanout gate."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from jasa.cache.memory import MemoryCache
from jasa.grounding.service import (
    GroundingContext,
    GroundingOutcome,
    GroundingStats,
)
from jasa.search.fanout import _FanoutKnobs
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import RankedWebResult, SearchResult
from jasa.search.service import (
    _deserialize_outcome,
    run_search,
    SearchError,
    SearchOptions,
)
from omnifetch.fetch.shared.types import ErrorType, ProviderError


async def _no_sleep(_seconds: float) -> None:
    return None


_KNOBS = _FanoutKnobs(retry_sleep=_no_sleep)


class Fake(SearchProvider):
    name = "fake"
    secret_env = "FAKE"
    base_url = ""
    default_timeout_s = 1.0

    def __init__(
        self,
        name: str,
        *,
        ok: list[SearchResult] | None = None,
        error: ProviderError | None = None,
    ) -> None:
        self.name = name
        self._ok = ok or []
        self._error = error
        self.calls = 0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return list(self._ok)


def _r(provider: str, url: str) -> SearchResult:
    return SearchResult(url, url, "s", provider)


def _long_r(provider: str, url: str) -> SearchResult:
    return SearchResult(url, url, "s" * 60, provider)


def _grounding_context() -> GroundingContext:
    return cast(GroundingContext, object())


class _BrokenGetCache:
    async def get(self, key: str) -> str | None:
        raise RuntimeError("boom")

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None

    async def close(self) -> None:
        return None


class _BrokenSetCache:
    def __init__(self) -> None:
        self.inner = MemoryCache()

    async def get(self, key: str) -> str | None:
        return await self.inner.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        raise RuntimeError("boom")

    async def close(self) -> None:
        return None


async def test_no_providers_raises() -> None:
    with pytest.raises(SearchError) as exc:
        await run_search({}, MemoryCache(), "q", knobs=_KNOBS)
    assert exc.value.kind == "no_providers"


async def test_all_failed_raises() -> None:
    p = Fake("p", error=ProviderError(ErrorType.API_ERROR, "e", "p"))
    with pytest.raises(SearchError) as exc:
        await run_search({"p": p}, MemoryCache(), "q", knobs=_KNOBS)
    assert exc.value.kind == "all_failed"


async def test_complete_fanout_cached_then_hit() -> None:
    a = Fake("a", ok=[_r("a", "https://a.com/1")])
    b = Fake("b", ok=[_r("b", "https://b.com/1")])
    cache = MemoryCache()
    outcome = await run_search({"a": a, "b": b}, cache, "q", knobs=_KNOBS)
    assert {s.provider for s in outcome.providers_succeeded} == {"a", "b"}
    assert a.calls == 1 and b.calls == 1
    cached = await run_search({"a": a, "b": b}, cache, "q", knobs=_KNOBS)
    assert a.calls == 1 and b.calls == 1
    assert cached.web_results == outcome.web_results


async def test_partial_failure_is_not_cached() -> None:
    ok = Fake("ok", ok=[_r("ok", "https://ok.com/1")])
    bad = Fake("bad", error=ProviderError(ErrorType.API_ERROR, "e", "bad"))
    cache = MemoryCache()
    outcome = await run_search({"ok": ok, "bad": bad}, cache, "q", knobs=_KNOBS)
    assert [s.provider for s in outcome.providers_succeeded] == ["ok"]
    assert [f.provider for f in outcome.providers_failed] == ["bad"]
    await run_search({"ok": ok, "bad": bad}, cache, "q", knobs=_KNOBS)
    assert ok.calls == 2


async def test_cache_read_failure_is_a_miss() -> None:
    a = Fake("a", ok=[_r("a", "https://a.com/1")])
    outcome = await run_search({"a": a}, _BrokenGetCache(), "q", knobs=_KNOBS)
    assert a.calls == 1
    assert [s.provider for s in outcome.providers_succeeded] == ["a"]


async def test_cache_write_failure_is_swallowed() -> None:
    a = Fake("a", ok=[_r("a", "https://a.com/1")])
    outcome = await run_search({"a": a}, _BrokenSetCache(), "q", knobs=_KNOBS)
    assert [s.provider for s in outcome.providers_succeeded] == ["a"]


def test_deserialize_query_mismatch_is_none() -> None:
    record = {
        "query": "other",
        "total_duration_ms": 0,
        "providers_succeeded": [],
        "providers_failed": [],
        "web_results": [],
    }
    assert _deserialize_outcome(record, "q") is None


def test_deserialize_malformed_record_is_none() -> None:
    assert _deserialize_outcome({"query": "q"}, "q") is None
    bad = {
        "query": "q",
        "total_duration_ms": 0,
        "providers_succeeded": [],
        "providers_failed": [],
        "web_results": [{"title": "t"}],
    }
    assert _deserialize_outcome(bad, "q") is None


class _JunkCache:
    async def get(self, key: str) -> str | None:
        return "not json"

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None

    async def close(self) -> None:
        return None


async def test_invalid_json_cache_is_a_miss() -> None:
    a = Fake("a", ok=[_r("a", "https://a.com/1")])
    outcome = await run_search({"a": a}, _JunkCache(), "q", knobs=_KNOBS)
    assert a.calls == 1
    assert [s.provider for s in outcome.providers_succeeded] == ["a"]


async def test_grounding_without_search_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ground(
        query: str,
        ranked: list[RankedWebResult],
        context: GroundingContext,
    ) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
        grounded = replace(
            ranked[0], snippets=["grounded"], snippet_source="grounded"
        )
        return [(grounded, "grounded")], GroundingStats(0, 1, len(ranked))

    monkeypatch.setattr("jasa.search.service.ground_results", ground)
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    options = SearchOptions(want_grounding=True, grounding=_grounding_context())
    outcome = await run_search(
        {"a": provider}, MemoryCache(), "q", options=options, knobs=_KNOBS
    )
    assert outcome.web_results[0].snippet_source == "grounded"


async def test_grounding_with_remaining_search_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ground(
        query: str,
        ranked: list[RankedWebResult],
        context: GroundingContext,
    ) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
        return [], GroundingStats(0, 0, len(ranked))

    monkeypatch.setattr("jasa.search.service.ground_results", ground)
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    options = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
        timeout_ms=10_000,
    )
    outcome = await run_search(
        {"a": provider}, MemoryCache(), "q", options=options, knobs=_KNOBS
    )
    assert outcome.web_results[0].snippet_source is None


async def test_grounding_timeout_blocks_cache_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ground(*_args: object) -> None:
        raise TimeoutError

    monkeypatch.setattr("jasa.search.service.ground_results", ground)
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    cache = MemoryCache()
    options = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
        timeout_ms=10_000,
    )
    await run_search({"a": provider}, cache, "q", options=options, knobs=_KNOBS)
    await run_search({"a": provider}, cache, "q", options=options, knobs=_KNOBS)
    assert provider.calls == 2


async def test_grounding_skipped_when_search_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_grounding(*_args: object) -> None:
        raise AssertionError("grounding should not run")

    ticks = iter([0.0, 0.0, 0.0, 1.0, 1.0])
    knobs = _FanoutKnobs(retry_sleep=_no_sleep, clock=lambda: next(ticks))
    monkeypatch.setattr(
        "jasa.search.service.ground_results", unexpected_grounding
    )
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    options = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
        timeout_ms=1,
    )
    outcome = await run_search(
        {"a": provider}, MemoryCache(), "q", options=options, knobs=knobs
    )
    assert outcome.web_results[0].snippet_source is None

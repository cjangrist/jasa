"""Search service: cache short-circuit, error taxonomy, complete-fanout gate."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from jasa.cache.base import SearchCacheIdentity, TTL_SECONDS
from jasa.cache.memory import MemoryCache
from jasa.config import GroundingSettings
from jasa.grounding.service import (
    GroundingContext,
    GroundingOutcome,
    GroundingStats,
)
from jasa.search.fanout import (
    _FanoutKnobs,
    DispatchResult,
    ProviderFailure,
    ProviderSuccess,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import RankedWebResult, SearchResult
from jasa.search.service import (
    _deserialize_outcome,
    _serialize,
    run_search,
    SearchError,
    SearchOptions,
    SearchOutcome,
)
from omnifetch.fetch.shared.types import ErrorType, ProviderError
from tests.conftest import single_tier_waterfall


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


def _grounding_context(
    config: GroundingSettings | None = None,
) -> GroundingContext:
    resolved = config or GroundingSettings()
    return cast(
        GroundingContext,
        SimpleNamespace(
            config=resolved,
            waterfall=single_tier_waterfall(resolved),
        ),
    )


def _cache_identity(
    *,
    query: str = "q",
    providers: tuple[str, ...] = ("a", "b"),
    grounding: bool = False,
    grounding_fingerprint: str | None = None,
) -> SearchCacheIdentity:
    return SearchCacheIdentity(
        query=query,
        skip_quality_filter=False,
        grounding=grounding,
        providers=providers,
        grounding_fingerprint=grounding_fingerprint,
    )


def _cached_record() -> tuple[dict[str, object], SearchOutcome]:
    outcome = SearchOutcome(
        query="q",
        total_duration_ms=12,
        providers_succeeded=[
            ProviderSuccess("a", 7),
            ProviderSuccess("b", 9),
        ],
        providers_failed=[],
        web_results=[
            RankedWebResult(
                title="Title",
                url="https://example.com",
                snippets=["One", "Two"],
                source_providers=["a", "b"],
                score=0.25,
                snippet_source="aggregated",
            )
        ],
    )
    return json.loads(_serialize(outcome, _cache_identity())), outcome


def _record_mapping(record: dict[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], record[key])


def _record_list(
    record: dict[str, object], key: str
) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], record[key])


class _BrokenGetCache:
    async def get(self, key: str) -> str | None:
        raise RuntimeError("boom")

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None

    async def close(self) -> None:
        return None


class _RecordingCache(MemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.write_ttls: list[int] = []

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.write_ttls.append(ttl_seconds)
        await super().set(key, value, ttl_seconds)


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


async def test_structured_fanout_deadline_raises_deadline_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def dispatch(*_args: object, **_kwargs: object) -> DispatchResult:
        return DispatchResult(
            {},
            [],
            [
                ProviderFailure(
                    "p",
                    "Provider exceeded the allotted search budget",
                    10,
                    deadline_exceeded=True,
                )
            ],
        )

    monkeypatch.setattr("jasa.search.service.dispatch_to_providers", dispatch)
    provider = Fake("p", ok=[_r("p", "https://p.example/1")])

    with pytest.raises(SearchError) as exc:
        await run_search({"p": provider}, MemoryCache(), "q", knobs=_KNOBS)

    assert exc.value.kind == "deadline_exceeded"
    assert provider.calls == 0


async def test_complete_fanout_cached_then_hit() -> None:
    a = Fake("a", ok=[_r("a", "https://a.com/1")])
    b = Fake("b", ok=[_r("b", "https://b.com/1")])
    cache = _RecordingCache()
    outcome = await run_search({"a": a, "b": b}, cache, "q", knobs=_KNOBS)
    assert {s.provider for s in outcome.providers_succeeded} == {"a", "b"}
    assert a.calls == 1 and b.calls == 1
    cached = await run_search({"a": a, "b": b}, cache, "q", knobs=_KNOBS)
    assert a.calls == 1 and b.calls == 1
    assert cached.web_results == outcome.web_results
    assert cache.write_ttls == [TTL_SECONDS]


async def test_complete_search_write_uses_requested_ttl() -> None:
    provider = Fake("a", ok=[_r("a", "https://a.com/1")])
    cache = _RecordingCache()
    options = SearchOptions(cache_ttl_seconds=321)

    await run_search({"a": provider}, cache, "q", options=options, knobs=_KNOBS)

    assert cache.write_ttls == [321]


async def test_progress_reporting_is_bounded_and_cannot_fail_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled_reports = 0

    async def blocked_reporter(
        _progress: float,
        _total: float | None,
        _message: str | None,
    ) -> None:
        nonlocal cancelled_reports
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_reports += 1
            raise

    monkeypatch.setattr(
        "jasa.search.service._PROGRESS_REPORT_TIMEOUT_SECONDS", 0.001
    )
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    options = SearchOptions(progress_reporter=blocked_reporter)

    outcome = await run_search(
        {"a": provider}, MemoryCache(), "q", options=options, knobs=_KNOBS
    )

    assert outcome.web_results
    assert cancelled_reports == 5


@pytest.mark.parametrize("timeout_ms", [None, 10_000])
async def test_grounding_progress_maps_completion_onto_search_range(
    monkeypatch: pytest.MonkeyPatch,
    timeout_ms: int | None,
) -> None:
    progress_updates: list[tuple[float, float | None, str | None]] = []

    async def reporter(
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        progress_updates.append((progress, total, message))

    async def ground(
        query: str,
        ranked: list[RankedWebResult],
        context: GroundingContext,
        deadline_at: float | None = None,
        *,
        progress_reporter: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
        assert (deadline_at is None) is (timeout_ms is None)
        assert progress_reporter is not None
        await progress_reporter(1, 1)
        grounded = replace(
            ranked[0], snippets=["grounded"], snippet_source="grounded"
        )
        return [(grounded, "grounded")], GroundingStats(0, 1, 1)

    monkeypatch.setattr("jasa.search.service.ground_results", ground)
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    options = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
        timeout_ms=timeout_ms,
        progress_reporter=reporter,
    )

    await run_search(
        {"a": provider}, MemoryCache(), "q", options=options, knobs=_KNOBS
    )

    assert (40, 100, "Grounding 1 ranked results") in progress_updates
    assert (90, 100, "Grounding results complete: 1/1") in progress_updates


async def test_provider_identity_changes_dispatch_and_hits() -> None:
    a = Fake("a", ok=[_r("a", "https://a.com/1")])
    b = Fake("b", ok=[_r("b", "https://b.com/1")])
    cache = MemoryCache()

    await run_search({"a": a}, cache, "q", knobs=_KNOBS)
    await run_search({"a": a, "b": b}, cache, "q", knobs=_KNOBS)
    await run_search({"a": a, "b": b}, cache, "q", knobs=_KNOBS)
    await run_search({"b": b, "a": a}, cache, "q", knobs=_KNOBS)

    assert a.calls == 3
    assert b.calls == 2


async def test_grounding_semantics_change_forces_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ground(
        query: str,
        ranked: list[RankedWebResult],
        context: GroundingContext,
        deadline_at: float | None = None,
    ) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
        return (
            [(result, "grounded") for result in ranked],
            GroundingStats(0, len(ranked), len(ranked)),
        )

    monkeypatch.setattr("jasa.search.service.ground_results", ground)
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    cache = MemoryCache()
    first = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(GroundingSettings(llm_model="first")),
    )
    second = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(GroundingSettings(llm_model="second")),
    )

    await run_search({"a": provider}, cache, "q", options=first, knobs=_KNOBS)
    await run_search({"a": provider}, cache, "q", options=first, knobs=_KNOBS)
    await run_search({"a": provider}, cache, "q", options=second, knobs=_KNOBS)

    assert provider.calls == 2


async def test_contextless_grounding_has_separate_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grounding_calls = 0

    async def ground(
        query: str,
        ranked: list[RankedWebResult],
        context: GroundingContext,
        deadline_at: float | None = None,
    ) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
        nonlocal grounding_calls
        grounding_calls += 1
        return (
            [(result, "grounded") for result in ranked],
            GroundingStats(0, len(ranked), len(ranked)),
        )

    monkeypatch.setattr("jasa.search.service.ground_results", ground)
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    cache = MemoryCache()
    contextless = SearchOptions(want_grounding=True)
    contextful = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
    )

    await run_search(
        {"a": provider}, cache, "q", options=contextless, knobs=_KNOBS
    )
    await run_search(
        {"a": provider}, cache, "q", options=contextless, knobs=_KNOBS
    )
    await run_search(
        {"a": provider}, cache, "q", options=contextful, knobs=_KNOBS
    )

    assert provider.calls == 2
    assert grounding_calls == 1


async def test_no_providers_rejects_existing_cached_result() -> None:
    provider = Fake("a", ok=[_r("a", "https://a.com/1")])
    cache = MemoryCache()
    await run_search({"a": provider}, cache, "q", knobs=_KNOBS)

    with pytest.raises(SearchError) as exc:
        await run_search({}, cache, "q", knobs=_KNOBS)

    assert exc.value.kind == "no_providers"


async def test_partial_failure_is_not_cached() -> None:
    ok = Fake("ok", ok=[_r("ok", "https://ok.com/1")])
    bad = Fake("bad", error=ProviderError(ErrorType.API_ERROR, "e", "bad"))
    cache = _RecordingCache()
    outcome = await run_search({"ok": ok, "bad": bad}, cache, "q", knobs=_KNOBS)
    assert [s.provider for s in outcome.providers_succeeded] == ["ok"]
    assert [f.provider for f in outcome.providers_failed] == ["bad"]
    await run_search({"ok": ok, "bad": bad}, cache, "q", knobs=_KNOBS)
    assert ok.calls == 2
    assert cache.write_ttls == []


async def test_cache_read_failure_is_a_miss() -> None:
    a = Fake("a", ok=[_r("a", "https://a.com/1")])
    outcome = await run_search({"a": a}, _BrokenGetCache(), "q", knobs=_KNOBS)
    assert a.calls == 1
    assert [s.provider for s in outcome.providers_succeeded] == ["a"]


async def test_cache_write_failure_is_swallowed() -> None:
    a = Fake("a", ok=[_r("a", "https://a.com/1")])
    outcome = await run_search({"a": a}, _BrokenSetCache(), "q", knobs=_KNOBS)
    assert [s.provider for s in outcome.providers_succeeded] == ["a"]


def test_strict_cache_record_round_trip_retains_nested_fields() -> None:
    record, outcome = _cached_record()

    assert _deserialize_outcome(record, _cache_identity()) == outcome


def test_legacy_wrong_version_malformed_and_extra_records_are_misses() -> None:
    valid, _outcome = _cached_record()
    cases: list[object] = [
        valid["outcome"],
        {**valid, "schema_version": 1},
        {**valid, "schema_version": 2},
        {**valid, "unexpected": True},
        {"schema_version": 3},
        [valid],
    ]

    for record in cases:
        assert _deserialize_outcome(record, _cache_identity()) is None


def test_identity_query_and_nested_field_drift_are_misses() -> None:
    valid, _outcome = _cached_record()
    wrong_identity = copy.deepcopy(valid)
    _record_mapping(wrong_identity, "identity")["providers"] = ["other"]
    wrong_query = copy.deepcopy(valid)
    _record_mapping(wrong_query, "outcome")["query"] = "other"
    nested_extra = copy.deepcopy(valid)
    nested_outcome = _record_mapping(nested_extra, "outcome")
    _record_list(nested_outcome, "providers_succeeded")[0]["extra"] = True

    for record in (wrong_identity, wrong_query, nested_extra):
        assert _deserialize_outcome(record, _cache_identity()) is None


def test_incomplete_or_inconsistent_outcomes_are_misses() -> None:
    valid, _outcome = _cached_record()
    partial = copy.deepcopy(valid)
    partial_outcome = _record_mapping(partial, "outcome")
    partial_outcome["providers_failed"] = [
        {"provider": "b", "error": "failed", "duration_ms": 9}
    ]
    incomplete = copy.deepcopy(valid)
    incomplete_outcome = _record_mapping(incomplete, "outcome")
    incomplete_outcome["providers_succeeded"] = [
        {"provider": "a", "duration_ms": 7}
    ]
    unknown_source = copy.deepcopy(valid)
    unknown_results = _record_list(
        _record_mapping(unknown_source, "outcome"), "web_results"
    )
    unknown_results[0]["source_providers"] = ["other"]
    missing_source = copy.deepcopy(valid)
    missing_results = _record_list(
        _record_mapping(missing_source, "outcome"), "web_results"
    )
    missing_results[0]["source_providers"] = []

    for record in (partial, incomplete, unknown_source, missing_source):
        assert _deserialize_outcome(record, _cache_identity()) is None


def test_wrong_nested_types_and_values_are_misses() -> None:
    valid, _outcome = _cached_record()
    wrong_duration = copy.deepcopy(valid)
    _record_mapping(wrong_duration, "outcome")["total_duration_ms"] = "12"
    negative_duration = copy.deepcopy(valid)
    _record_mapping(negative_duration, "outcome")["total_duration_ms"] = -1
    wrong_list_item = copy.deepcopy(valid)
    wrong_results = _record_list(
        _record_mapping(wrong_list_item, "outcome"), "web_results"
    )
    wrong_results[0]["snippets"] = [1]
    wrong_literal = copy.deepcopy(valid)
    literal_results = _record_list(
        _record_mapping(wrong_literal, "outcome"), "web_results"
    )
    literal_results[0]["snippet_source"] = "native"
    nonfinite_score = copy.deepcopy(valid)
    nonfinite_results = _record_list(
        _record_mapping(nonfinite_score, "outcome"), "web_results"
    )
    nonfinite_results[0]["score"] = float("nan")

    for record in (
        wrong_duration,
        negative_duration,
        wrong_list_item,
        wrong_literal,
        nonfinite_score,
    ):
        assert _deserialize_outcome(record, _cache_identity()) is None


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
        deadline_at: float | None = None,
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
        deadline_at: float | None = None,
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
    assert outcome.web_results[0].snippet_source == "aggregated"


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


async def test_exhausted_grounding_budget_returns_aggregated_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_grounding(*_args: object) -> None:
        raise AssertionError("grounding should not run")

    ticks = iter([0.0] * 6)
    knobs = _FanoutKnobs(
        retry_sleep=_no_sleep,
        clock=lambda: next(ticks, 1.0),
    )
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

    assert outcome.web_results[0].snippet_source == "fallback"
    assert outcome.grounding is not None
    assert outcome.grounding.attempted == 1
    assert outcome.grounding.grounded == 0
    assert outcome.grounding.outcomes == {"fallback:pipeline_timeout": 1}


async def test_expired_budget_keeps_snippets_that_were_already_paid_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage that runs out of budget must keep what it already produced.

    The stage used to be wrapped in the caller's timeout, so one slow URL
    discarded every snippet its siblings had already paid an LLM to write.
    """
    observed_deadline: list[float | None] = []

    async def partial_grounding(
        query: str,
        ranked: list[RankedWebResult],
        context: GroundingContext,
        deadline_at: float | None = None,
    ) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
        observed_deadline.append(deadline_at)
        assert deadline_at is not None
        await asyncio.sleep(
            max(0.0, deadline_at - asyncio.get_running_loop().time())
        )
        grounded = replace(
            ranked[0], snippets=["paid for"], snippet_source="grounded"
        )
        return (
            [(grounded, "grounded"), (ranked[1], "fallback:pipeline_timeout")],
            GroundingStats(1, 1, 2),
        )

    monkeypatch.setattr("jasa.search.service.ground_results", partial_grounding)
    provider = Fake(
        "a",
        ok=[
            _long_r("a", "https://a.com/1"),
            _long_r("a", "https://a.com/2"),
        ],
    )
    options = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
        timeout_ms=4_000,
    )

    outcome = await run_search(
        {"a": provider}, MemoryCache(), "q", options=options
    )

    assert observed_deadline and observed_deadline[0] is not None
    by_url = {r.url: r for r in outcome.web_results}
    assert by_url["https://a.com/1"].snippets == ["paid for"]
    assert by_url["https://a.com/1"].snippet_source == "grounded"
    assert by_url["https://a.com/2"].snippet_source != "grounded"
    assert outcome.grounding is not None
    assert outcome.grounding.grounded == 1
    assert outcome.grounding.attempted == 2


async def test_grounding_overrun_degrades_to_ungrounded_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False

    async def slow_grounding(*_args: object) -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled = True
            raise

    monkeypatch.setattr("jasa.search.service.ground_results", slow_grounding)
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    options = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
        timeout_ms=200,
    )

    outcome = await run_search(
        {"a": provider}, MemoryCache(), "q", options=options
    )

    assert cancelled
    assert len(outcome.web_results) == 1
    result = outcome.web_results[0]
    assert result.url == "https://a.com/1"
    assert result.snippet_source != "grounded"
    assert result.snippets == ["s" * 60]
    assert result.source_providers == ["a"]


async def test_grounding_caller_deadline_returns_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_grounding(*_args: object) -> None:
        await asyncio.sleep(1)

    monkeypatch.setattr("jasa.search.service.ground_results", slow_grounding)
    provider = Fake("a", ok=[_long_r("a", "https://a.com/1")])
    options = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
        timeout_ms=10,
    )

    outcome = await run_search(
        {"a": provider}, MemoryCache(), "q", options=options
    )

    assert outcome.web_results[0].snippet_source == "fallback"
    assert outcome.grounding is not None
    assert outcome.grounding.outcomes == {"fallback:pipeline_timeout": 1}


async def test_fanout_cap_applies_when_the_caller_sets_no_deadline() -> None:
    """An uncapped request still bounds the fan-out, leaving grounding time."""
    observed: list[int | None] = []

    async def dispatch(
        providers: object,
        query: str,
        *,
        timeout_ms: int | None = None,
        **_kwargs: object,
    ) -> DispatchResult:
        observed.append(timeout_ms)
        return DispatchResult(
            {"a": [_long_r("a", "https://a.com/1")]},
            [ProviderSuccess("a", 1)],
            [],
        )

    options = SearchOptions(timeout_ms=None, fanout_timeout_ms=40_000)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("jasa.search.service.dispatch_to_providers", dispatch)
        await run_search(
            {"a": Fake("a")}, MemoryCache(), "q", options=options, knobs=_KNOBS
        )

    assert observed == [40_000]


async def test_fanout_cap_never_exceeds_the_caller_deadline() -> None:
    observed: list[int | None] = []

    async def dispatch(
        providers: object,
        query: str,
        *,
        timeout_ms: int | None = None,
        **_kwargs: object,
    ) -> DispatchResult:
        observed.append(timeout_ms)
        return DispatchResult(
            {"a": [_long_r("a", "https://a.com/1")]},
            [ProviderSuccess("a", 1)],
            [],
        )

    options = SearchOptions(timeout_ms=5_000, fanout_timeout_ms=40_000)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("jasa.search.service.dispatch_to_providers", dispatch)
        await run_search(
            {"a": Fake("a")}, MemoryCache(), "q", options=options, knobs=_KNOBS
        )

    assert observed and observed[0] is not None and observed[0] <= 5_000


async def test_dispatch_deadline_is_read_once_not_twice() -> None:
    """A budget that expires between two clock reads must not go unbounded.

    Checking the remaining budget and then recomputing it for the call opens a
    window in which zero reaches the fan-out, where zero used to mean "no
    deadline" -- handing an unbounded spend to the request that had just run
    out of time.
    """
    observed: list[int | None] = []

    async def dispatch(
        providers: object,
        query: str,
        *,
        timeout_ms: int | None = None,
        **_kwargs: object,
    ) -> DispatchResult:
        observed.append(timeout_ms)
        return DispatchResult(
            {"a": [_long_r("a", "https://a.com/1")]},
            [ProviderSuccess("a", 1)],
            [],
        )

    def spent_clock(reads_before_expiry: int) -> Callable[[], float]:
        ticks = iter([0.0] * reads_before_expiry + [5.0] * 8)

        def clock() -> float:
            return next(ticks, 5.0)

        return clock

    for spent_at in range(8):
        knobs = _FanoutKnobs(retry_sleep=_no_sleep, clock=spent_clock(spent_at))
        options = SearchOptions(timeout_ms=1_000, fanout_timeout_ms=40_000)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("jasa.search.service.dispatch_to_providers", dispatch)
            try:
                await run_search(
                    {"a": Fake("a")},
                    MemoryCache(),
                    "q",
                    options=options,
                    knobs=knobs,
                )
            except SearchError as error:
                assert error.kind in {"deadline_exceeded", "all_failed"}

    assert observed, "the fan-out was never reached in any ordering"
    assert 0 not in observed, "a zero deadline reached the fan-out"


async def test_backstop_overrun_reports_the_urls_it_took_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backstopped stage ran; the report must not claim it never tried."""

    async def slow_grounding(*_args: object) -> None:
        await asyncio.sleep(1)

    monkeypatch.setattr("jasa.search.service.ground_results", slow_grounding)
    provider = Fake(
        "a",
        ok=[_long_r("a", "https://a.com/1"), _long_r("a", "https://a.com/2")],
    )
    options = SearchOptions(
        want_grounding=True,
        grounding=_grounding_context(),
        timeout_ms=300,
    )

    outcome = await run_search(
        {"a": provider}, MemoryCache(), "q", options=options
    )

    assert outcome.grounding is not None
    assert outcome.grounding.requested is True
    assert outcome.grounding.attempted == 2
    assert outcome.grounding.grounded == 0
    assert outcome.grounding.outcomes == {"fallback:pipeline_timeout": 2}
    assert all(r.snippet_source == "fallback" for r in outcome.web_results), (
        "a backstopped row still claims grounding never reached it"
    )
    assert [r.snippets for r in outcome.web_results] == [["s" * 60]] * 2

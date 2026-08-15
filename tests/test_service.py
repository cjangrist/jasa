"""Search service: cache short-circuit, error taxonomy, complete-fanout gate."""

from __future__ import annotations

import copy
import json
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
from jasa.search.fanout import _FanoutKnobs, ProviderSuccess
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
    return cast(
        GroundingContext,
        SimpleNamespace(config=config or GroundingSettings()),
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
        {**valid, "unexpected": True},
        {"schema_version": 2},
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

    ticks = iter([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
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

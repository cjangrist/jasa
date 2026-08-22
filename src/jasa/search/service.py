"""Search execution: strict cache read -> fan-out -> rank -> cache write.

MCP and REST share this path. Search cache v3 scopes entries to the exact query,
ordered provider registry, raw/grounded mode, and grounding semantics. Values
use an extra-forbidden versioned envelope and are reconstructed only after all
nested fields and the stored identity validate. ``include_snippets`` and
``timeout_ms`` stay outside the key because the cache stores the full ranked set
before output truncation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    FiniteFloat,
    NonNegativeInt,
    ValidationError,
)

from jasa.cache.base import (
    CacheBackend,
    make_cache_key,
    SearchCacheIdentity,
    should_cache,
    TTL_SECONDS,
)
from jasa.grounding.service import (
    ground_results,
    grounding_semantic_fingerprint,
    GroundingContext,
    GroundingOutcome,
    GroundingStats,
)
from jasa.logging import get_logger
from jasa.observability.metrics import (
    emit_search_cache_metric,
    emit_search_metric,
)
from jasa.search.fanout import (
    _FanoutKnobs,
    dispatch_to_providers,
    DispatchResult,
    ProviderFailure,
    ProviderSuccess,
)
from jasa.search.providers.base import SearchProvider
from jasa.search.ranking import rank_and_merge, RankedWebResult

_LOGGER = get_logger("search.service")

_NO_PROVIDERS_MESSAGE = (
    "No search providers are configured. Set at least one *_API_KEY."
)
_ALL_FAILED_MESSAGE = "All configured search providers failed."
_DEADLINE_EXCEEDED_MESSAGE = "Search request deadline exceeded."
_GROUNDING_BUDGET_SHARE = 0.9
_GROUNDING_HARVEST_SHARE = 0.95
_SEARCH_CACHE_SCHEMA_VERSION: Literal[4] = 4
_STRICT_RECORD_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)
_CacheEvent = Literal[
    "hit",
    "miss",
    "write",
    "read_skipped",
    "write_skipped",
    "read_error",
    "write_error",
    "coalesced",
]


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """What the grounding stage was asked to do and what it produced.

    ``attempted`` is zero when grounding never ran, which a caller cannot
    otherwise distinguish from a run where every page failed. ``outcomes``
    names the reason for each shortfall so a silent zero becomes a readable
    one.
    """

    requested: bool
    attempted: int
    grounded: int
    outcomes: dict[str, int]


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """The full ranked result of one search, before output truncation."""

    query: str
    total_duration_ms: int
    providers_succeeded: list[ProviderSuccess]
    providers_failed: list[ProviderFailure]
    web_results: list[RankedWebResult]
    grounding: GroundingReport | None = None


class SearchError(Exception):
    """A search failed from configuration, providers, or caller deadline."""

    def __init__(
        self,
        message: str,
        *,
        kind: Literal["no_providers", "all_failed", "deadline_exceeded"],
    ) -> None:
        """Record the stable failure category for transport mapping."""
        super().__init__(message)
        self.kind = kind


def _deadline_exceeded_error() -> SearchError:
    """Return the stable error for an exhausted caller budget."""
    return SearchError(_DEADLINE_EXCEEDED_MESSAGE, kind="deadline_exceeded")


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """Search execution + output options bundled to keep call sites lean."""

    skip_quality_filter: bool = False
    want_grounding: bool = False
    timeout_ms: int | None = None
    fanout_timeout_ms: int | None = None
    include_snippets: bool = True
    grounding: GroundingContext | None = None
    cache_ttl_seconds: int = TTL_SECONDS
    flights: SearchFlightRegistry | None = None


_DEFAULT_SEARCH_OPTIONS = SearchOptions()


@dataclass(slots=True)
class SearchFlightRegistry:
    """Composition-owned in-process flights for complete search misses.

    Flights hold loop-bound futures. One registry must serve exactly one event
    loop.
    """

    _flights: dict[str, asyncio.Future[None]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def active_count(self) -> int:
        """Return the number of currently led search identities."""
        return len(self._flights)

    def claim(self, key: str) -> tuple[bool, asyncio.Future[None]]:
        """Return whether this caller leads the identity's current flight."""
        existing = self._flights.get(key)
        if existing is not None:
            return False, existing
        completion = asyncio.get_running_loop().create_future()
        self._flights[key] = completion
        return True, completion

    def release(self, key: str, completion: asyncio.Future[None]) -> None:
        """Remove one flight and release every shielded waiter."""
        if self._flights.get(key) is completion:
            del self._flights[key]
        if not completion.done():
            completion.set_result(None)


@dataclass(frozen=True, slots=True)
class SearchRuntime:
    """Composition-owned providers, cache, TTL, and flight registry."""

    providers: Mapping[str, SearchProvider]
    cache: CacheBackend
    cache_ttl_seconds: int
    flights: SearchFlightRegistry


@dataclass(frozen=True, slots=True)
class _SearchExecution:
    """Immutable inputs shared across one elected search leader."""

    providers: Mapping[str, SearchProvider]
    cache: CacheBackend
    query: str
    identity: SearchCacheIdentity
    key: str
    options: SearchOptions
    knobs: _FanoutKnobs
    started_at: float


class _SearchIdentityRecord(BaseModel):
    """Strict JSON-native copy of the search key identity."""

    model_config = _STRICT_RECORD_CONFIG

    query: str
    skip_quality_filter: bool
    grounding: bool
    providers: list[str]
    grounding_fingerprint: str | None


class _ProviderSuccessRecord(BaseModel):
    """Strict cached provider-success fields."""

    model_config = _STRICT_RECORD_CONFIG

    provider: str
    duration_ms: NonNegativeInt


class _ProviderFailureRecord(BaseModel):
    """Strict cached provider-failure fields."""

    model_config = _STRICT_RECORD_CONFIG

    provider: str
    error: str
    duration_ms: NonNegativeInt


class _RankedWebResultRecord(BaseModel):
    """Strict cached ranked-result fields."""

    model_config = _STRICT_RECORD_CONFIG

    title: str
    url: str
    snippets: list[str]
    source_providers: list[str]
    score: FiniteFloat
    snippet_source: Literal["aggregated", "grounded", "fallback"] | None


class _GroundingReportRecord(BaseModel):
    """Strict cached grounding-stage summary."""

    model_config = _STRICT_RECORD_CONFIG

    requested: bool
    attempted: NonNegativeInt
    grounded: NonNegativeInt
    outcomes: dict[str, NonNegativeInt]


class _SearchOutcomeRecord(BaseModel):
    """Strict complete search outcome payload."""

    model_config = _STRICT_RECORD_CONFIG

    query: str
    total_duration_ms: NonNegativeInt
    providers_succeeded: list[_ProviderSuccessRecord]
    providers_failed: list[_ProviderFailureRecord]
    web_results: list[_RankedWebResultRecord]
    grounding: _GroundingReportRecord | None


class _SearchCacheRecord(BaseModel):
    """Versioned envelope binding an identity to one complete outcome."""

    model_config = _STRICT_RECORD_CONFIG

    schema_version: Literal[4]
    identity: _SearchIdentityRecord
    outcome: _SearchOutcomeRecord


def _elapsed_ms(start: float, now: float) -> int:
    return int((now - start) * 1000)


def _record_cache_event(
    event: _CacheEvent, error_type: str | None = None
) -> None:
    """Log and emit one bounded cache event without key or query material."""
    if error_type is None:
        _LOGGER.debug("Search cache event=%s", event)
        emit_search_cache_metric(event=event)
        return
    _LOGGER.warning("Search cache event=%s error_type=%s", event, error_type)
    emit_search_cache_metric(event=event, error_type=error_type)


def _identity_record(identity: SearchCacheIdentity) -> _SearchIdentityRecord:
    return _SearchIdentityRecord(
        query=identity.query,
        skip_quality_filter=identity.skip_quality_filter,
        grounding=identity.grounding,
        providers=list(identity.providers),
        grounding_fingerprint=identity.grounding_fingerprint,
    )


def _serialize(outcome: SearchOutcome, identity: SearchCacheIdentity) -> str:
    record = _SearchCacheRecord(
        schema_version=_SEARCH_CACHE_SCHEMA_VERSION,
        identity=_identity_record(identity),
        outcome=_SearchOutcomeRecord(
            query=outcome.query,
            total_duration_ms=outcome.total_duration_ms,
            providers_succeeded=[
                _ProviderSuccessRecord(
                    provider=item.provider, duration_ms=item.duration_ms
                )
                for item in outcome.providers_succeeded
            ],
            providers_failed=[
                _ProviderFailureRecord(
                    provider=item.provider,
                    error=item.error,
                    duration_ms=item.duration_ms,
                )
                for item in outcome.providers_failed
            ],
            web_results=[
                _RankedWebResultRecord(
                    title=item.title,
                    url=item.url,
                    snippets=item.snippets,
                    source_providers=item.source_providers,
                    score=item.score,
                    snippet_source=item.snippet_source,
                )
                for item in outcome.web_results
            ],
            grounding=(
                None
                if outcome.grounding is None
                else _GroundingReportRecord(
                    requested=outcome.grounding.requested,
                    attempted=outcome.grounding.attempted,
                    grounded=outcome.grounding.grounded,
                    outcomes=dict(outcome.grounding.outcomes),
                )
            ),
        ),
    )
    return record.model_dump_json()


def _deserialize_outcome(
    record: object, identity: SearchCacheIdentity
) -> SearchOutcome | None:
    try:
        cached = _SearchCacheRecord.model_validate(record)
    except ValidationError:
        return None
    if cached.identity != _identity_record(identity):
        return None
    outcome = cached.outcome
    if outcome.query != identity.query:
        return None
    succeeded_names = tuple(
        item.provider for item in outcome.providers_succeeded
    )
    if outcome.providers_failed or succeeded_names != identity.providers:
        return None
    active_providers = frozenset(identity.providers)
    if any(
        not item.source_providers
        or not set(item.source_providers).issubset(active_providers)
        for item in outcome.web_results
    ):
        return None
    return SearchOutcome(
        query=outcome.query,
        total_duration_ms=outcome.total_duration_ms,
        providers_succeeded=[
            ProviderSuccess(item.provider, item.duration_ms)
            for item in outcome.providers_succeeded
        ],
        providers_failed=[
            ProviderFailure(item.provider, item.error, item.duration_ms)
            for item in outcome.providers_failed
        ],
        web_results=[
            RankedWebResult(
                title=item.title,
                url=item.url,
                snippets=item.snippets,
                source_providers=item.source_providers,
                score=item.score,
                snippet_source=item.snippet_source,
            )
            for item in outcome.web_results
        ],
        grounding=(
            None
            if outcome.grounding is None
            else GroundingReport(
                requested=outcome.grounding.requested,
                attempted=outcome.grounding.attempted,
                grounded=outcome.grounding.grounded,
                outcomes=dict(outcome.grounding.outcomes),
            )
        ),
    )


async def _read_cache(
    cache: CacheBackend, key: str, identity: SearchCacheIdentity
) -> SearchOutcome | None:
    try:
        raw = await cache.get(key)
    except Exception as error:
        _record_cache_event("read_error", type(error).__name__)
        return None
    if raw is None:
        _record_cache_event("miss")
        return None
    if not isinstance(raw, str | bytes | bytearray):
        _record_cache_event("miss")
        return None
    try:
        record = _SearchCacheRecord.model_validate_json(raw)
    except ValidationError:
        _record_cache_event("miss")
        return None
    outcome = _deserialize_outcome(record, identity)
    _record_cache_event("hit" if outcome is not None else "miss")
    return outcome


async def _write_cache(
    cache: CacheBackend,
    key: str,
    identity: SearchCacheIdentity,
    outcome: SearchOutcome,
    ttl_seconds: int,
) -> bool:
    try:
        stored = await cache.set(
            key, _serialize(outcome, identity), ttl_seconds
        )
    except Exception as error:
        _record_cache_event("write_error", type(error).__name__)
        return False
    if stored is False:
        _record_cache_event("write_error", "BackendRejected")
        return False
    _record_cache_event("write")
    return True


async def _read_cache_with_remaining_budget(
    execution: _SearchExecution,
) -> SearchOutcome | None:
    """Read one cache entry within the caller's original deadline."""
    remaining_ms = _remaining_timeout_ms(
        execution.options,
        execution.started_at,
        execution.knobs,
    )
    if remaining_ms == 0:
        raise _deadline_exceeded_error()
    try:
        if remaining_ms is None:
            cached = await _read_cache(
                execution.cache, execution.key, execution.identity
            )
        else:
            async with asyncio.timeout(remaining_ms / 1000):
                cached = await _read_cache(
                    execution.cache, execution.key, execution.identity
                )
    except TimeoutError as error:
        _record_cache_event("read_skipped")
        raise _deadline_exceeded_error() from error
    if (
        remaining_ms is not None
        and _remaining_timeout_ms(
            execution.options,
            execution.started_at,
            execution.knobs,
        )
        == 0
    ):
        raise _deadline_exceeded_error()
    return cached


async def _write_cache_with_remaining_budget(
    execution: _SearchExecution,
    outcome: SearchOutcome,
) -> bool:
    """Write a complete outcome without delaying the caller past deadline."""
    remaining_ms = _remaining_timeout_ms(
        execution.options,
        execution.started_at,
        execution.knobs,
    )
    if remaining_ms == 0:
        _record_cache_event("write_skipped")
        return False
    if remaining_ms is None:
        return await _write_cache(
            execution.cache,
            execution.key,
            execution.identity,
            outcome,
            execution.options.cache_ttl_seconds,
        )
    try:
        async with asyncio.timeout(remaining_ms / 1000):
            return await _write_cache(
                execution.cache,
                execution.key,
                execution.identity,
                outcome,
                execution.options.cache_ttl_seconds,
            )
    except TimeoutError:
        _record_cache_event("write_skipped")
        return False


def _search_identity(
    providers: Mapping[str, SearchProvider],
    query: str,
    options: SearchOptions,
) -> SearchCacheIdentity:
    """Build the exact identity shared by storage and in-process flights."""
    grounding_fingerprint = (
        grounding_semantic_fingerprint(
            options.grounding.config, options.grounding.waterfall.chain
        )
        if options.want_grounding and options.grounding is not None
        else None
    )
    return SearchCacheIdentity(
        query=query,
        skip_quality_filter=options.skip_quality_filter,
        grounding=options.want_grounding,
        providers=tuple(providers),
        grounding_fingerprint=grounding_fingerprint,
    )


def _grounding_report(
    stats: GroundingStats, options: SearchOptions
) -> GroundingReport:
    """Turn one stage's stats into the caller-visible grounding state."""
    return GroundingReport(
        requested=options.want_grounding,
        attempted=stats.total_urls,
        grounded=stats.grounded_count,
        outcomes=dict(stats.outcomes),
    )


def _emit_outcome_metric(
    outcome: SearchOutcome, options: SearchOptions, *, cache_hit: bool
) -> None:
    """Emit the stable per-request search dimensions."""
    grounding = outcome.grounding
    emit_search_metric(
        mode="grounded" if options.want_grounding else "raw",
        total_duration_ms=outcome.total_duration_ms,
        cache_hit=cache_hit,
        providers_succeeded=len(outcome.providers_succeeded),
        providers_failed=len(outcome.providers_failed),
        grounded_count=0 if grounding is None else grounding.grounded,
        grounding_attempted=0 if grounding is None else grounding.attempted,
    )


def _remaining_timeout_ms(
    options: SearchOptions,
    started_at: float,
    knobs: _FanoutKnobs,
) -> int | None:
    """Return this request's remaining global budget in milliseconds."""
    if options.timeout_ms is None:
        return None
    elapsed_ms = _elapsed_ms(started_at, knobs.clock())
    return max(0, options.timeout_ms - elapsed_ms)


async def _wait_for_flight(
    completion: asyncio.Future[None],
    options: SearchOptions,
    started_at: float,
    knobs: _FanoutKnobs,
) -> None:
    """Await a leader without exceeding this waiter's original budget."""
    remaining_ms = _remaining_timeout_ms(options, started_at, knobs)
    if remaining_ms is None:
        await asyncio.shield(completion)
        return
    if remaining_ms <= 0:
        raise _deadline_exceeded_error()
    try:
        async with asyncio.timeout(remaining_ms / 1000):
            await asyncio.shield(completion)
    except TimeoutError as error:
        raise _deadline_exceeded_error() from error


async def _ground_with_remaining_budget(
    query: str,
    ranked: list[RankedWebResult],
    options: SearchOptions,
    start: float,
    knobs: _FanoutKnobs,
) -> tuple[list[RankedWebResult], GroundingStats]:
    """Ground ranked rows within the remaining caller budget.

    Grounding is given a fraction of what is left rather than all of it. The
    ranked rows are already paid for and are a usable answer on their own, so
    the reserve exists to keep a grounding overrun degrading into ungrounded
    results instead of failing the whole search: spending the last millisecond
    here would drive the elapsed budget to exactly zero, which this function
    cannot distinguish from a caller who was already out of time.

    The budget is handed to ``ground_results`` as a deadline rather than
    wrapped around it. Wrapping cancelled the whole stage on expiry and
    returned the ungrounded rows, so a single slow URL discarded every snippet
    its siblings had already paid an LLM to write. Passing the deadline down
    lets each URL end on its own and keeps whatever finished in time.
    """
    context = options.grounding
    if not options.want_grounding or context is None or not ranked:
        return ranked, GroundingStats(0, 0, 0)
    remaining_ms = _remaining_timeout_ms(options, start, knobs)
    if remaining_ms == 0:
        raise _deadline_exceeded_error()
    try:
        if remaining_ms is None:
            pairs, stats = await ground_results(query, ranked, context)
        else:
            pairs, stats = await _ground_under_backstop(
                query, ranked, context, remaining_ms
            )
    except TimeoutError as error:
        if _remaining_timeout_ms(options, start, knobs) == 0:
            raise _deadline_exceeded_error() from error
        _LOGGER.error(
            "Grounding stage overran its own deadline; degrading to "
            "ungrounded results"
        )
        return ranked, GroundingStats(1, 0, 0)
    outcome_map = {result.url: result for result, _ in pairs}
    grounded = [outcome_map.get(result.url, result) for result in ranked]
    return grounded, stats


async def _ground_under_backstop(
    query: str,
    ranked: list[RankedWebResult],
    context: GroundingContext,
    remaining_ms: int,
) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
    """Run the grounding stage on its own deadline, under a hard backstop.

    The stage deadline is the one that should ever fire. It is handed down so
    that each URL ends itself and the URLs that finished keep their snippets.
    The backstop sits a little later and exists only for a stage that fails to
    honour the deadline it was given; reaching it means every snippet is lost,
    so the gap between the two is the stage's room to harvest and drain.
    """
    backstop_seconds = (remaining_ms * _GROUNDING_BUDGET_SHARE) / 1000
    stage_seconds = backstop_seconds * _GROUNDING_HARVEST_SHARE
    deadline_at = asyncio.get_running_loop().time() + stage_seconds
    _LOGGER.info(
        "Grounding stage starting urls=%d budget_s=%.1f backstop_s=%.1f",
        min(len(ranked), context.config.top_n),
        stage_seconds,
        backstop_seconds,
    )
    async with asyncio.timeout(backstop_seconds):
        return await ground_results(query, ranked, context, deadline_at)


def _dispatch_timeout_ms(execution: _SearchExecution) -> int | None:
    """Return the fan-out's own deadline inside the caller's budget.

    The fan-out used to receive the entire remaining budget, which let the
    slowest provider decide how much time the stages behind it inherited. With
    LLM-backed search adapters in the registry that routinely reached twenty
    seconds, grounding was left a remainder too small to fetch and ground a
    single page in, so it spent an LLM call per URL and then ran out. Bounding
    the fan-out separately makes the split a configured decision instead of a
    race outcome.
    """
    remaining_ms = _remaining_timeout_ms(
        execution.options,
        execution.started_at,
        execution.knobs,
    )
    cap_ms = execution.options.fanout_timeout_ms
    if cap_ms is None:
        return remaining_ms
    if remaining_ms is None:
        return cap_ms
    return min(remaining_ms, cap_ms)


async def _execute_search_miss(execution: _SearchExecution) -> SearchOutcome:
    """Dispatch, rank, optionally ground, and cache one leader miss."""
    if (
        _remaining_timeout_ms(
            execution.options, execution.started_at, execution.knobs
        )
        == 0
    ):
        raise _deadline_exceeded_error()
    dispatch: DispatchResult = await dispatch_to_providers(
        execution.providers,
        execution.query,
        timeout_ms=_dispatch_timeout_ms(execution),
        knobs=execution.knobs,
    )
    if not dispatch.providers_succeeded:
        if any(
            failure.deadline_exceeded for failure in dispatch.providers_failed
        ):
            raise _deadline_exceeded_error()
        raise SearchError(_ALL_FAILED_MESSAGE, kind="all_failed")
    ranked = rank_and_merge(
        dispatch.results_by_provider,
        execution.query,
        execution.options.skip_quality_filter,
    )
    ranked, stats = await _ground_with_remaining_budget(
        execution.query,
        ranked,
        execution.options,
        execution.started_at,
        execution.knobs,
    )
    outcome = SearchOutcome(
        query=execution.query,
        total_duration_ms=_elapsed_ms(
            execution.started_at, execution.knobs.clock()
        ),
        providers_succeeded=list(dispatch.providers_succeeded),
        providers_failed=list(dispatch.providers_failed),
        web_results=ranked,
        grounding=_grounding_report(stats, execution.options),
    )
    if should_cache(
        providers_succeeded=len(dispatch.providers_succeeded),
        providers_failed=len(dispatch.providers_failed),
        want_grounding=execution.options.want_grounding,
        transient_failures=stats.transient_failures,
    ):
        await _write_cache_with_remaining_budget(execution, outcome)
    _emit_outcome_metric(outcome, execution.options, cache_hit=False)
    return outcome


async def run_search(
    providers: Mapping[str, SearchProvider],
    cache: CacheBackend,
    query: str,
    *,
    options: SearchOptions = _DEFAULT_SEARCH_OPTIONS,
    knobs: _FanoutKnobs | None = None,
) -> SearchOutcome:
    """Return a cache hit or lead/wait for one complete search miss."""
    if not providers:
        raise SearchError(_NO_PROVIDERS_MESSAGE, kind="no_providers")
    resolved_knobs = knobs if knobs is not None else _FanoutKnobs()
    started_at = resolved_knobs.clock()
    identity = _search_identity(providers, query, options)
    key = make_cache_key(identity)
    execution = _SearchExecution(
        providers,
        cache,
        query,
        identity,
        key,
        options,
        resolved_knobs,
        started_at,
    )
    flights = options.flights
    while True:
        cached = await _read_cache_with_remaining_budget(execution)
        if cached is not None:
            _emit_outcome_metric(cached, options, cache_hit=True)
            return cached
        if flights is None:
            return await _execute_search_miss(execution)
        is_leader, completion = flights.claim(key)
        if not is_leader:
            _record_cache_event("coalesced")
            await _wait_for_flight(
                completion, options, started_at, resolved_knobs
            )
            continue
        try:
            cached = await _read_cache_with_remaining_budget(execution)
            if cached is not None:
                _emit_outcome_metric(cached, options, cache_hit=True)
                return cached
            return await _execute_search_miss(execution)
        finally:
            flights.release(key, completion)

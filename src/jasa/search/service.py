"""Search execution: strict cache read -> fan-out -> rank -> cache write.

MCP and REST share this path. Search cache v2 scopes entries to the exact query,
ordered provider registry, raw/grounded mode, and grounding semantics. Values
use an extra-forbidden versioned envelope and are reconstructed only after all
nested fields and the stored identity validate. ``include_snippets`` and
``timeout_ms`` stay outside the key because the cache stores the full ranked set
before output truncation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
)
from jasa.logging import get_logger
from jasa.observability.metrics import emit_search_metric
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
_SEARCH_CACHE_SCHEMA_VERSION: Literal[2] = 2
_STRICT_RECORD_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """The full ranked result of one search, before output truncation."""

    query: str
    total_duration_ms: int
    providers_succeeded: list[ProviderSuccess]
    providers_failed: list[ProviderFailure]
    web_results: list[RankedWebResult]


class SearchError(Exception):
    """A search could not complete: no providers configured or all failed."""

    def __init__(self, message: str, *, kind: str) -> None:
        """Record the failure ``kind`` (no_providers / all_failed)."""
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """Search execution + output options bundled to keep call sites lean."""

    skip_quality_filter: bool = False
    want_grounding: bool = False
    timeout_ms: int | None = None
    include_snippets: bool = True
    grounding: GroundingContext | None = None
    cache_ttl_seconds: int = TTL_SECONDS


_DEFAULT_SEARCH_OPTIONS = SearchOptions()


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


class _SearchOutcomeRecord(BaseModel):
    """Strict complete search outcome payload."""

    model_config = _STRICT_RECORD_CONFIG

    query: str
    total_duration_ms: NonNegativeInt
    providers_succeeded: list[_ProviderSuccessRecord]
    providers_failed: list[_ProviderFailureRecord]
    web_results: list[_RankedWebResultRecord]


class _SearchCacheRecord(BaseModel):
    """Versioned envelope binding an identity to one complete outcome."""

    model_config = _STRICT_RECORD_CONFIG

    schema_version: Literal[2]
    identity: _SearchIdentityRecord
    outcome: _SearchOutcomeRecord


def _elapsed_ms(start: float, now: float) -> int:
    return int((now - start) * 1000)


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
    )


async def _read_cache(
    cache: CacheBackend, key: str, identity: SearchCacheIdentity
) -> SearchOutcome | None:
    try:
        raw = await cache.get(key)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        record = _SearchCacheRecord.model_validate_json(raw)
    except ValidationError:
        return None
    return _deserialize_outcome(record, identity)


async def _write_cache(
    cache: CacheBackend,
    key: str,
    identity: SearchCacheIdentity,
    outcome: SearchOutcome,
    ttl_seconds: int,
) -> None:
    try:
        await cache.set(key, _serialize(outcome, identity), ttl_seconds)
    except Exception as error:
        _LOGGER.debug("Cache write failed: %s", error)


async def run_search(
    providers: Mapping[str, SearchProvider],
    cache: CacheBackend,
    query: str,
    *,
    options: SearchOptions = _DEFAULT_SEARCH_OPTIONS,
    knobs: _FanoutKnobs | None = None,
) -> SearchOutcome:
    """Run one search: cache, fan-out, rank, and (gated) cache write."""
    if not providers:
        raise SearchError(_NO_PROVIDERS_MESSAGE, kind="no_providers")
    resolved_knobs = knobs if knobs is not None else _FanoutKnobs()
    grounding_fingerprint = (
        grounding_semantic_fingerprint(options.grounding.config)
        if options.want_grounding and options.grounding is not None
        else None
    )
    identity = SearchCacheIdentity(
        query=query,
        skip_quality_filter=options.skip_quality_filter,
        grounding=options.want_grounding,
        providers=tuple(providers),
        grounding_fingerprint=grounding_fingerprint,
    )
    key = make_cache_key(identity)
    cached = await _read_cache(cache, key, identity)
    if cached is not None:
        _LOGGER.debug("Cache hit for query.")
        emit_search_metric(
            mode="grounded" if options.want_grounding else "raw",
            total_duration_ms=cached.total_duration_ms,
            cache_hit=True,
            providers_succeeded=len(cached.providers_succeeded),
            providers_failed=len(cached.providers_failed),
        )
        return cached
    start = resolved_knobs.clock()
    dispatch: DispatchResult = await dispatch_to_providers(
        providers, query, timeout_ms=options.timeout_ms, knobs=resolved_knobs
    )
    if not dispatch.providers_succeeded:
        raise SearchError(_ALL_FAILED_MESSAGE, kind="all_failed")
    ranked = rank_and_merge(
        dispatch.results_by_provider, query, options.skip_quality_filter
    )
    transient_failures = 0
    if options.want_grounding and options.grounding is not None and ranked:
        import asyncio

        grounding_ctx = options.grounding
        grounding_ran = False
        if options.timeout_ms is not None:
            elapsed = _elapsed_ms(start, resolved_knobs.clock())
            remaining = max(0, options.timeout_ms - elapsed) / 1000
            if remaining > 0:
                try:
                    async with asyncio.timeout(remaining):
                        pairs, gstats = await ground_results(
                            query, ranked, grounding_ctx
                        )
                    grounding_ran = True
                except TimeoutError:
                    pass
        else:
            pairs, gstats = await ground_results(query, ranked, grounding_ctx)
            grounding_ran = True
        if grounding_ran:
            outcome_map = {r.url: r for r, _ in pairs}
            ranked = [outcome_map.get(r.url, r) for r in ranked]
            transient_failures = gstats.transient_failures
        else:
            transient_failures = 1
    outcome = SearchOutcome(
        query=query,
        total_duration_ms=_elapsed_ms(start, resolved_knobs.clock()),
        providers_succeeded=list(dispatch.providers_succeeded),
        providers_failed=list(dispatch.providers_failed),
        web_results=ranked,
    )
    if should_cache(
        providers_succeeded=len(dispatch.providers_succeeded),
        providers_failed=len(dispatch.providers_failed),
        want_grounding=options.want_grounding,
        transient_failures=transient_failures,
    ):
        await _write_cache(
            cache, key, identity, outcome, options.cache_ttl_seconds
        )
    emit_search_metric(
        mode="grounded" if options.want_grounding else "raw",
        total_duration_ms=outcome.total_duration_ms,
        cache_hit=False,
        providers_succeeded=len(outcome.providers_succeeded),
        providers_failed=len(outcome.providers_failed),
    )
    return outcome

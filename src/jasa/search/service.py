"""Search execution coroutine: cache read -> fan-out -> rank -> cache write.

The single path shared by the MCP tool and the REST routes (§6: one execution
path per capability). Grounding is a no-op until Phase 6 (``want_grounding``
stays false here); the cache gate already accounts for transient grounding
failures, so wiring grounding later is a localized insertion.
``include_snippets`` and ``timeout_ms`` never enter the cache key; the cache
stores the full ranked set before output truncation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from jasa.cache.base import (
    CacheBackend,
    make_cache_key,
    should_cache,
    TTL_SECONDS,
)
from jasa.grounding.service import ground_results, GroundingContext
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


_DEFAULT_SEARCH_OPTIONS = SearchOptions()


def _elapsed_ms(start: float, now: float) -> int:
    return int((now - start) * 1000)


def _serialize(outcome: SearchOutcome) -> str:
    return json.dumps(asdict(outcome))


def _deserialize_outcome(record: object, query: str) -> SearchOutcome | None:
    if not isinstance(record, dict) or record.get("query") != query:
        return None
    try:
        succeeded = [
            ProviderSuccess(**p) for p in record.get("providers_succeeded", [])
        ]
        failed = [
            ProviderFailure(**p) for p in record.get("providers_failed", [])
        ]
        web_results = [
            RankedWebResult(**r) for r in record.get("web_results", [])
        ]
        return SearchOutcome(
            query=str(record["query"]),
            total_duration_ms=int(record["total_duration_ms"]),
            providers_succeeded=succeeded,
            providers_failed=failed,
            web_results=web_results,
        )
    except (KeyError, TypeError, ValueError):
        return None


async def _read_cache(
    cache: CacheBackend, key: str, query: str
) -> SearchOutcome | None:
    try:
        raw = await cache.get(key)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return _deserialize_outcome(record, query)


async def _write_cache(
    cache: CacheBackend, key: str, outcome: SearchOutcome
) -> None:
    try:
        await cache.set(key, _serialize(outcome), TTL_SECONDS)
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
    key = make_cache_key(
        query,
        skip_quality_filter=options.skip_quality_filter,
        grounding=options.want_grounding,
    )
    cached = await _read_cache(cache, key, query)
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
        await _write_cache(cache, key, outcome)
    emit_search_metric(
        mode="grounded" if options.want_grounding else "raw",
        total_duration_ms=outcome.total_duration_ms,
        cache_hit=False,
        providers_succeeded=len(outcome.providers_succeeded),
        providers_failed=len(outcome.providers_failed),
    )
    return outcome

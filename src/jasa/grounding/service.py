"""Grounding service: success cache + bounded workers + 9 outcomes.

Each URL in the top-N is fetched once (via the in-process omnifetch engine),
junk-detected, and keyed into a process-local flight by its exact effective LLM
input. One leader reads cache and calls the snippet-writing LLM on a miss while
waiters release worker slots and later retry. Only accepted LLM output enters
the shared cache. Transient
outcomes (llm_error, pipeline_timeout, worker_rejected) block the complete
search cache write; durable fallbacks (junk, sentinel, too-short) do not.
Output order follows input order.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Literal

import httpx

from jasa.cache.base import CacheBackend
from jasa.config import (
    DEFAULT_GROUNDING_CACHE_TTL_SECONDS,
    GroundingSettings,
)
from jasa.grounding.cache import (
    FREQUENCY_PENALTY,
    grounding_cache_identity,
    GroundingCacheIdentity,
    GroundingCacheWrite,
    make_grounding_cache_key,
    MIN_SNIPPET_CHARS,
    read_grounding_cache,
    record_grounding_cache_event,
    TEMPERATURE,
    TOP_P,
    write_grounding_cache,
)
from jasa.grounding.detectors import (
    detect_grounded_junk,
    detect_grounded_sentinel,
    grounding_detector_semantics,
    repair_unbalanced_fence,
)
from jasa.grounding.flights import (
    GroundingFlightOwnership,
    GroundingFlightRegistry,
    GroundingWait,
    wait_for_grounding_flight,
)
from jasa.grounding.prompts import (
    build_grounded_user_message,
    GROUNDING_MAX_TOKENS,
    grounding_prompt_semantics,
    SNIPPET_MAX_CHARS,
    SYSTEM_PROMPT,
)
from jasa.logging import get_logger
from jasa.search.ranking import RankedWebResult
from omnifetch.tools.fetch import execute_web_fetch

_LOGGER = get_logger("grounding")

MIN_CONTENT_CHARS = 50
GROUNDING_SEMANTICS_VERSION = 1
GROUNDING_CACHE_READ_TIMEOUT_SECONDS = 0.25

GroundingOutcome = Literal[
    "grounded",
    "fallback:fetch_exhausted",
    "fallback:fetch_too_short",
    "fallback:fetch_junk",
    "fallback:llm_sentinel",
    "fallback:llm_error",
    "fallback:llm_empty",
    "fallback:pipeline_timeout",
    "fallback:worker_rejected",
]

TRANSIENT_OUTCOMES = frozenset(
    {
        "fallback:llm_error",
        "fallback:pipeline_timeout",
        "fallback:worker_rejected",
    }
)


@dataclass(frozen=True, slots=True)
class GroundingContext:
    """Resources injected from the composition for the grounding stage."""

    engine: object
    client: httpx.AsyncClient
    cache: CacheBackend
    cache_write_semaphore: asyncio.Semaphore
    flights: GroundingFlightRegistry
    api_key: str
    config: GroundingSettings
    cache_ttl_seconds: int = DEFAULT_GROUNDING_CACHE_TTL_SECONDS


@dataclass(frozen=True, slots=True)
class GroundingStats:
    """Aggregate statistics for one grounded search."""

    transient_failures: int
    grounded_count: int
    total_urls: int


@dataclass(frozen=True, slots=True)
class _GroundingAttempt:
    """One classified worker result and its optional successful write."""

    result: RankedWebResult
    outcome: GroundingOutcome
    cache_write: GroundingCacheWrite | None = None


@dataclass(frozen=True, slots=True)
class _GroundingInput:
    """Fetched, validated, exact inputs for cache lookup and one LLM call."""

    result: RankedWebResult
    fetched_title: str
    identity: GroundingCacheIdentity
    key: str


@dataclass(frozen=True, slots=True)
class _GroundingLeader:
    """An accepted leader result whose flight spans its cache write."""

    attempt: _GroundingAttempt
    pending: GroundingCacheWrite


@dataclass(frozen=True, slots=True)
class _GroundingExecution:
    """Immutable resources shared across one URL's worker phases."""

    result: RankedWebResult
    query: str
    context: GroundingContext
    semaphore: asyncio.Semaphore


def grounding_semantic_fingerprint(config: GroundingSettings) -> str:
    """Hash every configured or versioned input to grounded-search output."""
    identity = {
        "detectors": grounding_detector_semantics(),
        "frequency_penalty": FREQUENCY_PENALTY,
        "llm_base_url": config.llm_base_url,
        "llm_model": config.llm_model,
        "max_content_chars": config.max_content_chars,
        "max_tokens": GROUNDING_MAX_TOKENS,
        "min_content_chars": MIN_CONTENT_CHARS,
        "min_snippet_chars": MIN_SNIPPET_CHARS,
        "prompts": grounding_prompt_semantics(),
        "semantics_version": GROUNDING_SEMANTICS_VERSION,
        "snippet_max_chars": SNIPPET_MAX_CHARS,
        "temperature": TEMPERATURE,
        "top_n": config.top_n,
        "top_p": TOP_P,
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _llm_call(
    client: httpx.AsyncClient,
    api_key: str,
    config: GroundingSettings,
    user_message: str,
) -> str:
    """Call the grounding LLM and return the raw snippet text."""
    response = await client.post(
        f"{config.llm_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": config.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "frequency_penalty": FREQUENCY_PENALTY,
            "max_tokens": GROUNDING_MAX_TOKENS,
        },
        timeout=min(config.llm_timeout_ms, config.per_url_deadline_ms) / 1000,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if content is None:
        return ""
    return str(content)


async def _fetch_and_prepare(
    result: RankedWebResult,
    query: str,
    context: GroundingContext,
) -> _GroundingAttempt | _GroundingInput:
    """Fetch one URL and return a durable fallback or exact LLM input."""
    try:
        fetch_result = await execute_web_fetch(context.engine, result.url)
    except Exception:
        return _GroundingAttempt(result, "fallback:fetch_exhausted")
    content = getattr(fetch_result, "content", "") or ""
    title = getattr(fetch_result, "title", "") or ""
    if len(content) < MIN_CONTENT_CHARS:
        return _GroundingAttempt(result, "fallback:fetch_too_short")
    junk = detect_grounded_junk(content)
    if junk:
        return _GroundingAttempt(result, "fallback:fetch_junk")
    user_message = build_grounded_user_message(
        query, title, content, context.config.max_content_chars
    )
    identity = grounding_cache_identity(user_message, context.config)
    return _GroundingInput(
        result,
        title,
        identity,
        make_grounding_cache_key(identity),
    )


async def _read_cached_grounding(
    prepared: _GroundingInput,
    context: GroundingContext,
    read_deadline_at: float,
) -> _GroundingAttempt | None:
    """Return a bounded strict cache hit or None without spending the budget."""
    try:
        async with asyncio.timeout_at(read_deadline_at):
            cached = await read_grounding_cache(
                context.cache,
                prepared.key,
                prepared.identity,
                prepared.fetched_title,
            )
    except TimeoutError:
        record_grounding_cache_event("read_skipped")
        cached = None
    if cached is not None:
        return _accepted_grounding(
            prepared.result,
            prepared.fetched_title,
            cached,
        )
    return None


async def _call_live_grounding(
    prepared: _GroundingInput,
    context: GroundingContext,
) -> _GroundingAttempt:
    """Call and classify one live LLM request, preserving stable fallbacks."""
    try:
        snippet = await _llm_call(
            context.client,
            context.api_key,
            context.config,
            prepared.identity.user_message,
        )
    except Exception:
        return _GroundingAttempt(prepared.result, "fallback:llm_error")
    return _classify_live_grounding(
        prepared.result,
        prepared.fetched_title,
        snippet,
        prepared.identity,
        prepared.key,
    )


async def _ground_user_message(
    prepared: _GroundingInput,
    context: GroundingContext,
    deadline_at: float,
    ownership: GroundingFlightOwnership,
) -> _GroundingAttempt | GroundingWait | _GroundingLeader:
    """Return a cache hit, join a flight, or lead one live LLM request."""
    loop = asyncio.get_running_loop()
    now = loop.time()
    remaining_seconds = max(0.0, deadline_at - now)
    read_deadline_at = now + min(
        GROUNDING_CACHE_READ_TIMEOUT_SECONDS,
        remaining_seconds / 2,
    )
    cached = await _read_cached_grounding(prepared, context, read_deadline_at)
    if cached is not None:
        return cached
    is_leader, completion = context.flights.claim(prepared.key)
    if not is_leader:
        record_grounding_cache_event("coalesced")
        return GroundingWait(completion)
    ownership.hold(prepared.key, completion)
    cached = await _read_cached_grounding(prepared, context, read_deadline_at)
    attempt = (
        cached
        if cached is not None
        else await _call_live_grounding(prepared, context)
    )
    if attempt.cache_write is None:
        return attempt
    return _GroundingLeader(attempt, attempt.cache_write)


def _classify_live_grounding(
    result: RankedWebResult,
    fetched_title: str,
    snippet: str,
    identity: GroundingCacheIdentity,
    key: str,
) -> _GroundingAttempt:
    """Validate live LLM output and prepare only accepted output to write."""
    if len(snippet) < MIN_SNIPPET_CHARS:
        return _GroundingAttempt(result, "fallback:llm_empty")
    snippet = snippet[:SNIPPET_MAX_CHARS]
    snippet = repair_unbalanced_fence(snippet)
    sentinel = detect_grounded_sentinel(snippet)
    if sentinel:
        return _GroundingAttempt(result, "fallback:llm_sentinel")
    pending = GroundingCacheWrite(
        key,
        identity,
        snippet,
        fetched_title,
    )
    return _accepted_grounding(result, fetched_title, snippet, pending)


def _accepted_grounding(
    result: RankedWebResult,
    fetched_title: str,
    snippet: str,
    pending: GroundingCacheWrite | None = None,
) -> _GroundingAttempt:
    """Rebuild the exact public grounded result for a hit or live success."""
    updated = replace(
        result,
        snippets=[snippet],
        snippet_source="grounded",
        title=fetched_title or result.title,
    )
    return _GroundingAttempt(updated, "grounded", pending)


async def _run_grounding_worker_phase(
    execution: _GroundingExecution,
    deadline_at: float,
    prepared: _GroundingInput | None,
    ownership: GroundingFlightOwnership,
) -> tuple[
    _GroundingInput | None,
    _GroundingAttempt | GroundingWait | _GroundingLeader,
]:
    """Fetch once, then resolve cache/flight/LLM while holding a worker."""
    current = prepared
    if current is None:
        fetched = await _fetch_and_prepare(
            execution.result,
            execution.query,
            execution.context,
        )
        if isinstance(fetched, _GroundingAttempt):
            return None, fetched
        current = fetched
    resolution = await _ground_user_message(
        current,
        execution.context,
        deadline_at,
        ownership,
    )
    return current, resolution


async def _run_grounding_worker(
    execution: _GroundingExecution,
    deadline_at: float | None,
    prepared: _GroundingInput | None,
    ownership: GroundingFlightOwnership,
) -> tuple[
    float,
    _GroundingInput | None,
    _GroundingAttempt | GroundingWait | _GroundingLeader,
]:
    """Run one worker phase, bounding every reacquisition by its deadline."""
    if deadline_at is None:
        async with execution.semaphore:
            resolved_deadline = asyncio.get_running_loop().time() + (
                execution.context.config.per_url_deadline_ms / 1000
            )
            async with asyncio.timeout_at(resolved_deadline):
                current, resolution = await _run_grounding_worker_phase(
                    execution,
                    resolved_deadline,
                    prepared,
                    ownership,
                )
        return resolved_deadline, current, resolution
    async with asyncio.timeout_at(deadline_at):
        async with execution.semaphore:
            current, resolution = await _run_grounding_worker_phase(
                execution,
                deadline_at,
                prepared,
                ownership,
            )
    return deadline_at, current, resolution


async def _write_grounding_leader(
    leader: _GroundingLeader,
    context: GroundingContext,
    deadline_at: float,
) -> _GroundingAttempt:
    """Write one accepted leader result within its original deadline."""
    try:
        async with asyncio.timeout_at(deadline_at):
            async with context.cache_write_semaphore:
                await write_grounding_cache(
                    context.cache,
                    leader.pending,
                    context.cache_ttl_seconds,
                )
    except TimeoutError:
        record_grounding_cache_event("write_skipped")
    return leader.attempt


async def _ground_one(
    execution: _GroundingExecution,
) -> tuple[RankedWebResult, GroundingOutcome]:
    """Resolve one URL within worker, flight, and deadline bounds."""
    deadline_at: float | None = None
    prepared: _GroundingInput | None = None
    while True:
        ownership = GroundingFlightOwnership(execution.context.flights)
        try:
            try:
                deadline_at, prepared, resolution = await _run_grounding_worker(
                    execution, deadline_at, prepared, ownership
                )
            except TimeoutError:
                return execution.result, "fallback:pipeline_timeout"
            except Exception:
                return execution.result, "fallback:worker_rejected"
            if isinstance(resolution, GroundingWait):
                if not await wait_for_grounding_flight(
                    resolution.completion, deadline_at
                ):
                    return execution.result, "fallback:pipeline_timeout"
                continue
            if isinstance(resolution, _GroundingAttempt):
                return resolution.result, resolution.outcome
            attempt = await _write_grounding_leader(
                resolution, execution.context, deadline_at
            )
            return attempt.result, attempt.outcome
        finally:
            ownership.release()


async def ground_results(
    query: str,
    results: list[RankedWebResult],
    context: GroundingContext,
) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
    """Ground results in a bounded pool; preserve input order."""
    semaphore = asyncio.Semaphore(context.config.concurrency)

    pairs = await asyncio.gather(
        *(
            _ground_one(_GroundingExecution(result, query, context, semaphore))
            for result in results[: context.config.top_n]
        )
    )
    outcomes = [outcome for _, outcome in pairs]
    transient = sum(1 for o in outcomes if o in TRANSIENT_OUTCOMES)
    grounded = sum(1 for o in outcomes if o == "grounded")
    return list(pairs), GroundingStats(
        transient_failures=transient,
        grounded_count=grounded,
        total_urls=len(pairs),
    )

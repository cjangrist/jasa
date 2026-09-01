"""Grounding service: success cache + bounded workers + 9 outcomes.

Each URL in the top-N is fetched once (via the in-process omnifetch engine),
junk-detected, and keyed into a process-local flight by its canonical URL and
query -- not by the fetched bytes, so two providers' renderings of one page
share a flight. One leader reads cache and calls the snippet-writing LLM on a
miss while waiters release worker slots and later retry. Only accepted LLM
output enters the shared cache. Transient
outcomes (llm_error, pipeline_timeout, worker_rejected) block the complete
search cache write; durable fallbacks (junk, sentinel, too-short) do not.
Output order follows input order.

The LLM call is a waterfall, not one endpoint. The fetch is already billed by
the time any tier is reachable, so a tier that rate-limits, errors, or returns
no text advances to the next one rather than discarding that page. A sentinel
does not advance: it is the model's judgment about the fetched page, so asking
another model to disagree would defeat the sentinel contract. The whole chain
shares the one per-URL deadline, and each attempt is capped by its own budget,
by what remains, and by what the tiers behind it still need: a tier allowed to
spend the entire remaining budget leaves the fallbacks unreachable, which is
the failure they exist to cover.

Every URL is resolved independently and harvested independently. A shared
budget that expires cancels only the workers still running; the ones that
already produced a snippet keep it. Grounding costs a page fetch plus at least
one LLM completion per URL, both billed before any deadline can fire, so
discarding finished work to honour a deadline spends money and returns
nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal
from urllib.parse import urlsplit

import httpx

from jasa.cache.base import CacheBackend
from jasa.config import (
    DEFAULT_FETCH_CACHE_TTL_SECONDS,
    DEFAULT_GROUNDING_CACHE_TTL_SECONDS,
    DEFAULT_VOLATILE_FETCH_CACHE_TTL_SECONDS,
    GroundingSettings,
)
from jasa.grounding.cache import (
    FREQUENCY_PENALTY,
    grounding_cache_identity,
    grounding_cache_ttl_seconds,
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
    complete_coverage_line,
    detect_grounded_junk,
    detect_grounded_sentinel,
    FENCE_REPAIR_SUFFIX,
    grounding_detector_semantics,
    has_complete_coverage_line,
    repair_unbalanced_fence,
    trim_truncated_snippet,
)
from jasa.grounding.flights import (
    GroundingFlightOwnership,
    GroundingFlightRegistry,
    GroundingWait,
    wait_for_grounding_flight,
)
from jasa.grounding.prompts import (
    build_grounded_user_message,
    COVERAGE_SEPARATOR_MAX_CHARS,
    GROUNDING_MAX_TOKENS,
    grounding_prompt_semantics,
    SNIPPET_MAX_CHARS,
    SYSTEM_PROMPT,
)
from jasa.grounding.waterfall import (
    grounding_chain_semantics,
    GroundingChain,
    GroundingTier,
    ResolvedGroundingWaterfall,
)
from jasa.logging import get_logger
from jasa.search.ranking import RankedWebResult
from omnifetch.tools.fetch import cache_identity_url, execute_web_fetch

_LOGGER = get_logger("grounding")

MIN_CONTENT_CHARS = 50
GROUNDING_SEMANTICS_VERSION = 3
GROUNDING_CACHE_READ_TIMEOUT_SECONDS = 0.25
MIN_TIER_BUDGET_SECONDS = 8.0
MIN_WORKER_BUDGET_SECONDS = 2.0
_DRAIN_GRACE_SECONDS = 5.0

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
GroundingProgressReporter = Callable[[int, int], Awaitable[None]]

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
    waterfall: ResolvedGroundingWaterfall
    config: GroundingSettings
    cache_ttl_seconds: int = DEFAULT_GROUNDING_CACHE_TTL_SECONDS
    fetch_cache_ttl_seconds: int = DEFAULT_FETCH_CACHE_TTL_SECONDS
    volatile_cache_ttl_seconds: int = DEFAULT_VOLATILE_FETCH_CACHE_TTL_SECONDS


@dataclass(frozen=True, slots=True)
class GroundingStats:
    """Aggregate statistics for one grounded search.

    ``outcomes`` counts every classified URL by its exact outcome so that a
    caller can tell a search where grounding never ran from one where it ran
    and every page was a paywall, a timeout, or a sentinel. Request success on
    its own carries neither fact.
    """

    transient_failures: int
    grounded_count: int
    total_urls: int
    outcomes: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _GroundingAttempt:
    """One classified worker result and its optional successful write."""

    result: RankedWebResult
    outcome: GroundingOutcome
    cache_write: GroundingCacheWrite | None = None


@dataclass(frozen=True, slots=True)
class _GroundingInput:
    """Fetched, validated, exact inputs for cache lookup and one LLM call.

    ``user_message`` and ``identity`` are deliberately separate. The message is
    what the model is shown and carries the page's bytes; the identity is what
    the result is filed under and carries only the page's address. Folding the
    two together is what made a re-rendered page look like a different request.
    """

    result: RankedWebResult
    fetched_title: str
    user_message: str
    identity: GroundingCacheIdentity
    key: str


@dataclass(frozen=True, slots=True)
class _TierResponse:
    """One tier's assistant text and whether its generation was cut short."""

    text: str
    truncated: bool
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _WaterfallOutcome:
    """The chain's first usable snippet, or how the last tier was spent."""

    snippet: str | None
    failure: GroundingOutcome
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _GroundingLeader:
    """An accepted leader result whose flight spans its cache write."""

    attempt: _GroundingAttempt
    pending: GroundingCacheWrite


@dataclass(frozen=True, slots=True)
class _GroundingExecution:
    """Immutable resources shared across one URL's worker phases.

    ``budget_deadline_at`` is the whole stage's shared absolute deadline. A
    worker clamps its own per-URL deadline to it so that it self-terminates
    with a classified outcome instead of being cancelled mid-flight.
    """

    result: RankedWebResult
    query: str
    context: GroundingContext
    semaphore: asyncio.Semaphore
    budget_deadline_at: float | None = None


def grounding_semantic_fingerprint(
    config: GroundingSettings, chain: GroundingChain
) -> str:
    """Hash every configured or versioned input to grounded-search output."""
    identity = {
        "detectors": grounding_detector_semantics(),
        "frequency_penalty": FREQUENCY_PENALTY,
        "llm_chain": grounding_chain_semantics(chain),
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


class GroundingTierError(Exception):
    """One tier answered in a shape the next tier may still recover from."""


def _read_tier_response(payload: object) -> _TierResponse:
    """Return the text and stop reason of one chat-completions response.

    A gateway can report a failure inside a 200 body instead of as a status
    code, and is not obliged to return a well-formed choice, so every level is
    shape-checked before it is read. Anything unreadable advances the chain
    rather than escaping as an unclassified error.

    ``finish_reason`` is read because a generation stopped at the token ceiling
    is a partial answer wearing a success's clothes: the response is a well
    formed 200 whose text simply stops mid-word. It is optional and free-form
    across gateways, so an absent or non-string value is treated as no claim
    rather than as a malformed body.
    """
    if not isinstance(payload, dict):
        raise GroundingTierError("malformed_body")
    if payload.get("error") is not None:
        raise GroundingTierError("body_error")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GroundingTierError("no_choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise GroundingTierError("no_message")
    raw_finish_reason = first.get("finish_reason")
    finish_reason = (
        raw_finish_reason if isinstance(raw_finish_reason, str) else None
    )
    truncated = finish_reason == "length"
    content = message.get("content")
    if content is None:
        return _TierResponse("", truncated, finish_reason)
    if not isinstance(content, str):
        raise GroundingTierError("no_content")
    return _TierResponse(content, truncated, finish_reason)


async def _call_grounding_tier(
    client: httpx.AsyncClient,
    api_key: str,
    tier: GroundingTier,
    user_message: str,
    timeout_seconds: float,
) -> _TierResponse:
    """Call one waterfall tier and return its raw snippet and stop reason.

    The httpx budget bounds each connection phase separately, so the caller
    also wraps the whole attempt to keep one slow tier from consuming the
    budget the tiers behind it still need.
    """
    response = await client.post(
        f"{tier.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": tier.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "frequency_penalty": FREQUENCY_PENALTY,
            "max_tokens": GROUNDING_MAX_TOKENS,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return _read_tier_response(response.json())


def _tier_error_detail(error: BaseException) -> str:
    """Return a bounded, secret-free label for one failed tier attempt.

    ``HTTPStatusError`` alone cannot be acted on: a 429 means the concurrency
    is above the tier's rate limit, a 401 means its credential is wrong, and a
    5xx means the provider is down. Those call for opposite responses, so the
    status code is named. Only the code is taken -- never the body, which can
    echo the request.
    """
    if isinstance(error, httpx.HTTPStatusError):
        return f"http_{error.response.status_code}"
    return type(error).__name__


def _record_tier_advance(
    tier_name: str, error_type: str, spent_seconds: float
) -> None:
    """Log one bounded waterfall advance without any request material.

    ``spent_s`` is measured, not the slice the tier was allotted. A tier that
    fails instantly -- a 401, a 429, an empty body -- would otherwise report
    its whole budget as spent, sending an operator after a latency problem
    that is really a credential or quota one.
    """
    _LOGGER.warning(
        "Grounding tier advanced tier=%s error_type=%s spent_s=%.2f",
        tier_name,
        error_type,
        spent_seconds,
    )


def _tier_attempt_seconds(
    tier: GroundingTier, remaining_seconds: float, tiers_after: int
) -> float:
    """Bound one attempt so the tiers behind it stay reachable.

    Capping an attempt at its own timeout and the remaining budget is not
    enough. The first tier's timeout inherits an environment setting sized for
    a lone endpoint, so a tier that hangs rather than failing fast consumes the
    entire per-URL budget and every fallback behind it is skipped -- exactly
    the outage the waterfall was built to survive. Each tier therefore also
    yields a minimum slice to each tier still queued behind it.

    The reserve is advisory, not a hard floor. When too little budget remains
    to give every tier a usable slice, reserving anything only buys attempts
    too short to answer in, so the current tier takes everything left instead:
    one real attempt beats several doomed ones.
    """
    needed_for_every_tier = MIN_TIER_BUDGET_SECONDS * (tiers_after + 1)
    if remaining_seconds < needed_for_every_tier:
        return min(tier.timeout_ms / 1000, remaining_seconds)
    reserve_seconds = tiers_after * MIN_TIER_BUDGET_SECONDS
    return min(tier.timeout_ms / 1000, remaining_seconds - reserve_seconds)


async def _run_grounding_waterfall(
    prepared: _GroundingInput,
    context: GroundingContext,
    deadline_at: float,
) -> _WaterfallOutcome:
    """Walk the chain until one tier returns usable text or budget ends."""
    loop = asyncio.get_running_loop()
    failure: GroundingOutcome = "fallback:llm_error"
    chain = context.waterfall.chain
    for index, tier in enumerate(chain):
        remaining_seconds = deadline_at - loop.time()
        if remaining_seconds <= 0:
            _LOGGER.warning(
                "Grounding chain exhausted its budget tier=%s remaining=%d",
                tier.name,
                len(chain) - index,
            )
            break
        attempt_seconds = _tier_attempt_seconds(
            tier, remaining_seconds, len(chain) - index - 1
        )
        started_at = loop.time()
        try:
            async with asyncio.timeout(attempt_seconds):
                answer = await _call_grounding_tier(
                    context.client,
                    context.waterfall.api_keys[tier.api_key_env],
                    tier,
                    prepared.user_message,
                    attempt_seconds,
                )
        except Exception as error:
            failure = "fallback:llm_error"
            _record_tier_advance(
                tier.name,
                _tier_error_detail(error),
                loop.time() - started_at,
            )
            continue
        stripped_text = answer.text.strip()
        is_sentinel = detect_grounded_sentinel(stripped_text) is not None
        stopped_without_coverage = (
            answer.finish_reason == "stop"
            and not is_sentinel
            and not has_complete_coverage_line(stripped_text)
        )
        if stopped_without_coverage:
            failure = "fallback:llm_error"
            _record_tier_advance(
                tier.name,
                "missing_coverage",
                loop.time() - started_at,
            )
            continue
        if len(stripped_text) >= MIN_SNIPPET_CHARS:
            if answer.truncated:
                _LOGGER.warning(
                    "Grounding generation hit its token ceiling tier=%s "
                    "chars=%d max_tokens=%d",
                    tier.name,
                    len(answer.text.strip()),
                    GROUNDING_MAX_TOKENS,
                )
            else:
                _LOGGER.debug(
                    "Grounding tier answered tier=%s chars=%d",
                    tier.name,
                    len(answer.text.strip()),
                )
            return _WaterfallOutcome(answer.text, failure, answer.truncated)
        failure = "fallback:llm_empty"
        _record_tier_advance(
            tier.name, "empty_content", loop.time() - started_at
        )
    return _WaterfallOutcome(None, failure)


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
    identity = grounding_cache_identity(
        cache_identity_url(context.engine, result.url),
        query,
        context.config.max_content_chars,
        context.waterfall.chain,
    )
    return _GroundingInput(
        result,
        title,
        user_message,
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
    deadline_at: float,
) -> _GroundingAttempt:
    """Run the waterfall and classify its first usable snippet."""
    outcome = await _run_grounding_waterfall(prepared, context, deadline_at)
    if outcome.snippet is None:
        return _GroundingAttempt(prepared.result, outcome.failure)
    return _classify_live_grounding(prepared, outcome)


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
        else await _call_live_grounding(prepared, context, deadline_at)
    )
    if attempt.cache_write is None:
        return attempt
    return _GroundingLeader(attempt, attempt.cache_write)


def _trimmed_without_changing_the_verdict(
    snippet: str, sentinel: str | None
) -> str:
    """Trim a cut generation, unless trimming would read as a sentinel.

    The stored snippet is re-checked against the sentinel detector when it is
    read back from cache, so a value that trips the detector is written and
    then refused on every read: the accepted result never serves a second
    request and the paid LLM call is repeated forever. Because the verdict is
    taken before trimming, a trim that crosses the detector's length threshold
    would create exactly that value, so it is abandoned instead. An unpolished
    ending costs one ragged sentence; the alternative costs the call.
    """
    trimmed = trim_truncated_snippet(snippet)
    if sentinel is None and detect_grounded_sentinel(trimmed) is not None:
        return snippet
    return trimmed


def _classify_live_grounding(
    prepared: _GroundingInput,
    outcome: _WaterfallOutcome,
) -> _GroundingAttempt:
    """Validate live LLM output and prepare only accepted output to write.

    Emptiness is already resolved by the waterfall, which advances past a tier
    whose text is blank once stripped, so only a snippet carrying real
    characters reaches this point and no fallback can erase a valid aggregated
    snippet with whitespace.

    A snippet is trimmed back to its last complete sentence whenever it was
    cut short, whether the model stopped at its token ceiling or this function
    applied the character cap. Both leave the text ending mid-word; neither
    can recover what was lost, so the aim is only to avoid publishing a
    fragment that stops mid-thought. The character cap is applied first so the
    trim operates on the text that will actually be emitted.

    The sentinel verdict is taken from the model's own text, before trimming.
    Sentinel matching accepts a bracketed phrase as a substring only in a short
    snippet, so shortening the text can push a long answer that merely quotes
    such a phrase -- page content, which an author controls -- under that
    threshold and manufacture a verdict the model never gave. A sentinel is a
    judgment about the page; post-processing must read it, not create it.
    """
    snippet = outcome.snippet or ""
    sentinel = detect_grounded_sentinel(snippet)
    was_cut = outcome.truncated or len(snippet) > SNIPPET_MAX_CHARS
    coverage_line = complete_coverage_line(snippet)
    snippet = _cap_grounded_snippet(snippet, coverage_line)
    if was_cut and coverage_line is None:
        snippet = _trimmed_without_changing_the_verdict(snippet, sentinel)
    if sentinel:
        return _GroundingAttempt(prepared.result, "fallback:llm_sentinel")
    pending = GroundingCacheWrite(
        prepared.key,
        prepared.identity,
        snippet,
    )
    return _accepted_grounding(
        prepared.result, prepared.fetched_title, snippet, pending
    )


def _cap_grounded_snippet(snippet: str, coverage_line: str | None) -> str:
    """Cap and fence-repair output without severing a coverage line."""
    if coverage_line is not None:
        if (
            len(snippet) <= SNIPPET_MAX_CHARS
            and repair_unbalanced_fence(snippet) == snippet
        ):
            return snippet
        body = snippet[: snippet.rfind("Coverage:")].rstrip()
        return _cap_body_with_coverage(body, coverage_line)
    repaired = repair_unbalanced_fence(snippet)
    if len(repaired) <= SNIPPET_MAX_CHARS:
        return repaired
    capped = _cap_snippet_body(snippet, SNIPPET_MAX_CHARS)
    repaired = repair_unbalanced_fence(capped)
    if len(repaired) <= SNIPPET_MAX_CHARS:
        return repaired
    repair_aware_limit = SNIPPET_MAX_CHARS - len(FENCE_REPAIR_SUFFIX)
    return repair_unbalanced_fence(
        _cap_snippet_body(snippet, repair_aware_limit)
    )


def _cap_body_with_coverage(body: str, coverage_line: str) -> str:
    """Repair a bounded body, then attach its intact final coverage line."""
    body_limit = max(
        0,
        SNIPPET_MAX_CHARS - COVERAGE_SEPARATOR_MAX_CHARS - len(coverage_line),
    )
    repaired_body = repair_unbalanced_fence(_cap_snippet_body(body, body_limit))
    separator = "\n\n" if repaired_body else ""
    combined = f"{repaired_body}{separator}{coverage_line}"
    if len(combined) <= SNIPPET_MAX_CHARS:
        return combined
    repair_aware_limit = max(0, body_limit - len(FENCE_REPAIR_SUFFIX))
    repaired_body = repair_unbalanced_fence(
        _cap_snippet_body(body, repair_aware_limit)
    )
    separator = "\n\n" if repaired_body else ""
    return f"{repaired_body}{separator}{coverage_line}"


def _cap_snippet_body(body: str, limit: int) -> str:
    """Cap one body and retain a clean sentence when the cap cuts prose."""
    capped = body[:limit]
    if len(body) > limit:
        capped = trim_truncated_snippet(capped)
    return capped.rstrip()


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
    """Fetch once, then resolve cache/flight/LLM while holding a worker.

    The budget is checked here, with the worker slot already held, so it covers
    both ways in are covered: a worker joining the queue for the first time and
    a waiter that was released by a flight and re-queued. The check that a
    worker passed on the way in says nothing about the time left when it
    reaches the front, and what waits at the front is a page fetch and an LLM
    call -- the two things that cost money.
    """
    if not _has_usable_budget(execution):
        return prepared, _GroundingAttempt(
            execution.result, "fallback:pipeline_timeout"
        )
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


def _resolved_worker_deadline(execution: _GroundingExecution) -> float:
    """Return this worker's per-URL deadline clamped to the stage budget.

    A worker that outlives the shared budget would be cancelled from outside,
    which destroys whatever it had already produced. Clamping here lets it end
    on its own with a classified outcome instead.
    """
    own_deadline = asyncio.get_running_loop().time() + (
        execution.context.config.per_url_deadline_ms / 1000
    )
    if execution.budget_deadline_at is None:
        return own_deadline
    return min(own_deadline, execution.budget_deadline_at)


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
    """Run one worker phase, bounding every reacquisition by its deadline.

    The budget is re-checked once the slot is held, inside
    ``_run_grounding_worker_phase``, so both entry paths share it.
    """
    if deadline_at is None:
        async with execution.semaphore:
            resolved_deadline = _resolved_worker_deadline(execution)
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
    """Write one accepted leader result within its original deadline.

    The lifetime is resolved per URL rather than taken from configuration
    directly, because a snippet written from a rolling index is as perishable
    as the index. See ``grounding_cache_ttl_seconds``.
    """
    try:
        async with asyncio.timeout_at(deadline_at):
            async with context.cache_write_semaphore:
                await write_grounding_cache(
                    context.cache,
                    leader.pending,
                    grounding_cache_ttl_seconds(
                        leader.pending.identity.url,
                        context.cache_ttl_seconds,
                        context.fetch_cache_ttl_seconds,
                        context.volatile_cache_ttl_seconds,
                    ),
                )
    except TimeoutError:
        record_grounding_cache_event("write_skipped")
    return leader.attempt


def _result_host(url: str) -> str:
    """Return one result's host for logging, or a stable placeholder."""
    try:
        return urlsplit(url).hostname or "(unparseable)"
    except ValueError:
        return "(unparseable)"


def _has_usable_budget(execution: _GroundingExecution) -> bool:
    """Report whether enough budget remains to be worth paying a fetch for.

    Starting a worker that cannot finish still bills a page fetch and possibly
    an LLM completion before the deadline cancels it, so a worker that reaches
    the front of the queue too late declines rather than spending.
    """
    if execution.budget_deadline_at is None:
        return True
    remaining = execution.budget_deadline_at - asyncio.get_running_loop().time()
    return remaining >= MIN_WORKER_BUDGET_SECONDS


async def _ground_one(
    execution: _GroundingExecution,
) -> tuple[RankedWebResult, GroundingOutcome]:
    """Resolve one URL within worker, flight, and deadline bounds."""
    outcome = await _ground_one_classified(execution)
    if outcome[1] != "grounded":
        _LOGGER.debug(
            "Grounding outcome host=%s outcome=%s",
            _result_host(execution.result.url),
            outcome[1],
        )
    return outcome


async def _ground_one_classified(
    execution: _GroundingExecution,
) -> tuple[RankedWebResult, GroundingOutcome]:
    """Resolve one URL, declining outright when the budget is already spent."""
    deadline_at: float | None = None
    prepared: _GroundingInput | None = None
    if not _has_usable_budget(execution):
        return execution.result, "fallback:pipeline_timeout"
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


def _harvest_worker(
    task: asyncio.Task[tuple[RankedWebResult, GroundingOutcome]],
    result: RankedWebResult,
) -> tuple[RankedWebResult, GroundingOutcome]:
    """Read one finished worker, or classify why it produced nothing.

    Both guards come before ``exception()``, which raises rather than returning
    for a task that is cancelled or still running. The drain gives up after a
    grace period, so a worker that swallowed its cancellation can still be
    unfinished here; it is reported as the timeout it is.
    """
    if not task.done() or task.cancelled():
        return result, "fallback:pipeline_timeout"
    if task.exception() is not None:
        return result, "fallback:worker_rejected"
    return task.result()


async def _drain_pending_workers(
    tasks: list[asyncio.Task[tuple[RankedWebResult, GroundingOutcome]]],
    deadline_at: float | None = None,
) -> None:
    """Cancel every unfinished worker and await it before reading results.

    The wait is bounded twice over. Awaiting a cancelled task normally returns
    at once, but a worker that swallowed its cancellation would otherwise hold
    the request open with nothing left to fire. The grace period caps that, and
    the stage deadline caps the grace period: draining past the deadline would
    push the stage into the backstop, which discards every finished snippet --
    the exact loss this whole stage is built to prevent. A task still running
    afterwards is harvested as the timeout it is.
    """
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if not pending:
        return
    grace_seconds = _DRAIN_GRACE_SECONDS
    if deadline_at is not None:
        remaining = deadline_at - asyncio.get_running_loop().time()
        grace_seconds = max(0.0, min(grace_seconds, remaining))
    await asyncio.wait(pending, timeout=grace_seconds)


def _should_report_grounding_progress(
    completed: int,
    total: int,
    last_reported: int,
) -> bool:
    """Return whether a completion crosses a useful progress milestone."""
    report_interval = max(1, total // 4)
    return (
        completed in (total, 1) or completed - last_reported >= report_interval
    )


async def _wait_for_grounding_workers(
    tasks: list[asyncio.Task[tuple[RankedWebResult, GroundingOutcome]]],
    deadline_at: float | None,
    progress_reporter: GroundingProgressReporter | None,
) -> None:
    """Wait for workers and emit a bounded number of completion updates."""
    pending = set(tasks)
    completed = 0
    last_reported = 0
    loop = asyncio.get_running_loop()
    while pending:
        timeout = (
            None if deadline_at is None else max(0.0, deadline_at - loop.time())
        )
        done, pending = await asyncio.wait(
            pending,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            return
        completed += len(done)
        should_report = _should_report_grounding_progress(
            completed, len(tasks), last_reported
        )
        if progress_reporter is not None and should_report:
            try:
                await progress_reporter(completed, len(tasks))
            except Exception as error:
                _LOGGER.debug(
                    "Grounding progress update skipped error_type=%s",
                    type(error).__name__,
                )
            last_reported = completed


def _log_grounding_summary(
    stats: GroundingStats, elapsed_seconds: float
) -> None:
    """Record what grounding actually produced for this search.

    Emitted at INFO for every grounded search, and escalated when nothing was
    produced. A grounded search that returns zero grounded snippets is
    indistinguishable from an ungrounded one in the response body alone, which
    is precisely the case an operator most needs to see.
    """
    breakdown = " ".join(
        f"{outcome}={count}"
        for outcome, count in sorted(stats.outcomes.items())
    )
    message = (
        "Grounding complete urls=%d grounded=%d transient=%d elapsed_s=%.1f %s"
    )
    arguments = (
        stats.total_urls,
        stats.grounded_count,
        stats.transient_failures,
        elapsed_seconds,
        breakdown,
    )
    if stats.total_urls and not stats.grounded_count:
        _LOGGER.warning(message, *arguments)
        return
    _LOGGER.info(message, *arguments)


async def ground_results(
    query: str,
    results: list[RankedWebResult],
    context: GroundingContext,
    deadline_at: float | None = None,
    *,
    progress_reporter: GroundingProgressReporter | None = None,
) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
    """Ground results in a bounded pool; preserve input order.

    ``deadline_at`` bounds the whole stage. Reaching it cancels only the
    workers still running: every worker that already produced a snippet keeps
    it, because that snippet cost a page fetch and an LLM completion that were
    billed long before the deadline arrived.
    """
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    selected = results[: context.config.top_n]
    if not selected:
        return [], GroundingStats(0, 0, 0, {})
    semaphore = asyncio.Semaphore(context.config.concurrency)
    tasks = [
        asyncio.create_task(
            _ground_one(
                _GroundingExecution(
                    result, query, context, semaphore, deadline_at
                )
            )
        )
        for result in selected
    ]
    try:
        await _wait_for_grounding_workers(
            tasks,
            deadline_at,
            progress_reporter,
        )
        await _drain_pending_workers(tasks, deadline_at)
    except BaseException:
        await _drain_pending_workers(tasks, deadline_at)
        raise
    pairs = [
        _harvest_worker(task, result)
        for task, result in zip(tasks, selected, strict=True)
    ]
    stats = _grounding_stats(pairs)
    _log_grounding_summary(stats, loop.time() - started_at)
    return _marked_pairs(pairs), stats


def _grounding_stats(
    pairs: list[tuple[RankedWebResult, GroundingOutcome]],
) -> GroundingStats:
    """Summarize one grounded search by outcome."""
    counts = Counter(outcome for _, outcome in pairs)
    return GroundingStats(
        transient_failures=sum(
            count
            for outcome, count in counts.items()
            if outcome in TRANSIENT_OUTCOMES
        ),
        grounded_count=counts.get("grounded", 0),
        total_urls=len(pairs),
        outcomes={str(outcome): count for outcome, count in counts.items()},
    )


def _marked_pairs(
    pairs: list[tuple[RankedWebResult, GroundingOutcome]],
) -> list[tuple[RankedWebResult, GroundingOutcome]]:
    """Label every attempted result with the state grounding left it in.

    A result grounding tried and failed keeps its aggregated snippet but is
    marked ``fallback``, so a client can tell that the attempt happened and
    lost from one where grounding never ran at all. Labelling changes no
    snippet text, which keeps the no-erasure invariant intact.
    """
    return [
        (result, outcome)
        if outcome == "grounded"
        else (replace(result, snippet_source="fallback"), outcome)
        for result, outcome in pairs
    ]

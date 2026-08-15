"""Grounding service: success cache + bounded workers + 9 outcomes.

Each URL in the top-N is fetched once (via the in-process omnifetch engine),
junk-detected, looked up by its exact effective LLM input, sent to the
snippet-writing LLM on a miss, sentinel-detected, and classified into one of
nine outcomes. Only accepted LLM output enters the shared cache. Transient
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


async def _fetch_and_ground(
    result: RankedWebResult,
    query: str,
    context: GroundingContext,
) -> _GroundingAttempt:
    """Fetch one URL via the engine waterfall, junk-detect, call LLM."""
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
    return await _ground_user_message(result, title, user_message, context)


async def _ground_user_message(
    result: RankedWebResult,
    fetched_title: str,
    user_message: str,
    context: GroundingContext,
) -> _GroundingAttempt:
    """Return a strict cache hit or call the LLM for one effective input."""
    identity = grounding_cache_identity(user_message, context.config)
    key = make_grounding_cache_key(identity)
    cached = await read_grounding_cache(
        context.cache,
        key,
        identity,
        fetched_title,
    )
    if cached is not None:
        return _accepted_grounding(result, fetched_title, cached)
    try:
        snippet = await _llm_call(
            context.client,
            context.api_key,
            context.config,
            user_message,
        )
    except Exception:
        return _GroundingAttempt(result, "fallback:llm_error")
    return _classify_live_grounding(
        result,
        fetched_title,
        snippet,
        identity,
        key,
    )


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


async def ground_results(
    query: str,
    results: list[RankedWebResult],
    context: GroundingContext,
) -> tuple[list[tuple[RankedWebResult, GroundingOutcome]], GroundingStats]:
    """Ground results in a bounded pool; preserve input order."""
    semaphore = asyncio.Semaphore(context.config.concurrency)
    deadline_s = context.config.per_url_deadline_ms / 1000

    async def ground_one(
        result: RankedWebResult,
    ) -> tuple[RankedWebResult, GroundingOutcome]:
        async with semaphore:
            deadline_at = asyncio.get_running_loop().time() + deadline_s
            try:
                async with asyncio.timeout_at(deadline_at):
                    attempt = await _fetch_and_ground(result, query, context)
            except TimeoutError:
                return result, "fallback:pipeline_timeout"
            except Exception:
                return result, "fallback:worker_rejected"
            if attempt.cache_write is not None:
                try:
                    async with asyncio.timeout_at(deadline_at):
                        await write_grounding_cache(
                            context.cache,
                            attempt.cache_write,
                            context.cache_ttl_seconds,
                        )
                except TimeoutError:
                    record_grounding_cache_event("write_skipped")
            return attempt.result, attempt.outcome

    pairs = await asyncio.gather(
        *(ground_one(r) for r in results[: context.config.top_n])
    )
    outcomes = [outcome for _, outcome in pairs]
    transient = sum(1 for o in outcomes if o in TRANSIENT_OUTCOMES)
    grounded = sum(1 for o in outcomes if o == "grounded")
    return list(pairs), GroundingStats(
        transient_failures=transient,
        grounded_count=grounded,
        total_urls=len(pairs),
    )

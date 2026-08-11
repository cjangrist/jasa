"""Grounding service: bounded worker pool + per-URL deadline + 9 outcomes.

Each URL in the top-N is fetched once (via the in-process omnifetch engine),
junk-detected, sent to the snippet-writing LLM, sentinel-detected, and
classified into one of nine outcomes. Transient outcomes (llm_error,
pipeline_timeout, worker_rejected) block the cache write; durable fallbacks
(junk, sentinel, too-short) do not. Output order follows input order.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Literal

import httpx

from jasa.config import GroundingSettings
from jasa.grounding.detectors import (
    detect_grounded_junk,
    detect_grounded_sentinel,
    repair_unbalanced_fence,
)
from jasa.grounding.prompts import (
    build_grounded_user_message,
    SNIPPET_MAX_CHARS,
    SYSTEM_PROMPT,
)
from jasa.logging import get_logger
from jasa.search.ranking import RankedWebResult
from omnifetch.tools.fetch import execute_web_fetch

_LOGGER = get_logger("grounding")

MIN_CONTENT_CHARS = 50
MIN_SNIPPET_CHARS = 1
TEMPERATURE = 0.2
TOP_P = 0.9
FREQUENCY_PENALTY = 0.3

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
    api_key: str
    config: GroundingSettings


@dataclass(frozen=True, slots=True)
class GroundingStats:
    """Aggregate statistics for one grounded search."""

    transient_failures: int
    grounded_count: int
    total_urls: int


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
            "max_tokens": SNIPPET_MAX_CHARS,
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
) -> tuple[RankedWebResult, GroundingOutcome]:
    """Fetch one URL via the engine waterfall, junk-detect, call LLM."""
    try:
        fetch_result = await execute_web_fetch(context.engine, result.url)
    except Exception:
        return result, "fallback:fetch_exhausted"
    content = getattr(fetch_result, "content", "") or ""
    title = getattr(fetch_result, "title", "") or ""
    if len(content) < MIN_CONTENT_CHARS:
        return result, "fallback:fetch_too_short"
    junk = detect_grounded_junk(content)
    if junk:
        return result, "fallback:fetch_junk"
    user_msg = build_grounded_user_message(
        query, title, content, context.config.max_content_chars
    )
    try:
        snippet = await _llm_call(
            context.client, context.api_key, context.config, user_msg
        )
    except Exception:
        return result, "fallback:llm_error"
    snippet = repair_unbalanced_fence(snippet)
    if len(snippet) < MIN_SNIPPET_CHARS:
        return result, "fallback:llm_empty"
    snippet = snippet[:SNIPPET_MAX_CHARS]
    sentinel = detect_grounded_sentinel(snippet)
    if sentinel:
        return result, "fallback:llm_sentinel"
    updated = replace(
        result,
        snippets=[snippet],
        snippet_source="grounded",
        title=title or result.title,
    )
    return updated, "grounded"


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
            try:
                async with asyncio.timeout(deadline_s):
                    return await _fetch_and_ground(result, query, context)
            except TimeoutError:
                return result, "fallback:pipeline_timeout"
            except Exception:
                return result, "fallback:worker_rejected"

    pairs = await asyncio.gather(
        *(ground_one(r) for r in results[: context.config.top_n])
    )
    outcomes = [outcome for _, outcome in pairs]
    transient = sum(1 for o in outcomes if o in TRANSIENT_OUTCOMES)
    grounded = sum(1 for o in outcomes if o == "grounded")
    return list(pairs), GroundingStats(
        transient_failures=transient,
        grounded_count=grounded,
        total_urls=len(results),
    )

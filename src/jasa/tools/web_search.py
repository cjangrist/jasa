"""The ``web_search`` MCP tool: a thin adapter over ``run_search``.

One execution path per capability (§6): the MCP tool and the REST routes both
call ``run_search`` and format its outcome. The MCP response truncates to the
configured result ceiling plus eligible tail rescues (MCP-only; the cache
already stored the full ranked set) and applies ``include_snippets`` after
retrieval.
"""

from __future__ import annotations

from collections.abc import Mapping

from jasa.cache.base import CacheBackend
from jasa.config import DEFAULT_SEARCH_MAX_RESULTS
from jasa.schemas import (
    WebSearchGrounding,
    WebSearchProviderFailure,
    WebSearchProviderSuccess,
    WebSearchResponse,
    WebSearchResult,
    WebSearchTruncation,
)
from jasa.search.fanout import _FanoutKnobs, ProviderFailure, ProviderSuccess
from jasa.search.providers.base import SearchProvider
from jasa.search.ranking import (
    RankedWebResult,
    truncate_web_results,
    TruncationInfo,
)
from jasa.search.service import (
    _DEFAULT_SEARCH_OPTIONS,
    GroundingReport,
    run_search,
    SearchOptions,
    SearchOutcome,
)


def _result_model(
    result: RankedWebResult, include_snippets: bool
) -> WebSearchResult:
    return WebSearchResult(
        title=result.title,
        url=result.url,
        source_providers=list(result.source_providers),
        score=result.score,
        snippet_source=result.snippet_source or "aggregated",
        snippets=list(result.snippets) if include_snippets else None,
    )


def _success_model(success: ProviderSuccess) -> WebSearchProviderSuccess:
    return WebSearchProviderSuccess(
        provider=success.provider,
        duration_ms=success.duration_ms,
    )


def _failure_model(failure: ProviderFailure) -> WebSearchProviderFailure:
    return WebSearchProviderFailure(
        provider=failure.provider,
        error=failure.error,
        duration_ms=failure.duration_ms,
    )


def _truncation_model(info: TruncationInfo) -> WebSearchTruncation:
    return WebSearchTruncation(
        total_before=info.total_before,
        kept=info.kept,
        rescued=info.rescued,
    )


def _grounding_model(report: GroundingReport | None) -> WebSearchGrounding:
    """Report the grounding stage's state, including when it never ran.

    A successful response says nothing about grounding on its own, so a caller
    that assumed "the search worked" meant "the snippets are grounded" had no
    way to notice otherwise. This block is always present and always answers
    the question directly.
    """
    if report is None:
        return WebSearchGrounding(
            requested=False,
            attempted=0,
            grounded=0,
            outcomes={},
        )
    return WebSearchGrounding(
        requested=report.requested,
        attempted=report.attempted,
        grounded=report.grounded,
        outcomes=dict(report.outcomes),
    )


def format_web_search_response(
    outcome: SearchOutcome,
    *,
    include_snippets: bool = True,
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
) -> WebSearchResponse:
    """Shape the MCP tool response: truncate to top-N + rescue, then format."""
    truncated = truncate_web_results(outcome.web_results, max_results)
    return WebSearchResponse(
        query=outcome.query,
        total_duration_ms=outcome.total_duration_ms,
        providers_succeeded=[
            _success_model(success) for success in outcome.providers_succeeded
        ],
        providers_failed=[
            _failure_model(failure) for failure in outcome.providers_failed
        ],
        grounding=_grounding_model(outcome.grounding),
        truncation=_truncation_model(truncated.truncation),
        web_results=[
            _result_model(result, include_snippets)
            for result in truncated.results
        ],
    )


async def execute_web_search(
    providers: Mapping[str, SearchProvider],
    cache: CacheBackend,
    query: str,
    *,
    options: SearchOptions = _DEFAULT_SEARCH_OPTIONS,
    knobs: _FanoutKnobs | None = None,
) -> WebSearchResponse:
    """Run the search and return the formatted MCP tool response."""
    outcome = await run_search(
        providers, cache, query, options=options, knobs=knobs
    )
    return format_web_search_response(
        outcome,
        include_snippets=options.include_snippets,
    )

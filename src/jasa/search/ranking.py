"""Reciprocal Rank Fusion ranking, ported from omnisearch ``rrf_ranking.ts``.

Merges results from multiple providers into one ranked list: per-provider stable
sort by score, RRF score accumulation over URL dedup keys, snippet collapse, an
optional quality filter, and top-N truncation with tail rescue. Validated
against the ``ranking_*`` and ``truncate_*`` golden fixtures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from jasa.search.snippets import collapse_snippets
from jasa.search.urls import normalize_url

RRF_K = 60
DEFAULT_TOP_N = 20
RESCUE_INTRA_RANK_THRESHOLD = 2
MIN_RRF_SCORE = 0.01
MIN_SNIPPET_CHARS_SINGLE_PROVIDER = 50
MULTI_PROVIDER_THRESHOLD = 2

SnippetSource = Literal["aggregated", "grounded", "fallback"]


@dataclass
class SearchResult:
    """A single raw result returned by one search provider."""

    title: str
    url: str
    snippet: str
    source_provider: str
    score: float | None = None


@dataclass
class RankedWebResult:
    """A merged, ranked result with collapsed snippets."""

    title: str
    url: str
    snippets: list[str]
    source_providers: list[str]
    score: float
    snippet_source: SnippetSource | None = None


@dataclass(frozen=True, slots=True)
class TruncationInfo:
    """Counts describing a top-N truncation with tail rescue."""

    total_before: int
    kept: int
    rescued: int


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """The truncated result list plus truncation metadata."""

    results: list[RankedWebResult]
    truncation: TruncationInfo


def _host_of(url: str) -> str | None:
    """Return the lowercased hostname, or None if the URL is unparseable."""
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        return None
    return hostname.lower() if hostname else None


def _compute_rrf_scores(
    results_by_provider: Mapping[str, list[SearchResult]],
) -> list[tuple[dict[str, object], float]]:
    """Accumulate RRF contributions per normalized URL, in provider order."""
    rrf_scores: dict[str, float] = {}
    url_data: dict[str, dict[str, object]] = {}
    for provider_name, results in results_by_provider.items():
        ranked = sorted(results, key=lambda r: r.score or 0.0, reverse=True)
        ranked_by_url = {
            normalize_url(result.url): result for result in reversed(ranked)
        }
        unique_ranked = [
            ranked_by_url[key]
            for key in dict.fromkeys(
                normalize_url(result.url) for result in ranked
            )
        ]
        for rank, result in enumerate(unique_ranked):
            key = normalize_url(result.url)
            contribution = 1 / (RRF_K + rank + 1)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + contribution
            existing = url_data.get(key)
            if existing is None:
                url_data[key] = {
                    "title": result.title,
                    "url": result.url,
                    "snippets": [result.snippet] if result.snippet else [],
                    "source_providers": [provider_name],
                }
            else:
                providers = existing["source_providers"]
                assert isinstance(providers, list)
                providers.append(provider_name)
                snippets = existing["snippets"]
                assert isinstance(snippets, list)
                if result.snippet and result.snippet not in snippets:
                    snippets.append(result.snippet)
    return [(data, rrf_scores.get(url, 0.0)) for url, data in url_data.items()]


def _apply_quality_filters(
    results: list[RankedWebResult],
) -> list[RankedWebResult]:
    """Drop low-score and thin single-provider results."""

    def keep(result: RankedWebResult) -> bool:
        if result.score < MIN_RRF_SCORE:
            return False
        if len(result.source_providers) >= MULTI_PROVIDER_THRESHOLD:
            return True
        total_snippet_chars = sum(len(snippet) for snippet in result.snippets)
        if total_snippet_chars == 0:
            return True
        return total_snippet_chars >= MIN_SNIPPET_CHARS_SINGLE_PROVIDER

    return [result for result in results if keep(result)]


def _rescue_tail_results(
    top: list[RankedWebResult],
    tail: list[RankedWebResult],
    rescue_threshold: int,
) -> list[RankedWebResult]:
    """Rescue tail results on fresh hosts with a tight intra-provider rank."""
    top_domains = {host for result in top if (host := _host_of(result.url))}

    def rescuable(result: RankedWebResult) -> bool:
        host = _host_of(result.url)
        if host is None or host in top_domains:
            return False
        provider_count = len(result.source_providers)
        per_provider_score = result.score / provider_count
        intra_rank = (1 / per_provider_score) - RRF_K - 1
        return intra_rank < rescue_threshold

    return [result for result in tail if rescuable(result)]


def truncate_web_results(
    results: list[RankedWebResult], top_n: int = DEFAULT_TOP_N
) -> TruncationResult:
    """Truncate to ``top_n`` plus eligible tail rescues."""
    if len(results) <= top_n:
        return TruncationResult(
            results,
            TruncationInfo(len(results), len(results), 0),
        )
    top = results[:top_n]
    tail = results[top_n:]
    rescued = _rescue_tail_results(top, tail, RESCUE_INTRA_RANK_THRESHOLD)
    combined = [*top, *rescued]
    return TruncationResult(
        combined, TruncationInfo(len(results), len(combined), len(rescued))
    )


def rank_and_merge(
    results_by_provider: Mapping[str, list[SearchResult]],
    query: str,
    skip_quality_filter: bool = False,
) -> list[RankedWebResult]:
    """Rank, merge, collapse snippets, and (optionally) quality-filter."""
    scored = _compute_rrf_scores(results_by_provider)
    ranked = [
        RankedWebResult(
            title=data["title"],  # type: ignore[arg-type]
            url=data["url"],  # type: ignore[arg-type]
            snippets=data["snippets"],  # type: ignore[arg-type]
            source_providers=data["source_providers"],  # type: ignore[arg-type]
            score=score,
        )
        for data, score in scored
    ]
    ranked = [result for result in ranked if result.url and result.url.strip()]
    ranked.sort(key=lambda result: result.score, reverse=True)
    collapsed = collapse_snippets(ranked, query)
    return (
        collapsed if skip_quality_filter else _apply_quality_filters(collapsed)
    )

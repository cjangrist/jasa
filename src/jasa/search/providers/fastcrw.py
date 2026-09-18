"""FastCRW Cloud web-search provider.

The hosted API returns a flat ``data`` array for ordinary web search, while
self-hosted FastCRW can wrap flat or source-grouped rows under ``data.results``.
This adapter accepts all documented shapes but requests only the hosted default:
web results without per-result scraping.
"""

from __future__ import annotations

from collections.abc import Mapping

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_SEARCH_PATH = "/v1/search"
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 20
_DEFAULT_TITLE = "Source"
_FAILURE_MESSAGE = "FastCRW Cloud search returned success: false"


class FastcrwProvider(SearchProvider):
    """Search FastCRW Cloud's hosted web index."""

    name = "fastcrw"
    secret_env = "CRW_API_KEY"
    base_url = "https://api.fastcrw.com"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate, search FastCRW Cloud, and normalize its web rows."""
        api_key = self._validated_key()
        query = build_query_with_operators(
            apply_search_operators(parse_search_operators(request.query)),
            list(request.include_domains),
            list(request.exclude_domains),
        )
        limit = min(request.limit or _DEFAULT_LIMIT, _MAX_LIMIT)
        data = await self._fetch(
            f"{self.base_url}{_SEARCH_PATH}",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "limit": limit},
            timeout_s=self.default_timeout_s,
        )
        if not isinstance(data, Mapping) or not data.get("success"):
            raise ProviderError(
                ErrorType.API_ERROR, _FAILURE_MESSAGE, self.name
            )
        return _map_results(data.get("data"), limit, self.name)


def _map_results(data: object, limit: int, provider: str) -> list[SearchResult]:
    """Return well-formed web hits in provider order up to ``limit``."""
    return [
        SearchResult(
            title=_text(row.get("title")) or _DEFAULT_TITLE,
            url=url,
            snippet=_text(row.get("snippet")) or _text(row.get("description")),
            source_provider=provider,
            score=_score(row.get("score")),
        )
        for row in _rows(data)
        if (url := _text(row.get("url")))
    ][:limit]


def _rows(data: object) -> list[Mapping[str, object]]:
    """Extract documented flat, grouped, and wrapped FastCRW web rows."""
    if isinstance(data, Mapping) and "results" in data:
        data = data.get("results")
    if isinstance(data, Mapping):
        data = data.get("web")
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, Mapping)]


def _score(value: object) -> float | None:
    """Return a numeric relevance score without coercing malformed values."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _text(value: object) -> str:
    """Return a string field verbatim, or empty text for malformed values."""
    return value if isinstance(value, str) else ""

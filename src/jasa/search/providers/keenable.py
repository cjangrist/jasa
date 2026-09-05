"""Keenable search provider using the v1 Search API.

POSTs one JSON request with ``X-API-Key`` authentication and always requests
Keenable's maximum of fifty results. One inclusive domain and date operators
use native fields; ambiguous domain policies and unsupported operators stay
rendered. A neutral wildcard keeps operator-only searches non-empty without
silently discarding structural filters.
"""

from __future__ import annotations

from datetime import datetime, UTC

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.providers.keenable_query import (
    build_keenable_body,
    has_promoted_relative_date,
)
from jasa.search.providers.keenable_validation import (
    is_contradictory_date_range,
)
from jasa.search.ranking import SearchResult

_SEARCH_PATH = "/v1/search"


class KeenableProvider(SearchProvider):
    """Keenable web-search adapter."""

    name = "keenable"
    secret_env = "KEENABLE_API_KEY"
    base_url = "https://api.keenable.ai"
    default_timeout_s = 20.0

    def allows_cache(self, query: str) -> bool:
        """Disable long-lived aggregate caching for relative date queries."""
        return not has_promoted_relative_date(
            query, reference_datetime=datetime.now(UTC)
        )

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST native filters, and map ranked hits."""
        api_key = self._validated_key()
        reference_datetime = datetime.now(UTC)
        body = build_keenable_body(
            request, reference_datetime=reference_datetime
        )
        if not body["query"]:
            return []
        date_after = body.get("published_after")
        date_before = body.get("published_before")
        if (
            isinstance(date_after, str)
            and isinstance(date_before, str)
            and is_contradictory_date_range(
                date_after,
                date_before,
                reference_datetime=reference_datetime,
            )
        ):
            return []
        data = await self._fetch(
            f"{self.base_url}{_SEARCH_PATH}",
            method="POST",
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json=body,
            timeout_s=self.default_timeout_s,
        )
        return _map_results(data, self.name)


def _map_results(data: object, provider_name: str) -> list[SearchResult]:
    """Map well-formed result objects and ignore unusable vendor rows."""
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return []
    return [
        SearchResult(
            title=_text(item.get("title")) or url,
            url=url,
            snippet=(
                _text(item.get("snippet")) or _text(item.get("description"))
            ),
            source_provider=provider_name,
        )
        for item in results
        if isinstance(item, dict) and (url := _text(item.get("url")))
    ]


def _text(value: object) -> str:
    """Return a string field verbatim, or empty for any other JSON type."""
    return value if isinstance(value, str) else ""

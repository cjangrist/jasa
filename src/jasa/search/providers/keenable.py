"""Keenable search provider using the v1 Search API.

POSTs one JSON request with ``X-API-Key`` authentication and always requests
Keenable's maximum of fifty results. One inclusive domain and date operators
use native fields when textual query content remains; ambiguous domain policies
and unsupported operators stay rendered so the adapter never sends an empty
query or silently discards intent.
"""

from __future__ import annotations

from typing import cast

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_MAX_RESULTS = 50
_SEARCH_PATH = "/v1/search"


class KeenableProvider(SearchProvider):
    """Keenable web-search adapter."""

    name = "keenable"
    secret_env = "KEENABLE_API_KEY"
    base_url = "https://api.keenable.ai"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST native filters, and map ranked hits."""
        api_key = self._validated_key()
        data = await self._fetch(
            f"{self.base_url}{_SEARCH_PATH}",
            method="POST",
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json=_build_body(request),
            timeout_s=self.default_timeout_s,
        )
        return _map_results(data, self.name)


def _build_body(request: SearchRequest) -> dict[str, object]:
    """Build native filters without allowing them to empty the query."""
    search_params = apply_search_operators(
        parse_search_operators(request.query)
    )
    include_domains = [
        *request.include_domains,
        *cast(list[str], search_params.get("include_domains", [])),
    ]
    exclude_domains = [
        *request.exclude_domains,
        *cast(list[str], search_params.get("exclude_domains", [])),
    ]
    use_structural_site = len(include_domains) == 1
    query_params = {
        name: value
        for name, value in search_params.items()
        if name
        not in {
            "include_domains",
            "exclude_domains",
            "date_after",
            "date_before",
        }
    }
    query = build_query_with_operators(
        query_params,
        None if use_structural_site else include_domains,
        exclude_domains,
    ).strip()
    use_native_filters = bool(query)
    if not use_native_filters:
        query = build_query_with_operators(
            search_params,
            list(request.include_domains),
            list(request.exclude_domains),
        ).strip()
    body: dict[str, object] = {"query": query, "max_results": _MAX_RESULTS}
    if use_native_filters and use_structural_site:
        body["site"] = include_domains[0]
    if use_native_filters and (date_after := search_params.get("date_after")):
        body["published_after"] = str(date_after)
    if use_native_filters and (date_before := search_params.get("date_before")):
        body["published_before"] = str(date_before)
    return body


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

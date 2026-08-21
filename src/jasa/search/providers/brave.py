"""Brave search provider, ported from omnisearch ``providers/search/brave``.

GETs the Brave web-search endpoint with an ``X-Subscription-Token`` header. The
full operator vocabulary is re-rendered into the query string (filetype and
dates included). The response ``description`` field becomes the snippet; no
native score is emitted.

Brave rejects a ``count`` above 20, so the requested limit is clamped to that
maximum rather than forwarded verbatim. The fan-out asks every provider for
the same number, and that number is chosen for the pool rather than for any
one API, so an adapter whose upstream caps lower absorbs the difference here
instead of failing the whole leg.
"""

from __future__ import annotations

from urllib.parse import urlencode

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20
_MAX_COUNT = 20


class BraveProvider(SearchProvider):
    """Brave web-search adapter."""

    name = "brave"
    secret_env = "BRAVE_API_KEY"
    base_url = "https://api.search.brave.com/res/v1"
    default_timeout_s = 10.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, re-render operators, GET, and map results."""
        api_key = self._validated_key()
        search_params = apply_search_operators(
            parse_search_operators(request.query)
        )
        query = build_query_with_operators(
            search_params,
            list(request.include_domains),
            list(request.exclude_domains),
        )
        params = [
            ("q", query),
            (
                "count",
                str(min(request.limit or _DEFAULT_LIMIT, _MAX_COUNT)),
            ),
        ]
        url = f"{self.base_url}/web/search?{urlencode(params)}"
        data = await self._fetch(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            timeout_s=self.default_timeout_s,
        )
        web = data.get("web") if isinstance(data, dict) else None
        results = web.get("results") if isinstance(web, dict) else None
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source_provider=self.name,
            )
            for item in (results or [])
        ]

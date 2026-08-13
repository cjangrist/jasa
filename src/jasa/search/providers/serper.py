"""Serper Google web-search provider.

POSTs a JSON request to Serper's Google search endpoint with ``X-API-KEY``
authentication. Google-compatible operators are preserved in the query, and
the ``organic`` response entries map to Jasa search results.
"""

from __future__ import annotations

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20


class SerperProvider(SearchProvider):
    """Serper Google web-search adapter."""

    name = "serper"
    secret_env = "SERPER_API_KEY"
    base_url = "https://google.serper.dev"
    default_timeout_s = 15.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST a Google query, and map organic results."""
        api_key = self._validated_key()
        search_params = apply_search_operators(
            parse_search_operators(request.query)
        )
        query = build_query_with_operators(
            search_params,
            list(request.include_domains),
            list(request.exclude_domains),
        )
        data = await self._fetch(
            f"{self.base_url}/search",
            method="POST",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "num": request.limit or _DEFAULT_LIMIT,
            },
            timeout_s=self.default_timeout_s,
        )
        results = data.get("organic") if isinstance(data, dict) else None
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet") or "",
                source_provider=self.name,
            )
            for item in (results or [])
        ]

"""Tavily search provider, ported from omnisearch ``providers/search/tavily``.

POSTs to the Tavily search endpoint with Bearer auth. ``site:``/``-site:``
operators are merged into structural include/exclude domain lists (explicit
domains first, operator-derived after); all other operators are parsed out of
the query and dropped, matching upstream. The native score passes through.
"""

from __future__ import annotations

from jasa.search.operators import apply_search_operators, parse_search_operators
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20
_SEARCH_DEPTH = "basic"
_TOPIC = "general"


class TavilyProvider(SearchProvider):
    """Tavily web-search adapter."""

    name = "tavily"
    secret_env = "TAVILY_API_KEY"
    base_url = "https://api.tavily.com"
    default_timeout_s = 30.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, parse operators, POST, and map results."""
        api_key = self._validated_key()
        parsed = apply_search_operators(parse_search_operators(request.query))
        include = parsed.get("include_domains")
        exclude = parsed.get("exclude_domains")
        body = {
            "query": str(parsed["query"]),
            "max_results": request.limit or _DEFAULT_LIMIT,
            "include_domains": [
                *request.include_domains,
                *(include if isinstance(include, list) else []),
            ],
            "exclude_domains": [
                *request.exclude_domains,
                *(exclude if isinstance(exclude, list) else []),
            ],
            "search_depth": _SEARCH_DEPTH,
            "topic": _TOPIC,
        }
        data = await self._fetch(
            f"{self.base_url}/search",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout_s=self.default_timeout_s,
        )
        raw_results = data.get("results") if isinstance(data, dict) else None
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source_provider=self.name,
                score=item.get("score"),
            )
            for item in (raw_results or [])
        ]

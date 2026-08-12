"""Parallel search provider, ported from omnisearch.

POSTs to the Parallel v1 search endpoint with an ``x-api-key`` header in
advanced mode: a fixed objective string and the query wrapped as a
single-element ``search_queries`` array; ``max_results`` and ``source_policy``
nested under ``advanced_settings``. The snippet is ``excerpts`` joined with two
newlines; title falls back to the URL. No native score.
"""

from __future__ import annotations

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20
_MODE = "advanced"
_OBJECTIVE = (
    "Return the most relevant, recent, high-signal sources for this query."
)
_SNIPPET_JOIN = "\n\n"


class ParallelProvider(SearchProvider):
    """Parallel web-search adapter (advanced mode)."""

    name = "parallel"
    secret_env = "PARALLEL_API_KEY"
    base_url = "https://api.parallel.ai"
    default_timeout_s = 30.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST in advanced mode, join excerpts, map."""
        api_key = self._validated_key()
        advanced_settings: dict[str, object] = {
            "max_results": request.limit or _DEFAULT_LIMIT
        }
        source_policy: dict[str, list[str]] = {}
        if request.include_domains:
            source_policy["include_domains"] = list(request.include_domains)
        if request.exclude_domains:
            source_policy["exclude_domains"] = list(request.exclude_domains)
        if source_policy:
            advanced_settings["source_policy"] = source_policy
        body = {
            "objective": _OBJECTIVE,
            "search_queries": [request.query],
            "mode": _MODE,
            "advanced_settings": advanced_settings,
        }
        data = await self._fetch(
            f"{self.base_url}/v1/search",
            method="POST",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=body,
            timeout_s=self.default_timeout_s,
        )
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        return [
            SearchResult(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=_SNIPPET_JOIN.join(item.get("excerpts") or []),
                source_provider=self.name,
            )
            for item in results
            if item.get("url")
        ]

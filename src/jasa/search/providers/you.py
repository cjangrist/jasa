"""You.com search provider using the current JSON Search API contract.

POSTs a JSON request with an ``X-API-Key`` header. The ``snippets`` array is
joined with single spaces, falling back to ``description`` then an empty string.
The ``news`` section is ignored entirely.
"""

from __future__ import annotations

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20


class YouProvider(SearchProvider):
    """You.com web-search adapter."""

    name = "you"
    secret_env = "YOU_API_KEY"
    base_url = "https://ydc-index.io/v1"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST JSON, join snippets, and map web results."""
        api_key = self._validated_key()
        data = await self._fetch(
            f"{self.base_url}/search",
            method="POST",
            headers={
                "X-API-Key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "query": request.query,
                "count": request.limit or _DEFAULT_LIMIT,
            },
            timeout_s=self.default_timeout_s,
        )
        response_results = (
            data.get("results") if isinstance(data, dict) else None
        )
        results = (
            response_results.get("web")
            if isinstance(response_results, dict)
            else None
        )
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(
                    " ".join(item.get("snippets") or [])
                    or item.get("description")
                    or ""
                ),
                source_provider=self.name,
            )
            for item in (results or [])
        ]

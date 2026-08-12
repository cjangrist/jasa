"""You.com search provider, ported from omnisearch ``providers/search/you``.

GETs the You.com search endpoint with an ``X-API-Key`` header. The ``snippets``
array is joined with single spaces, falling back to ``description`` then an
empty string. The ``news`` section is ignored entirely.
"""

from __future__ import annotations

from urllib.parse import urlencode

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
        """Validate the key, GET, join snippets, and map results."""
        api_key = self._validated_key()
        params = [
            ("query", request.query),
            ("count", str(request.limit or _DEFAULT_LIMIT)),
        ]
        url = f"{self.base_url}/search?{urlencode(params)}"
        data = await self._fetch(
            url,
            method="GET",
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout_s=self.default_timeout_s,
        )
        results = data.get("results") if isinstance(data, dict) else None
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

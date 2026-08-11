"""Serpapi search provider, ported from omnisearch ``providers/search/serpapi``.

GETs the SerpAPI endpoint with the API key as a ``api_key`` query parameter --
the highest credential-leak path in the port (the key is in the URL). Engine is
hard-coded to ``google_light``; the snippet defaults to an empty string.
"""

from __future__ import annotations

from urllib.parse import urlencode

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20
_ENGINE = "google_light"


class SerpapiProvider(SearchProvider):
    """SerpAPI web-search adapter (Google lightweight engine)."""

    name = "serpapi"
    secret_env = "SERPAPI_API_KEY"
    base_url = "https://serpapi.com/search.json"
    default_timeout_s = 15.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, GET with the key in the query, and map results."""
        api_key = self._validated_key()
        params = [
            ("engine", _ENGINE),
            ("q", request.query),
            ("api_key", api_key),
            ("num", str(request.limit or _DEFAULT_LIMIT)),
        ]
        url = f"{self.base_url}?{urlencode(params)}"
        data = await self._fetch(
            url, method="GET", timeout_s=self.default_timeout_s
        )
        results = (
            data.get("organic_results") if isinstance(data, dict) else None
        )
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet") or "",
                source_provider=self.name,
            )
            for item in (results or [])
        ]

"""Firecrawl search provider, ported from omnisearch.

POSTs to the Firecrawl v2 search endpoint with Bearer auth. A false ``success``
flag in the envelope is a FAILURE; a missing ``web`` array is a successful
empty result. Title falls back to ``Source``; description to an empty string.
No score is emitted.
"""

from __future__ import annotations

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_DEFAULT_LIMIT = 20
_SEARCH_PATH = "/v2/search"
_DEFAULT_TITLE = "Source"
_SUCCESS_FALSE_MESSAGE = (
    "Failed to fetch search results: Firecrawl API returned success: false"
)


class FirecrawlProvider(SearchProvider):
    """Firecrawl web-search adapter."""

    name = "firecrawl"
    secret_env = "FIRECRAWL_API_KEY"
    base_url = "https://api.firecrawl.dev"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST, enforce the success flag, and map results."""
        api_key = self._validated_key()
        data = await self._fetch(
            f"{self.base_url}{_SEARCH_PATH}",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "query": request.query,
                "limit": request.limit or _DEFAULT_LIMIT,
            },
            timeout_s=self.default_timeout_s,
        )
        if not isinstance(data, dict) or not data.get("success"):
            raise ProviderError(
                ErrorType.API_ERROR, _SUCCESS_FALSE_MESSAGE, self.name
            )
        nested = data.get("data")
        web = nested.get("web") if isinstance(nested, dict) else None
        return [
            SearchResult(
                title=item.get("title") or _DEFAULT_TITLE,
                url=item.get("url", ""),
                snippet=item.get("description") or "",
                source_provider=self.name,
            )
            for item in (web or [])
            if item.get("url")
        ]

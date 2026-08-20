"""Firecrawl search provider, ported from omnisearch.

POSTs to the Firecrawl v2 search endpoint with Bearer auth. A false ``success``
flag in the envelope is a FAILURE; a missing result collection is a successful
empty result. Title falls back to ``Source``; description to an empty string.
No score is emitted.

``data`` is an object keyed by source type (``web``, plus ``news``/``images``
when a request asks for them), not the flat array the original port assumed.
Iterating the object yielded its string keys, so every response raised
``AttributeError`` outside the shared error taxonomy. Only ``web`` is read,
because this adapter requests no other source, and a legacy flat array is still
accepted. Every element is shape-checked, since a search adapter must not raise
on an unexpected payload.
"""

from __future__ import annotations

from typing import Any

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_DEFAULT_LIMIT = 20
_SEARCH_PATH = "/v2/search"
_DEFAULT_TITLE = "Source"
_WEB_COLLECTION = "web"
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
        return [
            SearchResult(
                title=_text(item.get("title")) or _DEFAULT_TITLE,
                url=url,
                snippet=_text(item.get("description")),
                source_provider=self.name,
            )
            for item in _web_results(data.get("data"))
            if (url := _text(item.get("url")))
        ]


def _web_results(collections: object) -> list[dict[str, Any]]:
    """Return the web result mappings from either response shape."""
    if isinstance(collections, dict):
        collections = collections.get(_WEB_COLLECTION)
    if not isinstance(collections, list):
        return []
    return [item for item in collections if isinstance(item, dict)]


def _text(value: object) -> str:
    """Return a string field verbatim, or empty for any other JSON type."""
    return value if isinstance(value, str) else ""

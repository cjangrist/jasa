"""Linkup search provider, ported from omnisearch ``providers/search/linkup``.

POSTs to the Linkup v1 search endpoint with Bearer auth, standard depth and
``searchResults`` output type. Only ``type === 'text'`` results survive;
``name`` maps to title, ``content`` to snippet. Native structural
include/exclude domain filters.
"""

from __future__ import annotations

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20
_DEPTH = "standard"
_OUTPUT_TYPE = "searchResults"
_SEARCH_PATH = "/v1/search"


class LinkupProvider(SearchProvider):
    """Linkup web-search adapter."""

    name = "linkup"
    secret_env = "LINKUP_API_KEY"
    base_url = "https://api.linkup.so"
    default_timeout_s = 30.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST, keep text results, and map name/content."""
        api_key = self._validated_key()
        body: dict[str, object] = {
            "q": request.query,
            "depth": _DEPTH,
            "outputType": _OUTPUT_TYPE,
            "maxResults": request.limit or _DEFAULT_LIMIT,
        }
        if request.include_domains:
            body["includeDomains"] = list(request.include_domains)
        if request.exclude_domains:
            body["excludeDomains"] = list(request.exclude_domains)
        data = await self._fetch(
            f"{self.base_url}{_SEARCH_PATH}",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout_s=self.default_timeout_s,
        )
        results = data.get("results") if isinstance(data, dict) else None
        return [
            SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source_provider=self.name,
            )
            for item in (results or [])
            if item.get("type") == "text"
        ]

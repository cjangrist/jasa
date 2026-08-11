"""Exa search provider, ported from omnisearch ``providers/search/exa``.

POSTs to the Exa search endpoint with BOTH ``x-api-key`` and
``Authorization: Bearer`` headers sent together. Auto search type, autoprompt
on, inline text capped at 1500 chars with livecrawl fallback. The snippet
prefers ``text`` over ``summary`` over a fixed placeholder; a falsy score
becomes 0. Does not parse search operators (operators reach Exa only via the
raw query).
"""

from __future__ import annotations

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20
_SEARCH_TYPE = "auto"
_MAX_CONTENT_CHARS = 1500
_LIVECRAWL = "fallback"
_NO_CONTENT = "No content available"


class ExaProvider(SearchProvider):
    """Exa web-search adapter."""

    name = "exa"
    secret_env = "EXA_API_KEY"
    base_url = "https://api.exa.ai"
    default_timeout_s = 30.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST with both auth headers, and map results."""
        api_key = self._validated_key()
        body: dict[str, object] = {
            "query": request.query,
            "type": _SEARCH_TYPE,
            "numResults": request.limit or _DEFAULT_LIMIT,
            "useAutoprompt": True,
            "contents": {
                "text": {"maxCharacters": _MAX_CONTENT_CHARS},
                "livecrawl": _LIVECRAWL,
            },
        }
        if request.include_domains:
            body["includeDomains"] = list(request.include_domains)
        if request.exclude_domains:
            body["excludeDomains"] = list(request.exclude_domains)
        data = await self._fetch(
            f"{self.base_url}/search",
            method="POST",
            headers={
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout_s=self.default_timeout_s,
        )
        results = data.get("results") if isinstance(data, dict) else None
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("text") or item.get("summary") or _NO_CONTENT,
                source_provider=self.name,
                score=item.get("score") or 0,
            )
            for item in (results or [])
        ]

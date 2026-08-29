"""Ollama web-search provider using the hosted REST API.

POSTs one JSON request to Ollama's ``/api/web_search`` endpoint with Bearer
authentication. Every request asks for Ollama's maximum of ten results so the
provider contributes its full ranked set to the fan-out. The complete operator
vocabulary is re-rendered into the query because the API exposes no structural
domain or date filters.
"""

from __future__ import annotations

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_MAX_RESULTS = 10
_SEARCH_PATH = "/api/web_search"


class OllamaProvider(SearchProvider):
    """Ollama hosted web-search adapter."""

    name = "ollama"
    secret_env = "OLLAMA_API_KEY"
    base_url = "https://ollama.com"
    default_timeout_s = 20.0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST a rendered query, and map ranked hits."""
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
            f"{self.base_url}{_SEARCH_PATH}",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": _MAX_RESULTS,
            },
            timeout_s=self.default_timeout_s,
        )
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        return [
            SearchResult(
                title=_text(item.get("title")) or url,
                url=url,
                snippet=_text(item.get("content")),
                source_provider=self.name,
            )
            for item in results
            if isinstance(item, dict) and (url := _text(item.get("url")))
        ]


def _text(value: object) -> str:
    """Return a string field verbatim, or empty for any other JSON type."""
    return value if isinstance(value, str) else ""

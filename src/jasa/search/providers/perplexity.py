"""Perplexity search provider, ported from omnisearch.

POSTs to the Perplexity chat-completions endpoint (``sonar`` by default,
overridable through ``PERPLEXITY_SEARCH_MODEL``; temperature 0.1, max_tokens
256, high search-context size) with Bearer auth. Prefers the structured
``search_results`` array; falls back to the URL-only ``citations`` array with a
``Source`` title and an empty snippet. The LLM prose in ``choices`` is never
read. No score is emitted.
"""

from __future__ import annotations

from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult

_DEFAULT_LIMIT = 20
_DEFAULT_MODEL = "sonar"
_MODEL_ENV = "PERPLEXITY_SEARCH_MODEL"
_TEMPERATURE = 0.1
_MAX_TOKENS = 256
_CONTEXT_SIZE = "high"
_DEFAULT_TITLE = "Source"


class PerplexityProvider(SearchProvider):
    """Perplexity web-search adapter."""

    name = "perplexity"
    secret_env = "PERPLEXITY_API_KEY"
    base_url = "https://api.perplexity.ai"
    default_timeout_s = 20.0
    setting_envs = (_MODEL_ENV,)
    model_env = _MODEL_ENV
    default_model = _DEFAULT_MODEL

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST, prefer structured results, else citations."""
        api_key = self._validated_key()
        body = {
            "model": self.model_id(),
            "messages": [{"role": "user", "content": request.query}],
            "temperature": _TEMPERATURE,
            "max_tokens": _MAX_TOKENS,
            "web_search_options": {"search_context_size": _CONTEXT_SIZE},
        }
        data = await self._fetch(
            f"{self.base_url}/chat/completions",
            method="POST",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=body,
            timeout_s=self.default_timeout_s,
        )
        limit = request.limit or _DEFAULT_LIMIT
        search_results = (
            data.get("search_results") if isinstance(data, dict) else None
        )
        if isinstance(search_results, list) and search_results:
            structured = [
                SearchResult(
                    title=item.get("title") or _DEFAULT_TITLE,
                    url=item.get("url", ""),
                    snippet=item.get("snippet") or "",
                    source_provider=self.name,
                )
                for item in search_results
                if item.get("url")
            ]
            return structured[:limit]
        citations = data.get("citations") if isinstance(data, dict) else None
        citation_urls = citations or []
        if not citation_urls:
            return []
        return [
            SearchResult(
                title=_DEFAULT_TITLE,
                url=url,
                snippet="",
                source_provider=self.name,
            )
            for url in citation_urls[:limit]
        ]

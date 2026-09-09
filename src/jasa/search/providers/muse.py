"""Meta Muse Spark search through the Responses API's hosted web-search tool.

Raw ``web_search_call.results`` supply source snippets in retrieval order.
Citation-only URLs are appended without snippets: citation spans describe model
prose, not extracted page text. Reasoning and unrelated output are ignored.
Operators stay in the query because Meta documents no native domain filters.
One request per attempt uses Jasa's shared client, error taxonomy, and deadline.
"""

from __future__ import annotations

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_DEFAULT_LIMIT = 30
_DEFAULT_MODEL = "muse-spark-1.2-contributor"
_BASE_URL_ENV = "MUSE_BASE_URL"
_MODEL_ENV = "MUSE_SEARCH_MODEL"
_PROMPT_PREFIX = "Use the web_search tool to search the web for: "
_DEFAULT_ERROR = "Muse web search failed"
_RATE_LIMIT_MARKERS = frozenset({"rate_limit_exceeded", "rate_limit_error"})
_UNFINISHED_STATUSES = frozenset({"incomplete", "in_progress", "queued"})


class MuseProvider(SearchProvider):
    """Meta Muse Spark hosted search adapter."""

    name = "muse"
    secret_env = "MODEL_API_KEY"
    base_url = "https://api.meta.ai/v1"
    default_timeout_s = 60.0
    setting_envs = (_BASE_URL_ENV, _MODEL_ENV)

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Request raw search hits and map them with citation fallbacks."""
        api_key = self._validated_key()
        endpoint = self._setting(_BASE_URL_ENV, self.base_url).rstrip("/")
        query = build_query_with_operators(
            apply_search_operators(parse_search_operators(request.query)),
            list(request.include_domains),
            list(request.exclude_domains),
        )
        data = await self._fetch(
            f"{endpoint}/responses",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._setting(_MODEL_ENV, _DEFAULT_MODEL),
                "input": _PROMPT_PREFIX + query,
                "tools": [
                    {"type": "web_search", "search_context_size": "medium"}
                ],
                "include": ["web_search_call.results"],
            },
            timeout_s=self.default_timeout_s,
        )
        payload = data if isinstance(data, dict) else {}
        failure = _failure(payload)
        if failure is not None:
            error_type, message = failure
            raise ProviderError(
                error_type, self._redact_secret(message), self.name
            )
        return _map_results(payload, request.limit or _DEFAULT_LIMIT, self.name)


def _mappings(items: object, item_type: str) -> list[dict[str, object]]:
    """Select typed mappings without assuming well-formed upstream arrays."""
    if not isinstance(items, list):
        return []
    return [
        item
        for item in items
        if isinstance(item, dict) and item.get("type") == item_type
    ]


def _text(value: object) -> str:
    """Return a string field or empty text for malformed leaf values."""
    return value if isinstance(value, str) else ""


def _failure(payload: dict[str, object]) -> tuple[ErrorType, str] | None:
    """Keep explicit failures from becoming cacheable empty searches."""
    error = payload.get("error")
    if isinstance(error, dict):
        markers = {_text(error.get("code")), _text(error.get("type"))}
        error_type = (
            ErrorType.RATE_LIMIT
            if markers & _RATE_LIMIT_MARKERS
            else ErrorType.API_ERROR
        )
        return error_type, _text(error.get("message")) or _DEFAULT_ERROR
    if isinstance(error, str) and error:
        return ErrorType.API_ERROR, error
    if payload.get("status") in ("failed", "cancelled"):
        return ErrorType.API_ERROR, _DEFAULT_ERROR
    return None


def _collect_hits(output: object) -> list[tuple[str, str, str]]:
    """Prefer raw source snippets, then append any citation-only sources."""
    raw_hits = [
        (
            _text(hit.get("title")),
            _text(hit.get("url")),
            _text(hit.get("snippet")),
        )
        for item in _mappings(output, "web_search_call")
        for hit in _mappings(item.get("results"), "text_result")
    ]
    citations = [
        (_text(citation.get("title")), _text(citation.get("url")), "")
        for item in _mappings(output, "message")
        for block in _mappings(item.get("content"), "output_text")
        for citation in _mappings(block.get("annotations"), "url_citation")
    ]
    collected: dict[str, tuple[str, str, str]] = {}
    for title, url, snippet in (*raw_hits, *citations):
        if url and url not in collected:
            collected[url] = (title or url, url, snippet)
    return list(collected.values())


def _map_results(
    payload: dict[str, object], limit: int, provider: str
) -> list[SearchResult]:
    """Return surviving hits or expose an unfinished/failed search turn."""
    hits = _collect_hits(payload.get("output"))
    if not hits:
        calls = _mappings(payload.get("output"), "web_search_call")
        if any(call.get("status") == "failed" for call in calls):
            raise ProviderError(
                ErrorType.PROVIDER_ERROR, _DEFAULT_ERROR, provider
            )
        if _text(payload.get("status")) in _UNFINISHED_STATUSES:
            raise ProviderError(
                ErrorType.PROVIDER_ERROR,
                "Muse ended the search turn before returning a result",
                provider,
            )
    return [
        SearchResult(
            title=title, url=url, snippet=snippet, source_provider=provider
        )
        for title, url, snippet in hits[:limit]
    ]

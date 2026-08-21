"""Z.AI search provider backed by the GLM chat-completions web-search tool.

Z.AI exposes web search as a server-side tool on a chat completion rather than
as a search endpoint: the standalone ``/web_search`` route is billed separately
and a coding-plan credential cannot reach it. Results arrive in a top-level
``web_search`` array carrying the upstream ranked list, so a hit does not
depend on the model citing it, and ``content`` is source text rather than model
prose. Array position carries rank for downstream RRF.

Because the model's answer is never read, generation is capped at a single
token. The search runs regardless and the saving is large -- roughly 13s to
2.3s per request in measurement -- since the turn no longer waits on prose the
adapter would discard. This is the one behavior here that the vendor does not
document, so it is the first thing to check if latency regresses.

The tool honors ``count`` only up to ten and ignores it above that; ten is
therefore a ceiling rather than a default and the requested limit is clamped
rather than forwarded. Domain and recency filters were measured as accepted-
but-ignored -- ``search_domain_filter``, ``domain_filter``, and
``search_recency_filter`` all returned byte-identical hosts -- so every
operator is re-rendered into the query text instead, and none is sent
structurally where it would be silently dropped.

A failure reported inside a 200 body fails the request rather than reading as
an empty result. That distinction matters more here than the empty list
suggests: an empty success is cacheable, so the error would be written into the
shared search cache as part of a complete fan-out and served for its whole TTL.

``search-prime`` is the only working engine; ``search-std``, ``search-pro``,
and ``search-prime-x`` each return zero hits. ``Z_AI_BASE_URL`` and
``ZAI_SEARCH_MODEL`` retarget the adapter, and the model id is a release-time
review item like the other LLM-backed adapters.
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

_DEFAULT_LIMIT = 10
_MAX_COUNT = 10
_MAX_TOKENS = 1
_DEFAULT_MODEL = "glm-4.6"
_BASE_URL_ENV = "Z_AI_BASE_URL"
_MODEL_ENV = "ZAI_SEARCH_MODEL"
_SEARCH_PATH = "/chat/completions"
_TOOL_TYPE = "web_search"
_SEARCH_ENGINE = "search-prime"
_DEFAULT_ERROR = "Z.AI web search failed"
_INSUFFICIENT_BALANCE_CODE = "1113"


class ZaiProvider(SearchProvider):
    """Z.AI GLM web-search adapter (server-tool web search)."""

    name = "zai"
    secret_env = "Z_AI_API_KEY"
    base_url = "https://api.z.ai/api/coding/paas/v4"
    default_timeout_s = 30.0
    setting_envs = (_BASE_URL_ENV, _MODEL_ENV)

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST one tool-backed search, and map hits."""
        api_key = self._validated_key()
        endpoint = self._setting(_BASE_URL_ENV, self.base_url).rstrip("/")
        search_params = apply_search_operators(
            parse_search_operators(request.query)
        )
        query = build_query_with_operators(
            search_params,
            list(request.include_domains),
            list(request.exclude_domains),
        )
        count = min(request.limit or _DEFAULT_LIMIT, _MAX_COUNT)
        data = await self._fetch(
            f"{endpoint}{_SEARCH_PATH}",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._setting(_MODEL_ENV, _DEFAULT_MODEL),
                "messages": [{"role": "user", "content": query}],
                "tools": [_build_tool(count)],
                "max_tokens": _MAX_TOKENS,
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
        hits = _collect_hits(payload.get("web_search"))
        return [
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source_provider=self.name,
            )
            for title, url, snippet in hits[:count]
        ]


def _build_tool(count: int) -> dict[str, object]:
    """Build the server-tool definition for one clamped result count."""
    return {
        "type": _TOOL_TYPE,
        _TOOL_TYPE: {
            "enable": True,
            "search_engine": _SEARCH_ENGINE,
            "search_result": True,
            "count": count,
        },
    }


def _failure(payload: dict[str, object]) -> tuple[ErrorType, str] | None:
    """Return the category and message of an explicit in-body failure.

    A gateway reached through ``Z_AI_BASE_URL`` may report a failure inside a
    200 body rather than as an HTTP status. Without this check the absent
    ``web_search`` key would read as a successful empty result, and an empty
    success is cacheable: the error would be written into the shared search
    cache as part of a complete fan-out and served for its whole TTL. Any
    ``error`` mapping therefore fails closed, even an empty one.

    Insufficient balance is classified as a rate limit so an in-body report
    lands in the same category as the HTTP 429 the vendor sends for it
    directly.
    """
    error = payload.get("error")
    if isinstance(error, dict):
        code = _text(error.get("code"))
        message = _text(error.get("message")) or _DEFAULT_ERROR
        if code == _INSUFFICIENT_BALANCE_CODE:
            return ErrorType.RATE_LIMIT, message
        return ErrorType.API_ERROR, message
    if isinstance(error, str) and error:
        return ErrorType.API_ERROR, error
    return None


def _text(value: object) -> str:
    """Return a string field verbatim, or empty for any other JSON type.

    Leaf fields are shape-checked like the blocks that hold them: a non-string
    ``link`` must not reach a set, and a non-string title or excerpt must not
    be coerced into its ``repr``.
    """
    return value if isinstance(value, str) else ""


def _collect_hits(hits: object) -> list[tuple[str, str, str]]:
    """Return distinct ``(title, url, snippet)`` triples in upstream order.

    A gateway reached through ``Z_AI_BASE_URL`` is not obliged to return
    well-formed hits, so every entry is shape-checked before it is read and a
    malformed one is ignored rather than raising outside the shared taxonomy.
    """
    collected: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    if not isinstance(hits, list):
        return collected
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        url = _text(hit.get("link"))
        if not url or url in seen:
            continue
        seen.add(url)
        collected.append(
            (_text(hit.get("title")) or url, url, _text(hit.get("content")))
        )
    return collected

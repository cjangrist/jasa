"""Claude search provider backed by the Anthropic Messages API.

POSTs one non-streaming Messages request carrying the ``web_search_20250305``
server tool. Anthropic runs the search itself and returns
``web_search_tool_result`` blocks holding the ranked result list, then cites the
sources it used with verbatim ``cited_text`` excerpts; those excerpts become the
snippet for their URL, so a snippet here is source text rather than model prose.
Result order is the upstream rank, so no native score is emitted.

Domain operators map to the tool's ``allowed_domains``/``blocked_domains``
lists, which the API rejects when both are sent; an include list therefore wins
and the excluded domains are re-rendered as ``-site:`` query operators instead
of being dropped. Every other operator stays in the query text.

Tool errors arrive inside a 200 response as a single error object rather than a
result list. They fail the request only when no search result survived, so a
late ``max_uses_exceeded`` cannot discard results that already arrived.

``ANTHROPIC_BASE_URL`` retargets the adapter at any Messages-compatible gateway
and ``CLAUDE_SEARCH_MODEL`` selects the model that drives the tool, because an
exact dated model id goes stale and gateways publish their own ids. Both
``x-api-key`` and ``Authorization: Bearer`` are sent so a provider-native API
key and a gateway bearer token each authenticate.

``_DEFAULT_MODEL`` is a dated id that Anthropic eventually retires, so it is a
release-time review item: check it against the vendor's model-deprecation page
and update the constant, ``.env.example``, and ``README.md`` together. An
operator can move off a retired default at any time through the setting.

The request budget matches the repository's other LLM timeout default because
one search pays for an inference turn on top of the upstream search. The
fan-out deadline still governs a normal request.
"""

from __future__ import annotations

from typing import Any, cast

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_DEFAULT_LIMIT = 20
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
_MODEL_ENV = "CLAUDE_SEARCH_MODEL"
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_TOKENS = 2048
_MAX_USES = 1
_TOOL_TYPE = "web_search_20250305"
_TOOL_NAME = "web_search"
_SYSTEM_PROMPT = (
    "You are a web-search tool. Search the web for the user's query, then "
    "summarize what the sources say and cite every source you use."
)
_USER_PROMPT_PREFIX = "Search the web for: "
_DOMAIN_FIELDS = frozenset({"include_domains", "exclude_domains"})
_SEARCH_RESULT = "web_search_result"
_SEARCH_RESULT_BLOCK = "web_search_tool_result"
_TEXT_BLOCK = "text"
_CITATION_LOCATION = "web_search_result_location"
_RATE_LIMIT_ERROR_CODE = "too_many_requests"
_SNIPPET_JOIN = " "


class ClaudeProvider(SearchProvider):
    """Anthropic Claude web-search adapter (server-tool web search)."""

    name = "claude"
    secret_env = "ANTHROPIC_AUTH_TOKEN"
    base_url = "https://api.anthropic.com"
    default_timeout_s = 60.0
    setting_envs = (_BASE_URL_ENV, _MODEL_ENV)

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST one server-tool search, and map results."""
        api_key = self._validated_key()
        endpoint = self._setting(_BASE_URL_ENV, self.base_url).rstrip("/")
        search_params = apply_search_operators(
            parse_search_operators(request.query)
        )
        include_domains = [
            *request.include_domains,
            *cast(list[str], search_params.get("include_domains", [])),
        ]
        exclude_domains = [
            *request.exclude_domains,
            *cast(list[str], search_params.get("exclude_domains", [])),
        ]
        data = await self._fetch(
            f"{endpoint}/v1/messages",
            method="POST",
            headers={
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "anthropic-version": _ANTHROPIC_VERSION,
            },
            json={
                "model": self._setting(_MODEL_ENV, _DEFAULT_MODEL),
                "max_tokens": _MAX_TOKENS,
                "system": _SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": _USER_PROMPT_PREFIX
                        + _build_query(
                            search_params, include_domains, exclude_domains
                        ),
                    }
                ],
                "tools": [_build_tool(include_domains, exclude_domains)],
            },
            timeout_s=self.default_timeout_s,
        )
        blocks = data.get("content") if isinstance(data, dict) else None
        hits, error_code = _collect_hits(blocks)
        if not hits and error_code is not None:
            raise ProviderError(
                _error_type(error_code),
                f"Claude web search failed: {error_code}",
                self.name,
            )
        excerpts = _collect_excerpts(blocks)
        return [
            SearchResult(
                title=title,
                url=url,
                snippet=_SNIPPET_JOIN.join(excerpts.get(url, [])),
                source_provider=self.name,
            )
            for title, url in hits[: request.limit or _DEFAULT_LIMIT]
        ]


def _build_query(
    search_params: dict[str, object],
    include_domains: list[str],
    exclude_domains: list[str],
) -> str:
    """Re-render the non-domain operators plus any unsent exclusions."""
    query_params = {
        name: value
        for name, value in search_params.items()
        if name not in _DOMAIN_FIELDS
    }
    unsent_exclusions = exclude_domains if include_domains else []
    return build_query_with_operators(query_params, [], unsent_exclusions)


def _build_tool(
    include_domains: list[str], exclude_domains: list[str]
) -> dict[str, object]:
    """Build the server-tool definition with at most one domain list."""
    tool: dict[str, object] = {
        "type": _TOOL_TYPE,
        "name": _TOOL_NAME,
        "max_uses": _MAX_USES,
    }
    if include_domains:
        tool["allowed_domains"] = include_domains
    elif exclude_domains:
        tool["blocked_domains"] = exclude_domains
    return tool


def _mappings(items: object, block_type: str) -> list[dict[str, Any]]:
    """Return the mapping entries of ``items`` whose ``type`` matches.

    A Messages-compatible gateway is not obliged to return well-formed blocks,
    so every element is shape-checked before it is read. Malformed entries are
    ignored rather than raising outside the shared error taxonomy.
    """
    if not isinstance(items, list):
        return []
    return [
        item
        for item in items
        if isinstance(item, dict) and item.get("type") == block_type
    ]


def _collect_hits(
    blocks: object,
) -> tuple[list[tuple[str, str]], str | None]:
    """Return distinct ranked ``(title, url)`` pairs and any error code."""
    hits: list[tuple[str, str]] = []
    seen: set[str] = set()
    error_code: str | None = None
    for block in _mappings(blocks, _SEARCH_RESULT_BLOCK):
        content = block.get("content")
        if isinstance(content, dict):
            code = content.get("error_code")
            if error_code is None and isinstance(code, str):
                error_code = code
            continue
        for hit in _mappings(content, _SEARCH_RESULT):
            url = hit.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            hits.append((str(hit.get("title") or url), str(url)))
    return hits, error_code


def _collect_excerpts(blocks: object) -> dict[str, list[str]]:
    """Return the distinct cited source excerpts keyed by cited URL."""
    excerpts: dict[str, list[str]] = {}
    for block in _mappings(blocks, _TEXT_BLOCK):
        citations = block.get("citations")
        for citation in _mappings(citations, _CITATION_LOCATION):
            url = citation.get("url")
            if not url:
                continue
            collected = excerpts.setdefault(str(url), [])
            cited_text = citation.get("cited_text")
            if cited_text and cited_text not in collected:
                collected.append(str(cited_text))
    return excerpts


def _error_type(error_code: str) -> ErrorType:
    """Map a server-tool error code onto the shared error taxonomy."""
    if error_code == _RATE_LIMIT_ERROR_CODE:
        return ErrorType.RATE_LIMIT
    return ErrorType.API_ERROR

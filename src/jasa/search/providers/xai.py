"""Agentic Grok web search through a Responses-compatible gateway.

Grok selects sources and writes titles/descriptions; these are not verbatim
SERP rows. Require an observed hosted web-search call and reject unsafe URLs
before allowing model-generated links into Jasa's ranking/grounding pipeline.
"""

from __future__ import annotations

import ipaddress
import json
import re
from urllib.parse import urlsplit

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_DEFAULT_MODEL = "grok-4.7-build-fast"
_MAX_DOMAIN_FILTERS = 5
_BASE_URL_ENV = "XAI_SEARCH_BASE_URL"
_MODEL_ENV = "XAI_SEARCH_MODEL"
_PROMPT_TEMPLATE = (
    "Use the web_search tool to find current information for the query below, "
    "then respond with ONLY a single JSON object — no prose, no markdown "
    "fences, no inline citation links — matching this exact schema:\n\n"
    '{{"results": [{{"title": "string", "url": "string", '
    '"description": "1-2 sentence summary"}}]}}\n\n'
    "Return at most {limit} results, ordered by relevance, with absolute "
    'https:// URLs. If no usable results exist, return {{"results": []}}.\n\n'
    "Query: {query}"
)
_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")
_NONCANONICAL_IP = re.compile(
    r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*"
)
_INVALID_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".test",
    ".invalid",
)
_RATE_LIMIT_MARKERS = frozenset({"rate_limit_exceeded", "rate_limit_error"})


def _endpoint(configured: str) -> str:
    """Require HTTPS and an uncredentialed authority before sending the key."""
    try:
        parts = urlsplit(configured)
        _port = parts.port
        valid = (
            parts.scheme == "https"
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ProviderError(
            ErrorType.INVALID_INPUT,
            "XAI_SEARCH_BASE_URL must be an absolute HTTPS URL without "
            "credentials, query or fragment",
            "xai",
        )
    return configured.rstrip("/")


def _safe_url(url: str) -> bool:
    """Model-generated links must not point grounding at local/private hosts."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _port = parts.port
    except ValueError:
        return False
    if parts.scheme != "https" or not host or parts.username is not None:
        return False
    lowered = host.lower()
    if lowered == "localhost" or lowered.endswith(_INVALID_HOST_SUFFIXES):
        return False
    try:
        return ipaddress.ip_address(lowered).is_global
    except ValueError:
        # Standard resolvers accept short, octal, hex, and trailing-dot IPs.
        # Never treat a rejected IP spelling as a public DNS hostname.
        if _NONCANONICAL_IP.fullmatch(lowered.rstrip(".")):
            return False
        return "." in lowered


def _objects(value: object) -> list[dict[str, object]]:
    """Keep mappings from an untrusted Responses collection."""
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _row(value: object) -> tuple[str, str, str] | None:
    if not isinstance(value, dict):
        return None
    url = _text(value.get("url"))
    if not _safe_url(url):
        return None
    return (
        _text(value.get("title")) or url,
        url,
        _text(value.get("description")),
    )


def _json_rows(text: str, limit: int) -> list[tuple[str, str, str]]:
    match = _JSON_BLOCK.search(text)
    for candidate in (text, match.group(0) if match else ""):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            return [
                row for item in parsed["results"][:limit] if (row := _row(item))
            ]
    return []


def _collect_rows(
    payload: dict[str, object], limit: int
) -> list[tuple[str, str, str]]:
    """Prefer JSON rows, then annotation URLs, then top-level citations."""
    messages = [
        item
        for item in _objects(payload.get("output"))
        if item.get("type") == "message"
    ]
    blocks = [
        block
        for item in messages
        for block in _objects(item.get("content"))
        if block.get("type") == "output_text"
    ]
    for block in blocks:
        if rows := _json_rows(_text(block.get("text")), limit):
            return rows
    annotations = [
        ann
        for block in blocks
        for ann in _objects(block.get("annotations"))
        if ann.get("type") == "url_citation"
    ]
    # Citation titles are positional labels ("1", "2"), not page titles.
    sources = [_row({"url": ann.get("url")}) for ann in annotations]
    if not any(sources):
        citations = payload.get("citations")
        sources = (
            [_row({"url": url}) for url in citations if isinstance(url, str)]
            if isinstance(citations, list)
            else []
        )
    return [row for row in sources if row][:limit]


def _error(payload: dict[str, object]) -> tuple[ErrorType, str] | None:
    error = payload.get("error")
    if isinstance(error, dict):
        markers = {_text(error.get("code")), _text(error.get("type"))}
        kind = (
            ErrorType.RATE_LIMIT
            if markers & _RATE_LIMIT_MARKERS
            else ErrorType.API_ERROR
        )
        return kind, _text(error.get("message")) or "xAI web search failed"
    if isinstance(error, str) and error:
        return ErrorType.API_ERROR, error
    if payload.get("status") in ("failed", "cancelled"):
        return ErrorType.API_ERROR, "xAI web search failed"
    return None


class XaiProvider(SearchProvider):
    """Hosted Grok web-search adapter (model-selected source rows)."""

    name = "xai"
    secret_env = "XAI_API_KEY"
    base_url = "https://ai.angrist.net/v1"
    default_timeout_s = 60.0
    setting_envs = (_BASE_URL_ENV, _MODEL_ENV)

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Search once through Grok and map only public HTTPS source URLs."""
        key = self._validated_key()
        endpoint = _endpoint(self._setting(_BASE_URL_ENV, self.base_url))
        params = apply_search_operators(parse_search_operators(request.query))
        includes = [
            *request.include_domains,
            *_domain_list(params.get("include_domains")),
        ]
        excludes = [
            *request.exclude_domains,
            *_domain_list(params.get("exclude_domains")),
        ]
        query = build_query_with_operators(
            params, list(request.include_domains), list(request.exclude_domains)
        )
        tool: dict[str, object] = {"type": "web_search"}
        allowed = list(dict.fromkeys(includes))
        blocked = list(dict.fromkeys(excludes))
        if allowed and not blocked and len(allowed) <= _MAX_DOMAIN_FILTERS:
            tool["filters"] = {"allowed_domains": allowed}
        elif blocked and not allowed and len(blocked) <= _MAX_DOMAIN_FILTERS:
            tool["filters"] = {"excluded_domains": blocked}
        # Too many or mixed domains stay rendered in the query; a truncated
        # allowlist would make the omitted domains impossible to find.
        limit = request.limit or 30
        data = await self._fetch(
            f"{endpoint}/responses",
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._setting(_MODEL_ENV, _DEFAULT_MODEL),
                "input": [
                    {
                        "role": "user",
                        "content": _PROMPT_TEMPLATE.format(
                            query=query, limit=limit
                        ),
                    }
                ],
                "tools": [tool],
                "include": ["no_inline_citations"],
            },
            timeout_s=self.default_timeout_s,
        )
        payload = data if isinstance(data, dict) else {}
        if failure := _error(payload):
            raise ProviderError(
                failure[0], self._redact_secret(failure[1]), self.name
            )
        calls = [
            item
            for item in _objects(payload.get("output"))
            if item.get("type") == "web_search_call"
        ]
        if not calls:
            raise ProviderError(
                ErrorType.PROVIDER_ERROR,
                "xAI did not execute web_search",
                self.name,
            )
        rows = _collect_rows(payload, limit)
        failed_search = any(call.get("status") == "failed" for call in calls)
        completed_search = any(
            call.get("status") == "completed" for call in calls
        )
        if failed_search and (not rows or not completed_search):
            raise ProviderError(
                ErrorType.PROVIDER_ERROR,
                "xAI web_search call failed",
                self.name,
            )
        if not rows and payload.get("status") in (
            "incomplete",
            "in_progress",
            "queued",
        ):
            raise ProviderError(
                ErrorType.PROVIDER_ERROR,
                "xAI search turn ended before returning results",
                self.name,
            )
        unique: dict[str, tuple[str, str]] = {}
        for title, url, description in rows:
            unique.setdefault(url, (title, description))
        return [
            SearchResult(
                title=title,
                url=url,
                snippet=description,
                source_provider=self.name,
            )
            for url, (title, description) in list(unique.items())[:limit]
        ]


def _domain_list(value: object) -> list[str]:
    return (
        [item for item in value if isinstance(item, str)]
        if isinstance(value, list)
        else []
    )

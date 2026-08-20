"""Codex search provider backed by the OpenAI Responses API.

POSTs one non-streaming Responses request carrying the hosted ``web_search``
tool. OpenAI runs the search itself and reports the sources it used as
``url_citation`` annotations on the assistant message; those annotations carry
only a title and a URL, so results are ordered by first citation and no native
score is emitted. The hosted tool exposes no per-result excerpt, and the
message text is model prose rather than source content, so snippets stay empty
and let ranking, other providers, or grounding supply them.

Every citation URL is tagged with ``utm_source=openai``. Tracking parameters
are stripped so the dedup key matches the same page found by another provider;
a URL without them is passed through byte-for-byte.

Domain operators map to the tool's ``filters``, which accepts allow and block
lists together; every other operator stays in the query text. A ``failed``
response is an explicit provider failure, and so is an ``incomplete`` one that
produced no citation: that outcome is transient, so it is reported as a
provider error the fan-out retries once rather than a successful empty result
the search cache would keep. An ``incomplete`` response that did produce
citations returns them. Because the base URL can point at any
Responses-compatible gateway, every output item, content part, annotation, and
leaf field is shape-checked before it is read, so a malformed payload is
ignored rather than raising outside the shared error taxonomy.

``OPENAI_BASE_URL`` retargets the adapter at any Responses-compatible gateway
and ``CODEX_SEARCH_MODEL`` selects the model that drives the tool, because
gateways publish their own model ids and hosted-search model support changes
over time. ``_DEFAULT_MODEL`` is therefore a release-time review item: check it
against the vendor's model list and update the constant, ``.env.example``, and
``README.md`` together.

The request budget matches the repository's other LLM timeout default because
one search pays for an inference turn on top of the upstream search. The
fan-out deadline still governs a normal request.
"""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from jasa.search.operators import (
    apply_search_operators,
    build_query_with_operators,
    parse_search_operators,
)
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from omnifetch.fetch.shared.types import ErrorType, ProviderError

_DEFAULT_LIMIT = 20
_DEFAULT_MODEL = "gpt-5.6"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_MODEL_ENV = "CODEX_SEARCH_MODEL"
_TOOL_TYPE = "web_search"
_SEARCH_CONTEXT_SIZE = "medium"
_USER_PROMPT_PREFIX = "Use the web_search tool to search the web for: "
_DOMAIN_FIELDS = frozenset({"include_domains", "exclude_domains"})
_MESSAGE_ITEM = "message"
_URL_CITATION = "url_citation"
_FAILED_STATUS = "failed"
_INCOMPLETE_STATUS = "incomplete"
_INCOMPLETE_MESSAGE = "OpenAI ended the search turn early without a result"
_RATE_LIMIT_MARKERS = frozenset({"rate_limit_exceeded", "rate_limit_error"})
_TRACKING_PREFIX = "utm_"
_DEFAULT_ERROR = "OpenAI web search failed"


class CodexProvider(SearchProvider):
    """OpenAI hosted web-search adapter (Responses API)."""

    name = "codex"
    secret_env = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1"
    default_timeout_s = 60.0
    setting_envs = (_BASE_URL_ENV, _MODEL_ENV)

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Validate the key, POST one hosted search, and map citations."""
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
            f"{endpoint}/responses",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._setting(_MODEL_ENV, _DEFAULT_MODEL),
                "input": _USER_PROMPT_PREFIX + _build_query(search_params),
                "tools": [_build_tool(include_domains, exclude_domains)],
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
        citations = _collect_citations(payload.get("output"))
        if not citations and payload.get("status") == _INCOMPLETE_STATUS:
            raise ProviderError(
                ErrorType.PROVIDER_ERROR, _INCOMPLETE_MESSAGE, self.name
            )
        return [
            SearchResult(
                title=title,
                url=url,
                snippet="",
                source_provider=self.name,
            )
            for title, url in citations[: request.limit or _DEFAULT_LIMIT]
        ]


def _build_query(search_params: dict[str, object]) -> str:
    """Re-render every operator the structural filters do not carry."""
    query_params = {
        name: value
        for name, value in search_params.items()
        if name not in _DOMAIN_FIELDS
    }
    return build_query_with_operators(query_params)


def _build_tool(
    include_domains: list[str], exclude_domains: list[str]
) -> dict[str, object]:
    """Build the hosted-tool definition with both domain filters."""
    tool: dict[str, object] = {
        "type": _TOOL_TYPE,
        "search_context_size": _SEARCH_CONTEXT_SIZE,
    }
    filters: dict[str, list[str]] = {}
    if include_domains:
        filters["allowed_domains"] = include_domains
    if exclude_domains:
        filters["blocked_domains"] = exclude_domains
    if filters:
        tool["filters"] = filters
    return tool


def _mappings(items: object, item_type: str | None = None) -> list[Any]:
    """Return the mapping entries of ``items``, optionally filtered by type.

    A Responses-compatible gateway is not obliged to return well-formed output
    items, so every element is shape-checked before it is read. Malformed
    entries are ignored rather than raising outside the shared error taxonomy.
    """
    if not isinstance(items, list):
        return []
    return [
        item
        for item in items
        if isinstance(item, dict)
        and (item_type is None or item.get("type") == item_type)
    ]


def _text(value: object) -> str:
    """Return a string field verbatim, or empty for any other JSON type."""
    return value if isinstance(value, str) else ""


def _failure(payload: dict[str, Any]) -> tuple[ErrorType, str] | None:
    """Return the category and message of an explicit in-body failure.

    A gateway reached through ``OPENAI_BASE_URL`` may report a rate limit in a
    200 body rather than as HTTP 429, so an error object is inspected for a
    rate-limit marker before falling back to ``API_ERROR``.
    """
    error = payload.get("error")
    if isinstance(error, dict):
        message = _text(error.get("message")) or _DEFAULT_ERROR
        return _error_type(error), message
    if isinstance(error, str) and error:
        return ErrorType.API_ERROR, error
    if payload.get("status") == _FAILED_STATUS:
        return ErrorType.API_ERROR, _DEFAULT_ERROR
    return None


def _error_type(error: dict[str, Any]) -> ErrorType:
    """Map an in-body error object onto the shared error taxonomy."""
    markers = {_text(error.get("code")), _text(error.get("type"))}
    if markers & _RATE_LIMIT_MARKERS:
        return ErrorType.RATE_LIMIT
    return ErrorType.API_ERROR


def _collect_citations(output: object) -> list[tuple[str, str]]:
    """Return distinct ``(title, url)`` pairs in first-citation order."""
    citations: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in _mappings(output, _MESSAGE_ITEM):
        for part in _mappings(item.get("content")):
            for annotation in _mappings(part.get("annotations"), _URL_CITATION):
                url = _text(annotation.get("url"))
                if not url:
                    continue
                clean_url = _strip_tracking_params(url)
                if clean_url in seen:
                    continue
                seen.add(clean_url)
                citations.append(
                    (_text(annotation.get("title")) or clean_url, clean_url)
                )
    return citations


def _is_tracking_pair(pair: str) -> bool:
    """Return whether a raw ``name=value`` pair is a tracking parameter."""
    name, _, _value = pair.partition("=")
    return unquote_plus(name).startswith(_TRACKING_PREFIX)


def _strip_tracking_params(url: str) -> str:
    """Remove ``utm_*`` parameters, leaving every other byte untouched.

    The surviving pairs are spliced out of the raw query rather than decoded
    and re-encoded, because a round trip through ``urlencode`` would rewrite
    neighbors it never meant to touch: ``%20`` would become ``+`` and a
    reserved character such as ``;`` would gain percent-escapes, changing a URL
    that is both the result URL and the title fallback.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.query:
        return url
    pairs = parts.query.split("&")
    kept = [pair for pair in pairs if not _is_tracking_pair(pair)]
    if len(kept) == len(pairs):
        return url
    return urlunsplit(parts._replace(query="&".join(kept)))

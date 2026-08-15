"""REST routes (/search, /fetch, /researcher) + provider resources.

All routes share the auth guard, bounded body parsing (64 KiB, enforced during
streaming for chunked bodies), and the query/URL 2000-char cap. Status codes:
503 no-providers, 502 all-failed, 413 body-too-large, 400 bad-input, 401
unauthorized. The ``/researcher`` route is GPT-Researcher custom-retriever
compatible.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from jasa.auth import is_authorized
from jasa.cache.base import CacheBackend
from jasa.search.providers.base import SearchProvider
from jasa.search.service import run_search, SearchError, SearchOptions
from omnifetch.cache import CacheBackend as SharedCacheBackend

_MAX_BODY_BYTES = 65536
_MAX_QUERY_CHARS = 2000
_MAX_URL_CHARS = 2000
_DEFAULT_SEARCH_COUNT = 20
_HTTP_UNAUTHORIZED = 401
_HTTP_BAD_REQUEST = 400
_HTTP_PAYLOAD_TOO_LARGE = 413
_HTTP_BAD_GATEWAY = 502
_HTTP_SERVICE_UNAVAILABLE = 503


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {"error": "unauthorized"}, status_code=_HTTP_UNAUTHORIZED
    )


def _too_large() -> JSONResponse:
    return JSONResponse(
        {"error": "request body too large"}, status_code=_HTTP_PAYLOAD_TOO_LARGE
    )


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=_HTTP_BAD_REQUEST)


async def _read_capped_body(request: Request) -> bytes | JSONResponse:
    """Read up to 64 KiB from the stream; return body or a 413 response."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_BODY_BYTES:
            return _too_large()
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_json(request: Request) -> dict[str, Any] | JSONResponse:
    """Read and parse a JSON object body with the 64 KiB cap."""
    body = await _read_capped_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _bad_request("request body must be valid JSON")
    if not isinstance(payload, dict):
        return _bad_request("request body must be a JSON object")
    return payload


def _clamp_count(raw: Any, default: int = _DEFAULT_SEARCH_COUNT) -> int:
    """Clamp a count to 0-100; invalid values use the bounded default."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(0, min(100, value))


def _fetch_inputs(
    payload: dict[str, Any],
) -> tuple[str, str | list[str] | None] | JSONResponse:
    """Validate and normalize REST fetch inputs."""
    url = payload.get("url")
    if not isinstance(url, str) or not url.strip() or len(url) > _MAX_URL_CHARS:
        return _bad_request("url is required (1-2000 chars)")
    skip_providers = payload.get("skip_providers")
    if not (
        skip_providers is None
        or isinstance(skip_providers, str)
        or (
            isinstance(skip_providers, list)
            and all(isinstance(name, str) for name in skip_providers)
        )
    ):
        return _bad_request(
            "skip_providers must be a string or list of strings"
        )
    return url.strip(), skip_providers


def register_rest_routes(
    server: FastMCP,
    providers: Mapping[str, SearchProvider],
    cache: CacheBackend,
    engine: object,
) -> None:
    """Register /search, /fetch, and /researcher REST routes."""

    @server.custom_route("/search", methods=["POST"], include_in_schema=False)
    async def rest_search(request: Request) -> JSONResponse:
        if not is_authorized(request):
            return _unauthorized()
        payload = await _read_json(request)
        if isinstance(payload, JSONResponse):
            return payload
        query = payload.get("query")
        if (
            not isinstance(query, str)
            or not query
            or len(query) > _MAX_QUERY_CHARS
        ):
            return _bad_request("query is required (1-2000 chars)")
        count = _clamp_count(payload.get("count", _DEFAULT_SEARCH_COUNT))
        raw = bool(payload.get("raw", False))
        options = SearchOptions(skip_quality_filter=raw, timeout_ms=30000)
        try:
            outcome = await run_search(providers, cache, query, options=options)
        except SearchError as error:
            status = (
                _HTTP_SERVICE_UNAVAILABLE
                if error.kind == "no_providers"
                else _HTTP_BAD_GATEWAY
            )
            return JSONResponse({"error": str(error)}, status_code=status)
        results = (
            outcome.web_results if count == 0 else outcome.web_results[:count]
        )
        return JSONResponse(
            [
                {
                    "link": r.url,
                    "title": r.title,
                    "snippet": " ".join(r.snippets),
                }
                for r in results
            ]
        )

    @server.custom_route("/fetch", methods=["POST"], include_in_schema=False)
    async def rest_fetch(request: Request) -> JSONResponse:
        if not is_authorized(request):
            return _unauthorized()
        payload = await _read_json(request)
        if isinstance(payload, JSONResponse):
            return payload
        fetch_inputs = _fetch_inputs(payload)
        if isinstance(fetch_inputs, JSONResponse):
            return fetch_inputs
        url, skip_providers = fetch_inputs
        from omnifetch.fetch.shared.types import ProviderError
        from omnifetch.tools.fetch import execute_web_fetch

        try:
            async with asyncio.timeout(30):
                result = await execute_web_fetch(
                    engine,
                    url,
                    skip_providers=skip_providers,
                )
        except TimeoutError:
            return JSONResponse({"error": "fetch timed out"}, status_code=504)
        except ProviderError as error:
            error_type = error.error_type
            if error_type == "INVALID_INPUT":
                status = _HTTP_BAD_REQUEST
            elif error_type == "RATE_LIMIT":
                status = 429
            elif error_type == "NOT_FOUND":
                status = 404
            else:
                status = _HTTP_BAD_GATEWAY
            return JSONResponse({"error": str(error)}, status_code=status)
        return JSONResponse(result.model_dump(mode="json"))

    @server.custom_route(
        "/researcher", methods=["GET", "POST"], include_in_schema=False
    )
    async def rest_researcher(request: Request) -> JSONResponse:
        if not is_authorized(request):
            return _unauthorized()
        if request.method == "GET":
            query: str | None = request.query_params.get("query")
        else:
            payload = await _read_json(request)
            if isinstance(payload, JSONResponse):
                return payload
            query = payload.get("query")
        if (
            not isinstance(query, str)
            or not query
            or len(query) > _MAX_QUERY_CHARS
        ):
            return _bad_request("query is required (1-2000 chars)")
        options = SearchOptions(timeout_ms=30000)
        try:
            outcome = await run_search(providers, cache, query, options=options)
        except SearchError as error:
            status = (
                _HTTP_SERVICE_UNAVAILABLE
                if error.kind == "no_providers"
                else _HTTP_BAD_GATEWAY
            )
            return JSONResponse({"error": str(error)}, status_code=status)
        entries = [
            {"href": r.url, "body": "\n".join(r.snippets)}
            for r in outcome.web_results[:10]
            if r.snippets
        ]
        return JSONResponse(entries)


def register_provider_resources(
    server: FastMCP,
    search_names: list[str],
    fetch_names: list[str],
    config: Any,
    cache: SharedCacheBackend,
) -> None:
    """Register the provider-status and provider-info MCP resources."""

    @server.resource("jasa://providers/status")
    async def provider_status() -> str:
        import os

        from jasa.server import build_health_payload, grounding_enabled

        cache_backend = config.cache.backend
        grounding_mode = config.grounding.mode
        payload = build_health_payload(
            search_providers=search_names,
            fetch_providers=fetch_names,
            grounding_on=grounding_enabled(
                grounding_mode, os.getenv("CEREBRAS_API_KEY")
            ),
            cache_backend=cache_backend,
            cache_ready=await cache.is_ready(),
        )
        return json.dumps(payload)

    @server.resource("jasa://providers/{provider}/info")
    async def provider_info(provider: str) -> str:
        all_names = search_names + fetch_names
        if provider not in all_names:
            raise ValueError(f"unknown provider: {provider}")
        family = "search" if provider in search_names else "fetch"
        return json.dumps(
            {
                "name": provider,
                "status": "operational",
                "family": family,
                "capabilities": ["web_search"]
                if family == "search"
                else ["fetch"],
            }
        )

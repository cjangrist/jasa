"""Request tracing with non-blocking, S3-compatible background delivery.

Search tasks collect one Omnisearch-compatible document in memory. Completion
takes a bounded snapshot and calls ``Queue.put_nowait``; JSON encoding,
recursive redaction, AWS signing, DNS, TLS, and object upload all execute in a
worker thread.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from httpx._decoders import (
    ContentDecoder,
    IdentityDecoder,
    MultiDecoder,
    SUPPORTED_DECODERS,
)

_REDACTED = "[REDACTED]"
_HTTP_CALL_EXTENSION = "jasa_trace_http_call"
_HTTP_TRACE_EXTENSION = "jasa_trace_owner"
_MAX_CAPTURED_RESPONSE_BYTES = 5 * 1024 * 1024
_MAX_CAPTURED_TRACE_RESPONSE_BYTES = 8 * 1024 * 1024
_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_WORD_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NAME_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")
_SENSITIVE_NAMES = frozenset(
    {
        "aws_access_key_id",
        "authorization",
        "cookie",
        "google_access_id",
        "key",
        "password",
        "proxy_authorization",
        "secret",
        "set_cookie",
        "sig",
        "signature",
        "token",
        "x_api_key",
        "x_amz_credential",
        "x_amz_security_token",
        "x_amz_signature",
        "x_goog_credential",
        "x_goog_security_token",
        "x_goog_signature",
        "x-subscription-token",
    }
)
_SENSITIVE_SUFFIXES = (
    "-key-id",
    "-key",
    "_key_id",
    "_key",
    "-password",
    "_password",
    "-secret",
    "_secret",
    "-token",
    "_token",
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sensitive_name(name: object) -> bool:
    with_acronym_boundaries = _ACRONYM_BOUNDARY.sub(r"\1_\2", str(name))
    with_word_boundaries = _WORD_BOUNDARY.sub(r"\1_\2", with_acronym_boundaries)
    normalized = _NAME_SEPARATOR.sub("_", with_word_boundaries).lower()
    return normalized in _SENSITIVE_NAMES or normalized.endswith(
        _SENSITIVE_SUFFIXES
    )


def _redact(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _sensitive_name(key) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [_redact(item) for item in value]
    if isinstance(value, str) and _is_http_url(value):
        return _sanitize_url(value)
    return value


def _is_http_url(value: str) -> bool:
    """Recognize HTTP URL schemes without trusting caller casing."""
    return value.lstrip()[:8].lower().startswith(("http://", "https://"))


def _sanitize_url(raw_url: str) -> str:
    try:
        parts = urlsplit(raw_url.strip())
        hostname = parts.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        port = f":{parts.port}" if parts.port is not None else ""
        userinfo = (
            f"{_REDACTED}@"
            if parts.username is not None or parts.password is not None
            else ""
        )
        query = urlencode(
            [
                (key, _REDACTED if _sensitive_name(key) else value)
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
            ]
        )
    except ValueError:
        return _REDACTED
    return urlunsplit(
        (
            parts.scheme,
            f"{userinfo}{hostname}{port}",
            parts.path,
            query,
            "",
        )
    )


def _decode_body(body: bytes | None) -> object:
    if not body:
        return None
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


@dataclass(slots=True)
class HttpCallRecord:
    """One provider HTTP round trip, buffered until trace serialization."""

    timestamp: datetime
    started_monotonic: float
    method: str
    url: str
    request_headers: dict[str, str]
    request_body: bytes | None
    response_status: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: bytes | None = None
    response_size_bytes: int = 0
    response_body_truncated: bool = False
    duration_ms: int = 0
    error: str | None = None


class _TraceCaptureStream(httpx.AsyncByteStream):
    """Tee an async response stream into a capped trace buffer."""

    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        call: HttpCallRecord,
        trace: SearchTrace,
        decoder: ContentDecoder,
    ) -> None:
        self._stream = stream
        self._call = call
        self._trace = trace
        self._decoder = decoder
        self._chunks: list[bytes] = []
        self._finished = False

    def _capture(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._call.response_size_bytes += len(chunk)
        if self._call.response_body_truncated:
            return
        if self._call.response_size_bytes > _MAX_CAPTURED_RESPONSE_BYTES:
            self._discard_capture()
            return
        if not self._trace.reserve_response_capture(len(chunk)):
            self._discard_capture()
            return
        self._chunks.append(chunk)

    def _capture_decoded(self, raw_chunk: bytes | None) -> None:
        try:
            decoded = (
                self._decoder.flush()
                if raw_chunk is None
                else self._decoder.decode(raw_chunk)
            )
        except Exception:
            self._discard_capture()
            return
        self._capture(decoded)

    def _discard_capture(self) -> None:
        self._trace.release_response_capture(
            sum(len(chunk) for chunk in self._chunks)
        )
        self._chunks.clear()
        self._call.response_body_truncated = True

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._chunks or not self._call.response_body_truncated:
            self._call.response_body = b"".join(self._chunks)
        self._call.duration_ms = int(
            (time.monotonic() - self._call.started_monotonic) * 1000
        )

    async def __aiter__(self) -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in self._stream:
                self._capture_decoded(chunk)
                yield chunk
            completed = True
        except BaseException as error:
            self._call.error = type(error).__name__
            raise
        finally:
            if completed:
                self._capture_decoded(None)
            else:
                self._call.response_body_truncated = True
            self._finish()

    async def aclose(self) -> None:
        if not self._finished:
            self._call.response_body_truncated = True
        try:
            await self._stream.aclose()
        finally:
            self._finish()


@dataclass(slots=True)
class ProviderRecord:
    """Provider input, normalized output, failure, and outbound calls."""

    started_at: datetime
    input: object
    duration_ms: int = 0
    success: bool = False
    output: object = None
    error: str | None = None
    http_calls: list[HttpCallRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OrchestratorDecision:
    """Timestamped search-orchestrator decision."""

    timestamp: datetime
    action: str
    details: object


@dataclass(slots=True)
class SearchTrace:
    """Mutable request-local trace populated by one search task tree."""

    query: str
    active_providers: list[str]
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=_utc_now)
    parent_trace_id: str | None = None
    cache_hit: bool = False
    providers: dict[str, ProviderRecord] = field(default_factory=dict)
    decisions: list[OrchestratorDecision] = field(default_factory=list)
    captured_response_bytes: int = 0
    orchestrator_strategy: str = "parallel_fanout"

    def reserve_response_capture(self, size_bytes: int) -> bool:
        """Reserve bounded response-body memory for this complete trace."""
        if (
            self.captured_response_bytes + size_bytes
            > _MAX_CAPTURED_TRACE_RESPONSE_BYTES
        ):
            return False
        self.captured_response_bytes += size_bytes
        return True

    def release_response_capture(self, size_bytes: int) -> None:
        """Release bytes discarded from an incomplete captured response."""
        self.captured_response_bytes -= size_bytes

    def record_decision(self, action: str, details: object) -> None:
        """Append an orchestrator action using wall-clock UTC."""
        self.decisions.append(OrchestratorDecision(_utc_now(), action, details))

    def record_provider_start(
        self, provider: str, provider_input: object
    ) -> None:
        """Create or replace one provider attempt record."""
        self.providers[provider] = ProviderRecord(_utc_now(), provider_input)

    def record_provider_complete(
        self, provider: str, output: object, duration_ms: int
    ) -> None:
        """Mark one provider successful with its normalized output."""
        record = self.providers[provider]
        record.success = True
        record.output = output
        record.duration_ms = duration_ms

    def record_provider_error(
        self, provider: str, error: str, duration_ms: int
    ) -> None:
        """Mark one provider failed and close its pending HTTP calls."""
        record = self.providers.get(provider)
        if record is None:
            record = ProviderRecord(_utc_now(), None)
            self.providers[provider] = record
        record.error = error
        record.duration_ms = duration_ms
        record.success = False
        for call in record.http_calls:
            if call.response_status == 0 and call.error is None:
                call.duration_ms = int(
                    (time.monotonic() - call.started_monotonic) * 1000
                )
                call.error = error


@dataclass(slots=True)
class FetchTrace:
    """Mutable public-fetch trace completed at the transport boundary."""

    request_environment: object
    active_providers: tuple[str, ...]
    transport: str
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=_utc_now)
    parent_trace_id: str | None = None
    orchestrator_strategy: str = "provider_waterfall"


def _fetch_result_payload(result: object) -> object:
    """Return the structured tool result without copying response content."""
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    return {
        "content": getattr(result, "content", None),
        "is_error": bool(getattr(result, "is_error", False)),
    }


class FetchTraceMiddleware(Middleware):
    """Archive mounted ``web_fetch`` MCP calls without delaying callers."""

    def __init__(
        self,
        sink: Any,
        active_providers: Sequence[str],
    ) -> None:
        """Snapshot the sink and provider catalog for later MCP calls."""
        self._sink = sink
        self._active_providers = tuple(active_providers)

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        """Trace only the mounted public fetch tool."""
        if getattr(context.message, "name", None) != "web_fetch":
            return await call_next(context)
        arguments = getattr(context.message, "arguments", None)
        parent = active_trace()
        trace = FetchTrace(
            request_environment=arguments,
            active_providers=self._active_providers,
            transport="mcp",
            parent_trace_id=parent.trace_id if parent is not None else None,
        )
        try:
            result = await call_next(context)
        except BaseException as error:
            self._sink.submit_fetch(trace, None, error=type(error).__name__)
            raise
        self._sink.submit_fetch(trace, _fetch_result_payload(result))
        return result


_ACTIVE_TRACE: ContextVar[SearchTrace | None] = ContextVar(
    "jasa_active_search_trace", default=None
)
_ACTIVE_PROVIDER: ContextVar[str | None] = ContextVar(
    "jasa_active_search_provider", default=None
)


def active_trace() -> SearchTrace | None:
    """Return the trace scoped to the current task tree."""
    return _ACTIVE_TRACE.get()


def activate_trace(trace: SearchTrace) -> Token[SearchTrace | None]:
    """Scope one trace and return the token required to restore context."""
    return _ACTIVE_TRACE.set(trace)


def reset_trace(token: Token[SearchTrace | None]) -> None:
    """Restore the preceding request-trace context."""
    _ACTIVE_TRACE.reset(token)


def activate_provider(provider: str) -> Token[str | None]:
    """Scope one provider name for shared-client HTTP hooks."""
    return _ACTIVE_PROVIDER.set(provider)


def reset_provider(token: Token[str | None]) -> None:
    """Restore the preceding provider context."""
    _ACTIVE_PROVIDER.reset(token)


async def record_http_request(request: httpx.Request) -> None:
    """Begin an HTTP call only inside an active search-provider context."""
    trace = active_trace()
    provider = _ACTIVE_PROVIDER.get()
    if trace is None or provider is None:
        return
    try:
        request_body = request.content
    except httpx.RequestNotRead:
        request_body = None
    call = HttpCallRecord(
        timestamp=_utc_now(),
        started_monotonic=time.monotonic(),
        method=request.method,
        url=str(request.url),
        request_headers=dict(request.headers),
        request_body=request_body,
    )
    trace.providers[provider].http_calls.append(call)
    request.extensions[_HTTP_CALL_EXTENSION] = call
    request.extensions[_HTTP_TRACE_EXTENSION] = trace


def _trace_content_decoder(headers: httpx.Headers) -> ContentDecoder:
    """Build an independent decoder from HTTPX's pinned decoder registry."""
    encoding_names = headers.get_list("content-encoding", split_commas=True)
    decoder_classes = [
        SUPPORTED_DECODERS[normalized]
        for name in encoding_names
        if (normalized := name.strip().lower()) in SUPPORTED_DECODERS
    ]
    return (
        MultiDecoder([decoder_class() for decoder_class in decoder_classes])
        if decoder_classes
        else IdentityDecoder()
    )


def _capture_preloaded_response(
    response: httpx.Response, call: HttpCallRecord, trace: SearchTrace
) -> None:
    """Capture HTTPX's already-decoded response through the same caps."""
    body = response.content
    call.response_size_bytes = len(body)
    body_exceeds_call_cap = len(body) > _MAX_CAPTURED_RESPONSE_BYTES
    if body_exceeds_call_cap or not trace.reserve_response_capture(len(body)):
        call.response_body_truncated = True
        return
    call.response_body = body


async def record_http_response(response: httpx.Response) -> None:
    """Complete a buffered HTTP record while leaving response bytes reusable."""
    call = response.request.extensions.get(_HTTP_CALL_EXTENSION)
    trace = response.request.extensions.get(_HTTP_TRACE_EXTENSION)
    if not isinstance(call, HttpCallRecord) or not isinstance(
        trace, SearchTrace
    ):
        return
    call.response_status = response.status_code
    call.response_headers = dict(response.headers)
    call.duration_ms = int((time.monotonic() - call.started_monotonic) * 1000)
    if response.is_stream_consumed:
        _capture_preloaded_response(response, call, trace)
        return
    response.stream = _TraceCaptureStream(
        cast(httpx.AsyncByteStream, response.stream),
        call,
        trace,
        _trace_content_decoder(response.headers),
    )

"""Request tracing with non-blocking, S3-compatible background delivery.

Search tasks collect one Omnisearch-compatible document in memory. Completion
only performs ``Queue.put_nowait``; JSON encoding, recursive redaction, AWS
signing, DNS, TLS, and object upload all execute in a worker thread.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, UTC
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx

_REDACTED = "[REDACTED]"
_HTTP_CALL_EXTENSION = "jasa_trace_http_call"
_SENSITIVE_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "key",
        "password",
        "proxy-authorization",
        "secret",
        "set-cookie",
        "token",
        "x-api-key",
        "x-subscription-token",
    }
)
_SENSITIVE_SUFFIXES = (
    "-key",
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
    normalized = str(name).lower()
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
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return _sanitize_url(value)
    return value


def _sanitize_url(raw_url: str) -> str:
    try:
        parts = urlsplit(raw_url)
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
        return raw_url
    return urlunsplit(
        (
            parts.scheme,
            f"{userinfo}{hostname}{port}",
            parts.path,
            query,
            parts.fragment,
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
    duration_ms: int = 0
    error: str | None = None


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


async def record_http_response(response: httpx.Response) -> None:
    """Complete a buffered HTTP record while leaving response bytes reusable."""
    call = response.request.extensions.get(_HTTP_CALL_EXTENSION)
    if not isinstance(call, HttpCallRecord):
        return
    await response.aread()
    call.response_status = response.status_code
    call.response_headers = dict(response.headers)
    call.response_body = response.content
    call.duration_ms = int((time.monotonic() - call.started_monotonic) * 1000)

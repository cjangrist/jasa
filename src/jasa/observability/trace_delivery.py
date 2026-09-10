"""Serialize redacted traces and deliver them to S3-compatible storage."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast
from urllib.parse import parse_qsl, unquote, urlsplit

from jasa.config import TraceSettings
from jasa.logging import get_logger
from jasa.observability.traces import (
    _decode_body,
    _iso_timestamp,
    _redact,
    _sensitive_name,
    _utc_now,
    HttpCallRecord,
    ProviderRecord,
    SearchTrace,
)

_LOGGER = get_logger("observability.trace_delivery")
_SENTINEL = object()
_MINIMUM_SECRET_LENGTH = 4
_MAX_QUEUED_CAPTURE_BYTES = 32 * 1024 * 1024


def validate_trace_settings(settings: TraceSettings) -> None:
    """Reject incomplete or unsafe enabled trace destinations."""
    if not settings.enabled:
        return
    required = {
        "JASA_TRACE_S3_ENDPOINT": settings.endpoint,
        "JASA_TRACE_S3_BUCKET": settings.bucket,
        "JASA_TRACE_S3_ACCESS_KEY_ID": settings.access_key_id,
        "JASA_TRACE_S3_SECRET_ACCESS_KEY": settings.secret_access_key,
    }
    missing = [name for name, value in required.items() if not value.strip()]
    if missing:
        raise ValueError("S3 request tracing requires: " + ", ".join(missing))
    try:
        endpoint = urlsplit(settings.endpoint)
        _ = endpoint.port
    except ValueError as error:
        raise ValueError(
            "JASA_TRACE_S3_ENDPOINT must be an HTTPS URL without credentials."
        ) from error
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
    ):
        raise ValueError(
            "JASA_TRACE_S3_ENDPOINT must be an HTTPS URL without credentials."
        )


@dataclass(frozen=True, slots=True)
class TraceEnvelope:
    """Completed trace waiting for off-loop serialization and upload."""

    trace: SearchTrace
    final_result: object
    completed_at: datetime
    captured_response_bytes: int = 0


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
            if item.name != "deadline_exceeded"
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [_jsonable(item) for item in value]
    return value


def _sensitive_values(value: object) -> set[str]:
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return {secret for item in value for secret in _sensitive_values(item)}
    if not isinstance(value, Mapping):
        return set()
    direct = {
        secret
        for key, item in value.items()
        for secret in _sensitive_string_values(key, item)
    }
    nested = {
        secret for item in value.values() for secret in _sensitive_values(item)
    }
    return direct | nested


def _sensitive_string_values(name: object, value: object) -> set[str]:
    """Extract complete and structured credentials from sensitive strings."""
    if not _sensitive_name(name) or not isinstance(value, str):
        return set()
    candidates = {value}
    normalized_name = "".join(
        character for character in str(name).lower() if character.isalnum()
    )
    if normalized_name in {"authorization", "proxyauthorization"}:
        scheme_and_value = value.split(maxsplit=1)
        if scheme_and_value[1:]:
            payload = scheme_and_value[-1].strip()
            candidates.add(payload)
            for parameter in payload.split(","):
                _, separator, parameter_value = parameter.partition("=")
                if separator:
                    unquoted = parameter_value.strip().strip("\"'")
                    candidates.update({unquoted, unquoted.partition("/")[0]})
    variants = {
        variant
        for candidate in candidates
        for variant in (
            candidate,
            candidate.strip().strip('"').strip("'"),
        )
    }
    return {
        candidate
        for candidate in variants
        if len(candidate) >= _MINIMUM_SECRET_LENGTH
    }


def _url_sensitive_values(raw_url: str) -> set[str]:
    """Return raw and decoded URL userinfo values that need global scrubbing."""
    try:
        parts = urlsplit(raw_url)
    except ValueError:
        return set()
    values = {
        value
        for item in (parts.username, parts.password)
        if item is not None
        for value in (item, unquote(item))
        if len(value) >= _MINIMUM_SECRET_LENGTH
    }
    return values


def _scrub_text(value: str, secrets: set[str]) -> str:
    for secret in sorted(secrets, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    return value


def _scrub_strings(value: object, secrets: set[str]) -> object:
    if isinstance(value, str):
        return _scrub_text(value, secrets)
    if isinstance(value, Mapping):
        return {
            _scrub_text(key, secrets) if isinstance(key, str) else key: (
                _scrub_strings(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [_scrub_strings(item, secrets) for item in value]
    return value


def _http_document(call: HttpCallRecord) -> dict[str, object]:
    document: dict[str, object] = {
        "timestamp": _iso_timestamp(call.timestamp),
        "method": call.method,
        "url": _redact(call.url),
        "request_headers": _redact(call.request_headers),
        "request_body": _redact(_decode_body(call.request_body)),
        "response_status": call.response_status,
        "response_headers": _redact(call.response_headers),
        "response_body": _redact(_decode_body(call.response_body)),
        "response_size_bytes": call.response_size_bytes
        or len(call.response_body or b""),
        "response_body_truncated": call.response_body_truncated,
        "duration_ms": call.duration_ms,
    }
    if call.error is not None:
        document["error"] = call.error
    return document


def _provider_document(record: ProviderRecord) -> dict[str, object]:
    document: dict[str, object] = {
        "started_at": _iso_timestamp(record.started_at),
        "duration_ms": record.duration_ms,
        "success": record.success,
        "input": _redact(_jsonable(record.input)),
        "output": _redact(_jsonable(record.output)),
        "http_calls": [_http_document(call) for call in record.http_calls],
    }
    if record.error is not None:
        document["error"] = record.error
    return document


def _http_call_secrets(call: HttpCallRecord) -> set[str]:
    structured_secrets = {
        secret
        for value in (
            call.request_headers,
            _decode_body(call.request_body),
            dict(parse_qsl(urlsplit(call.url).query, keep_blank_values=True)),
            call.response_headers,
            _decode_body(call.response_body),
        )
        for secret in _sensitive_values(value)
    }
    return structured_secrets | _url_sensitive_values(call.url)


def _provider_secrets(record: ProviderRecord) -> set[str]:
    direct_secrets = {
        secret
        for value in (_jsonable(record.input), _jsonable(record.output))
        for secret in _sensitive_values(value)
    }
    call_secrets = {
        secret
        for call in record.http_calls
        for secret in _http_call_secrets(call)
    }
    return direct_secrets | call_secrets


def _trace_secrets(envelope: TraceEnvelope) -> set[str]:
    trace = envelope.trace
    provider_secrets = {
        secret
        for record in trace.providers.values()
        for secret in _provider_secrets(record)
    }
    result_secrets = _sensitive_values(_jsonable(envelope.final_result))
    decision_secrets = {
        secret
        for decision in trace.decisions
        for secret in _sensitive_values(_jsonable(decision.details))
    }
    return provider_secrets | result_secrets | decision_secrets


def _trace_document(
    envelope: TraceEnvelope, configured_secrets: Collection[str] = ()
) -> dict[str, object]:
    trace = envelope.trace
    providers_hit = list(trace.providers)
    succeeded = [
        name for name in providers_hit if trace.providers[name].success
    ]
    failures = [
        {
            "provider": name,
            "error": trace.providers[name].error or "unknown",
            "duration_ms": trace.providers[name].duration_ms,
        }
        for name in providers_hit
        if not trace.providers[name].success
    ]
    document = {
        "trace_id": trace.trace_id,
        "tool": "web_search",
        "parent_trace_id": trace.parent_trace_id,
        "started_at": _iso_timestamp(trace.started_at),
        "completed_at": _iso_timestamp(envelope.completed_at),
        "total_duration_ms": int(
            (envelope.completed_at - trace.started_at).total_seconds() * 1000
        ),
        "cache_hit": trace.cache_hit,
        "request_environment": {"query": trace.query},
        "orchestrator": {
            "strategy": trace.orchestrator_strategy,
            "active_providers": trace.active_providers,
            "decisions": [
                {
                    "timestamp": _iso_timestamp(item.timestamp),
                    "action": item.action,
                    "details": _redact(_jsonable(item.details)),
                }
                for item in trace.decisions
            ],
        },
        "providers_hit": providers_hit,
        "providers_succeeded": succeeded,
        "providers_failed": failures,
        "providers": {
            name: _provider_document(record)
            for name, record in trace.providers.items()
        },
        "final_result": _redact(_jsonable(envelope.final_result)),
    }
    return cast(
        dict[str, object],
        _scrub_strings(
            document, _trace_secrets(envelope) | set(configured_secrets)
        ),
    )


def _object_key(settings: TraceSettings, envelope: TraceEnvelope) -> str:
    completed_at = envelope.completed_at
    prefix = settings.prefix.strip("/")
    suffix = (
        f"tool=web_search/date={completed_at:%Y-%m-%d}/"
        f"hour={completed_at:%H}/trace_id={envelope.trace.trace_id}.json"
    )
    return f"{prefix}/{suffix}" if prefix else suffix


def _build_s3_client(settings: TraceSettings) -> object:
    validate_trace_settings(settings)
    import boto3
    from botocore.config import Config

    addressing_style = "path" if settings.force_path_style else "auto"
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint,
        region_name=settings.region,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        config=Config(
            connect_timeout=5,
            read_timeout=15,
            retries={"max_attempts": 3, "mode": "standard"},
            s3={"addressing_style": addressing_style},
        ),
    )


class S3TraceSink:
    """Bounded queue whose request-path operation is only ``put_nowait``."""

    def __init__(
        self,
        settings: TraceSettings,
        sync_uploader: Callable[[TraceEnvelope], None] | None = None,
        configured_secrets: Collection[str] = (),
    ) -> None:
        """Create an inactive sink without starting network work."""
        self.settings = settings
        self._queue: asyncio.Queue[TraceEnvelope | object] = asyncio.Queue(
            maxsize=settings.queue_capacity
        )
        self._sync_uploader = sync_uploader
        self._client: object | None = None
        self._worker: asyncio.Task[None] | None = None
        self._dropped_submissions = 0
        self._accepted_capture_bytes = 0
        self._configured_secrets = frozenset(configured_secrets)

    def start(self) -> None:
        """Create the lightweight coordinator on the active event loop."""
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._run(), name="jasa-s3-trace-uploader"
            )

    def submit(self, trace: SearchTrace, final_result: object) -> bool:
        """Enqueue a stable snapshot without encoding or network activity."""
        if self._worker is None:
            return False
        captured_response_bytes = trace.captured_response_bytes
        if (
            self._queue.full()
            or captured_response_bytes
            > _MAX_QUEUED_CAPTURE_BYTES - self._accepted_capture_bytes
        ):
            self._dropped_submissions += 1
            return False
        try:
            envelope = copy.deepcopy(
                TraceEnvelope(
                    trace,
                    final_result,
                    _utc_now(),
                    captured_response_bytes,
                )
            )
            self._queue.put_nowait(envelope)
        except Exception:
            self._dropped_submissions += 1
            return False
        self._accepted_capture_bytes += captured_response_bytes
        return True

    async def close(self) -> None:
        """Drain accepted traces during orderly process shutdown."""
        worker = self._worker
        if worker is None:
            return
        await self._queue.put(_SENTINEL)
        await worker
        self._worker = None

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            accepted_capture_bytes = 0
            try:
                if item is _SENTINEL:
                    return
                envelope = cast(TraceEnvelope, item)
                accepted_capture_bytes = envelope.captured_response_bytes
                await asyncio.to_thread(self._upload_sync, envelope)
            except Exception as error:
                await asyncio.to_thread(
                    _LOGGER.warning,
                    "S3 trace upload failed error_type=%s",
                    type(error).__name__,
                )
            finally:
                self._queue.task_done()
                self._accepted_capture_bytes -= accepted_capture_bytes
                await self._report_dropped_submissions()

    async def _report_dropped_submissions(self) -> None:
        dropped_submissions = self._dropped_submissions
        self._dropped_submissions = 0
        if dropped_submissions:
            await asyncio.to_thread(
                _LOGGER.warning,
                "Trace queue saturation dropped_count=%s",
                dropped_submissions,
            )

    def _upload_sync(self, envelope: TraceEnvelope) -> None:
        if self._sync_uploader is not None:
            self._sync_uploader(envelope)
            return
        if self._client is None:
            self._client = _build_s3_client(self.settings)
        body = json.dumps(
            _trace_document(envelope, self._configured_secrets),
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        client = cast(Any, self._client)
        client.put_object(
            Bucket=self.settings.bucket,
            Key=_object_key(self.settings, envelope),
            Body=body,
            ContentType="application/json",
        )
        _LOGGER.debug("S3 trace uploaded trace_id=%s", envelope.trace.trace_id)


def build_trace_sink(
    settings: TraceSettings,
    secret_environment: Mapping[str, str] | None = None,
) -> S3TraceSink | None:
    """Return an enabled sink after bootstrap validated its settings."""
    validate_trace_settings(settings)
    if not settings.enabled:
        return None
    configured_secrets = _sensitive_values(secret_environment or {})
    return S3TraceSink(settings, configured_secrets=configured_secrets)

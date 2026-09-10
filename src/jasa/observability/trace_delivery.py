"""Serialize redacted traces and deliver them to S3-compatible storage."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast
from urllib.parse import parse_qsl, urlsplit

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


@dataclass(frozen=True, slots=True)
class TraceEnvelope:
    """Completed trace waiting for off-loop serialization and upload."""

    trace: SearchTrace
    final_result: object
    completed_at: datetime


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
    if not isinstance(value, Mapping):
        return set()
    direct = {
        str(item)
        for key, item in value.items()
        if _sensitive_name(key)
        and isinstance(item, str)
        and len(item) >= _MINIMUM_SECRET_LENGTH
    }
    nested = {
        secret
        for key, item in value.items()
        if not _sensitive_name(key)
        for secret in _sensitive_values(item)
    }
    return direct | nested


def _scrub_strings(value: object, secrets: set[str]) -> object:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, Mapping):
        return {
            key: _scrub_strings(item, secrets) for key, item in value.items()
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
        "response_size_bytes": len(call.response_body or b""),
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


def _trace_secrets(trace: SearchTrace) -> set[str]:
    return {
        secret
        for record in trace.providers.values()
        for call in record.http_calls
        for value in (
            call.request_headers,
            _decode_body(call.request_body),
            dict(parse_qsl(urlsplit(call.url).query, keep_blank_values=True)),
        )
        for secret in _sensitive_values(value)
    }


def _trace_document(envelope: TraceEnvelope) -> dict[str, object]:
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
            "strategy": "parallel_fanout",
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
        dict[str, object], _scrub_strings(document, _trace_secrets(trace))
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
    ) -> None:
        """Create an inactive sink without starting network work."""
        self.settings = settings
        self._queue: asyncio.Queue[TraceEnvelope | object] = asyncio.Queue(
            maxsize=settings.queue_capacity
        )
        self._sync_uploader = sync_uploader
        self._client: object | None = None
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Create the lightweight coordinator on the active event loop."""
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._run(), name="jasa-s3-trace-uploader"
            )

    def submit(self, trace: SearchTrace, final_result: object) -> bool:
        """Enqueue without awaiting, encoding, signing, or network activity."""
        if self._worker is None:
            return False
        try:
            self._queue.put_nowait(
                TraceEnvelope(trace, final_result, _utc_now())
            )
        except asyncio.QueueFull:
            _LOGGER.warning(
                "Trace queue full; dropping trace_id=%s", trace.trace_id
            )
            return False
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
            try:
                if item is _SENTINEL:
                    return
                await asyncio.to_thread(
                    self._upload_sync, cast(TraceEnvelope, item)
                )
            except Exception as error:
                _LOGGER.warning(
                    "S3 trace upload failed error_type=%s", type(error).__name__
                )
            finally:
                self._queue.task_done()

    def _upload_sync(self, envelope: TraceEnvelope) -> None:
        if self._sync_uploader is not None:
            self._sync_uploader(envelope)
            return
        if self._client is None:
            self._client = _build_s3_client(self.settings)
        body = json.dumps(
            _trace_document(envelope), indent=2, ensure_ascii=False
        ).encode("utf-8")
        client = cast(Any, self._client)
        client.put_object(
            Bucket=self.settings.bucket,
            Key=_object_key(self.settings, envelope),
            Body=body,
            ContentType="application/json",
        )
        _LOGGER.debug("S3 trace uploaded trace_id=%s", envelope.trace.trace_id)


def build_trace_sink(settings: TraceSettings) -> S3TraceSink | None:
    """Return an enabled sink after bootstrap validated its settings."""
    return S3TraceSink(settings) if settings.enabled else None

"""Request-trace capture, redaction, background delivery, and search wiring."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import cast
from unittest.mock import MagicMock

import boto3
import httpx
import pytest
from botocore.config import Config

import jasa.observability.trace_delivery as delivery_module
import jasa.search.service as service_module
from jasa.cache.memory import MemoryCache
from jasa.config import TraceSettings
from jasa.observability.trace_delivery import (
    _bounded_snapshot,
    _build_s3_client,
    _jsonable,
    _object_key,
    _prepare_trace,
    _PreparedTrace,
    _scrub_strings,
    _sensitive_string_values,
    _sensitive_values,
    _SnapshotBudget,
    _trace_document,
    _url_sensitive_values,
    build_trace_sink,
    S3TraceSink,
    TraceEnvelope,
)
from jasa.observability.traces import (
    _decode_body,
    _redact,
    _sanitize_url,
    _sensitive_name,
    activate_provider,
    activate_trace,
    active_trace,
    HttpCallRecord,
    record_http_request,
    record_http_response,
    reset_provider,
    reset_trace,
    SearchTrace,
)
from jasa.search.fanout import _FanoutKnobs, DispatchResult, ProviderSuccess
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from jasa.search.service import (
    run_search,
    SearchError,
    SearchFlightRegistry,
    SearchOptions,
)
from omnifetch.fetch.shared.types import ErrorType, ProviderError


def _settings(**overrides: object) -> TraceSettings:
    values = {
        "JASA_TRACE_S3_ENABLED": True,
        "JASA_TRACE_S3_ENDPOINT": "https://objects.example.test",
        "JASA_TRACE_S3_REGION": "auto",
        "JASA_TRACE_S3_BUCKET": "traces",
        "JASA_TRACE_S3_PREFIX": "request_traces",
        "JASA_TRACE_S3_ACCESS_KEY_ID": "access-id",
        "JASA_TRACE_S3_SECRET_ACCESS_KEY": "secret-key",
        "JASA_TRACE_S3_FORCE_PATH_STYLE": True,
        "JASA_TRACE_S3_QUEUE_CAPACITY": 2,
    }
    values.update(overrides)
    return TraceSettings.model_validate(values)


def _envelope(trace: SearchTrace | None = None) -> TraceEnvelope:
    started = datetime(2026, 9, 10, 3, 4, 5, tzinfo=UTC)
    resolved = trace or SearchTrace(
        "query", ["alpha"], trace_id="trace-1", started_at=started
    )
    return TraceEnvelope(resolved, {"ok": True}, started + timedelta(seconds=1))


class _HttpProvider(SearchProvider):
    name = "alpha"
    secret_env = "ALPHA_API_KEY"
    base_url = "https://provider.example.test"
    default_timeout_s = 1.0

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        error: Exception | None = None,
        delay: float = 0,
        gate: asyncio.Event | None = None,
        cache_allowed: bool = True,
    ) -> None:
        self.client = client
        self.error = error
        self.delay = delay
        self.gate = gate
        self.cache_allowed = cache_allowed
        self.calls = 0

    def allows_cache(
        self,
        query: str,
        *,
        reference_datetime: datetime | None = None,
    ) -> bool:
        return self.cache_allowed

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        if self.client is not None:
            response = await self.client.post(
                "/search?token=query-secret",
                headers={"Authorization": "Bearer header-secret"},
                json={"query": request.query, "api_key": "body-secret"},
            )
            assert response.json() == {"ok": True}
        return [
            SearchResult(
                "Title",
                "https://result.example.test",
                "long result snippet " * 5,
                self.name,
            )
        ]


def test_redaction_helpers_cover_nested_values_and_urls() -> None:
    assert _sensitive_name("Authorization")
    assert _sensitive_name("Proxy-Authorization")
    assert _sensitive_name("Set-Cookie")
    assert _sensitive_name("client_token")
    assert _sensitive_name("apiKey")
    assert _sensitive_name("awsAccessKeyId")
    assert _sensitive_name("accessToken")
    assert _sensitive_name("clientSecret")
    assert not _sensitive_name("monkey")
    assert _decode_body(None) is None
    assert _decode_body(b'{"ok": true}') == {"ok": True}
    assert _decode_body(b"not-json") == "not-json"
    assert _redact(
        {
            "token": "secret",
            "items": ["https://user:pass@example.test/x?api_key=secret", 3],
        }
    ) == {
        "token": "[REDACTED]",
        "items": [
            "https://[REDACTED]@example.test/x?api_key=%5BREDACTED%5D",
            3,
        ],
    }
    assert _sanitize_url("https://user:pass@[::1]:8443/x?token=value") == (
        "https://[REDACTED]@[::1]:8443/x?token=%5BREDACTED%5D"
    )
    assert (
        _sanitize_url(
            "https://example.test/x?public=yes#access_token=fragment-secret"
        )
        == "https://example.test/x?public=yes"
    )
    assert _url_sensitive_values(
        "https://example.test/#/callback?access_token=fragment-secret"
    ) == {"fragment-secret"}
    malformed = "https://username:password@example.test:invalid/x"
    sanitized = _sanitize_url(malformed)
    assert sanitized == "[REDACTED]"
    assert "username" not in sanitized
    assert "password" not in sanitized
    assert _url_sensitive_values("https://[invalid") == set()


async def test_http_hooks_scope_calls_and_leave_response_reusable() -> None:
    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {"query": "query"})
    request = httpx.Request(
        "POST",
        "https://provider.example.test/search",
        json={"query": "query"},
    )
    await record_http_request(request)
    trace_token = activate_trace(trace)
    await record_http_request(request)
    provider_token = activate_provider("alpha")
    try:
        await record_http_request(request)
        response = httpx.Response(
            200, request=request, json={"results": ["reusable"]}
        )
        await record_http_response(response)
        assert response.json() == {"results": ["reusable"]}
        assert trace.providers["alpha"].http_calls[0].response_status == 200
        unrelated = httpx.Response(
            204,
            request=httpx.Request("GET", "https://example.test"),
        )
        await record_http_response(unrelated)
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)
    assert active_trace() is None


async def test_http_response_hook_caps_trace_body_without_consuming_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jasa.observability.traces._MAX_CAPTURED_RESPONSE_BYTES", 4
    )
    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    request = httpx.Request("GET", "https://provider.example.test")

    async def response_body() -> AsyncIterator[bytes]:
        yield b"12345"
        yield b"6"

    try:
        await record_http_request(request)
        response = httpx.Response(200, request=request, content=response_body())
        await record_http_response(response)
        assert await response.aread() == b"123456"
        await response.aclose()
        preloaded_request = httpx.Request(
            "GET", "https://provider.example.test/preloaded"
        )
        await record_http_request(preloaded_request)
        preloaded_response = httpx.Response(
            200, request=preloaded_request, content=b"12345"
        )
        await record_http_response(preloaded_response)
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)
    call = trace.providers["alpha"].http_calls[0]
    assert call.response_size_bytes == 6
    assert call.response_body is None
    assert call.response_body_truncated is True
    preloaded_call = trace.providers["alpha"].http_calls[1]
    assert preloaded_call.response_size_bytes == 5
    assert preloaded_call.response_body is None
    assert preloaded_call.response_body_truncated is True


async def test_http_hook_captures_bounded_stream_and_stream_error() -> None:
    async def successful_body() -> AsyncIterator[bytes]:
        yield b"12"
        yield b"34"

    async def failing_body() -> AsyncIterator[bytes]:
        yield b"12"
        raise RuntimeError("stream failed")

    decoded_body = b'{"decoded":true}'

    async def compressed_body() -> AsyncIterator[bytes]:
        yield gzip.compress(decoded_body)

    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    try:
        responses: tuple[
            tuple[AsyncIterator[bytes], dict[str, str], bytes | None], ...
        ] = (
            (successful_body(), {}, b"1234"),
            (failing_body(), {}, None),
            (
                compressed_body(),
                {"Content-Encoding": "gzip"},
                decoded_body,
            ),
        )
        for body, headers, expected_body in responses:
            request = httpx.Request("GET", "https://provider.example.test")
            await record_http_request(request)
            response = httpx.Response(
                200, request=request, content=body, headers=headers
            )
            await record_http_response(response)
            if expected_body is None:
                with pytest.raises(RuntimeError, match="stream failed"):
                    await response.aread()
            else:
                assert await response.aread() == expected_body
            await response.aclose()
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)

    successful, failed, compressed = trace.providers["alpha"].http_calls
    assert successful.response_body == b"1234"
    assert successful.response_size_bytes == 4
    assert failed.response_body == b"12"
    assert failed.response_body_truncated is True
    assert failed.error == "RuntimeError"
    assert compressed.response_body == decoded_body
    assert compressed.response_size_bytes == len(decoded_body)
    assert compressed.response_body_truncated is False


async def test_http_hook_caps_total_decoded_trace_response_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jasa.observability.traces._MAX_CAPTURED_TRACE_RESPONSE_BYTES", 3
    )

    async def response_body() -> AsyncIterator[bytes]:
        yield b"12"
        yield b"34"

    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    request = httpx.Request("GET", "https://provider.example.test")
    try:
        await record_http_request(request)
        response = httpx.Response(200, request=request, content=response_body())
        await record_http_response(response)
        assert await response.aread() == b"1234"
        await response.aclose()
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)

    call = trace.providers["alpha"].http_calls[0]
    assert call.response_size_bytes == 4
    assert call.response_body is None
    assert call.response_body_truncated is True
    assert trace.captured_response_bytes == 0


async def test_http_hook_fails_open_when_trace_decoder_rejects_body() -> None:
    async def invalid_compressed_body() -> AsyncIterator[bytes]:
        yield b"not-gzip"

    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    try:
        streaming_request = httpx.Request(
            "GET", "https://provider.example.test/streaming"
        )
        await record_http_request(streaming_request)
        streaming_response = httpx.Response(
            200,
            request=streaming_request,
            headers={"Content-Encoding": "gzip"},
            content=invalid_compressed_body(),
        )
        await record_http_response(streaming_response)
        with pytest.raises(httpx.DecodingError):
            await streaming_response.aread()
        await streaming_response.aclose()

        preloaded_request = httpx.Request(
            "GET", "https://provider.example.test/preloaded"
        )
        await record_http_request(preloaded_request)
        preloaded_response = httpx.Response(
            200,
            request=preloaded_request,
            content=b"not-gzip",
        )
        preloaded_response.headers["Content-Encoding"] = "gzip"
        await record_http_response(preloaded_response)
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)

    streaming, preloaded = trace.providers["alpha"].http_calls
    assert streaming.response_body is None
    assert streaming.response_body_truncated is True
    assert preloaded.response_body == b"not-gzip"
    assert preloaded.response_body_truncated is False


async def test_http_response_hook_closes_unread_stream() -> None:
    async def response_body() -> AsyncIterator[bytes]:
        yield b"unread"

    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    request = httpx.Request("GET", "https://provider.example.test")
    try:
        await record_http_request(request)
        response = httpx.Response(200, request=request, content=response_body())
        await record_http_response(response)
        await response.aclose()
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)
    call = trace.providers["alpha"].http_calls[0]
    assert call.response_body is None
    assert call.response_body_truncated is True


async def test_frozen_trace_ignores_late_http_activity() -> None:
    async def successful_body() -> AsyncIterator[bytes]:
        yield b"late"

    async def failing_body() -> AsyncIterator[bytes]:
        yield b""
        raise RuntimeError("late failure")

    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    responses: list[httpx.Response] = []
    try:
        for body in (successful_body(), failing_body(), successful_body()):
            request = httpx.Request("GET", "https://provider.example.test")
            await record_http_request(request)
            response = httpx.Response(200, request=request, content=body)
            await record_http_response(response)
            responses.append(response)
        preloaded_request = httpx.Request(
            "GET", "https://provider.example.test/preloaded"
        )
        await record_http_request(preloaded_request)
        trace.freeze()
        await record_http_request(
            httpx.Request("GET", "https://provider.example.test/ignored")
        )
        await record_http_response(
            httpx.Response(200, request=preloaded_request, content=b"ignored")
        )
        assert await responses[0].aread() == b"late"
        with pytest.raises(RuntimeError, match="late failure"):
            await responses[1].aread()
        await responses[2].aclose()
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)

    calls = trace.providers["alpha"].http_calls
    assert len(calls) == 4
    assert all(call.response_body is None for call in calls)
    assert all(call.response_body_truncated for call in calls[:3])
    assert calls[-1].response_body_truncated is False
    assert all(call.error is None for call in calls)
    assert calls[-1].response_status == 0


async def test_freeze_retains_partial_open_stream_and_marks_truncated() -> None:
    release_stream = asyncio.Event()

    async def response_body() -> AsyncIterator[bytes]:
        yield b'{"token":"stream-secret"}'
        await release_stream.wait()
        yield b"late"

    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    request = httpx.Request("GET", "https://provider.example.test")
    try:
        await record_http_request(request)
        response = httpx.Response(200, request=request, content=response_body())
        await record_http_response(response)
        iterator = response.aiter_bytes()
        assert await anext(iterator) == b'{"token":"stream-secret"}'
        uploaded: list[_PreparedTrace] = []
        sink = S3TraceSink(_settings(), uploaded.append)
        sink.start()
        assert sink.submit(trace, {})
        release_stream.set()
        assert await anext(iterator) == b"late"
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        await response.aclose()
        await sink.close()
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)

    document = json.loads(uploaded[0].body)
    provider = cast(dict[str, object], document["providers"])["alpha"]
    calls = cast(dict[str, object], provider)["http_calls"]
    call = cast(list[dict[str, object]], calls)[0]
    assert call["response_body"] == {"token": "[REDACTED]"}
    assert call["response_body_truncated"] is True
    assert "stream-secret" not in uploaded[0].body.decode()


async def test_frozen_stream_snapshot_never_blocks_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    snapshot_threads: list[int] = []
    original_snapshot_http_call = delivery_module._snapshot_http_call

    def slow_snapshot_http_call(
        call: HttpCallRecord, budget: _SnapshotBudget
    ) -> HttpCallRecord:
        snapshot_threads.append(threading.get_ident())
        snapshot_started.set()
        release_snapshot.wait(timeout=2)
        return original_snapshot_http_call(call, budget)

    monkeypatch.setattr(
        delivery_module, "_snapshot_http_call", slow_snapshot_http_call
    )
    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    call = HttpCallRecord(
        trace.started_at,
        time.monotonic(),
        "GET",
        "https://provider.example.test",
        {},
        None,
        response_status=200,
    )
    call._response_body_chunks = [b"partial"]
    trace.providers["alpha"].http_calls.append(call)
    sink = S3TraceSink(_settings(), lambda _prepared: None)
    sink.start()
    event_loop_thread = threading.get_ident()
    assert sink.submit(trace, {})
    assert await asyncio.to_thread(snapshot_started.wait, 1)
    await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
    assert len(snapshot_threads) == 1
    assert snapshot_threads[0] != event_loop_thread
    release_snapshot.set()
    await sink.close()


async def test_http_request_hook_accepts_unread_streaming_body() -> None:
    async def body() -> AsyncIterator[bytes]:
        yield b"body"

    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    try:
        request = httpx.Request(
            "POST", "https://provider.example.test", content=body()
        )
        await record_http_request(request)
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)
    assert trace.providers["alpha"].http_calls[0].request_body is None


def test_provider_records_cover_success_existing_and_synthetic_failures() -> (
    None
):
    trace = SearchTrace("query", ["alpha", "beta"])
    trace.record_decision("dispatch", {"count": 2})
    trace.record_provider_start("alpha", {"query": "query"})
    pending = HttpCallRecord(
        datetime.now(UTC),
        0,
        "GET",
        "https://example.test",
        {},
        None,
    )
    completed = HttpCallRecord(
        datetime.now(UTC),
        0,
        "GET",
        "https://example.test",
        {},
        None,
        response_status=200,
    )
    trace.providers["alpha"].http_calls.extend([pending, completed])
    trace.record_provider_error("alpha", "failed", 12)
    trace.record_provider_error("beta", "missing start", 4)
    assert pending.error == "failed"
    assert completed.error is None
    assert trace.providers["beta"].input is None
    trace.record_provider_start("alpha", {})
    trace.record_provider_complete("alpha", ["result"], 8)
    assert trace.providers["alpha"].success is True


def test_trace_document_matches_legacy_shape_and_scrubs_secret_echoes() -> None:
    secret = "credential-value"
    trace = _envelope().trace
    trace.cache_hit = True
    trace.record_decision("dispatch", {"api_token": secret})
    trace.record_provider_start("alpha", {"query": "query"})
    call = HttpCallRecord(
        trace.started_at,
        0,
        "POST",
        f"https://user:pass@example.test/search?token={secret}&q=public",
        {"Authorization": f"Bearer {secret}", "Accept": "application/json"},
        json.dumps({"api_key": secret, "query": "query"}).encode(),
        response_status=401,
        response_headers={"Set-Cookie": secret},
        response_body=json.dumps({"echo": secret}).encode(),
        duration_ms=12,
        error=f"request rejected for {secret}",
    )
    trace.providers["alpha"].http_calls.append(call)
    trace.record_provider_error("alpha", f"provider rejected {secret}", 12)
    envelope = TraceEnvelope(
        trace,
        {"error": f"search failed with {secret}"},
        trace.started_at + timedelta(milliseconds=15),
    )
    document = _trace_document(envelope)
    assert set(document) == {
        "trace_id",
        "tool",
        "parent_trace_id",
        "started_at",
        "completed_at",
        "total_duration_ms",
        "cache_hit",
        "request_environment",
        "orchestrator",
        "providers_hit",
        "providers_succeeded",
        "providers_failed",
        "providers",
        "final_result",
    }
    assert document["total_duration_ms"] == 15
    encoded = json.dumps(document)
    assert secret not in encoded
    assert "[REDACTED]" in encoded
    provider = cast(dict[str, object], document["providers"])["alpha"]
    call_document = cast(dict[str, object], provider)["http_calls"]
    assert (
        cast(list[dict[str, object]], call_document)[0]["response_status"]
        == 401
    )
    cache_trace = SearchTrace("cached", ["alpha"])
    cache_trace.cache_hit = True
    cache_document = _trace_document(
        TraceEnvelope(cache_trace, {"echo": secret}, cache_trace.started_at),
        {secret},
    )
    assert cache_document["final_result"] == {"echo": "[REDACTED]"}


def test_trace_document_collects_secrets_from_every_trace_source() -> None:
    secrets = {
        "provider-input-secret",
        "provider-output-secret",
        "response-header-secret",
        "response-body-secret",
        "decision-detail-secret",
        "final-result-secret",
    }
    trace = _envelope().trace
    trace.record_provider_start(
        "alpha",
        {
            "api_key": "provider-input-secret",
            "mirror": "provider-input-secret",
        },
    )
    trace.record_provider_complete(
        "alpha",
        {
            "password": "provider-output-secret",
            "mirror": "provider-output-secret",
        },
        1,
    )
    trace.record_decision(
        "dispatch",
        {
            "token": "decision-detail-secret",
            "mirror": "decision-detail-secret",
        },
    )
    trace.providers["alpha"].http_calls.append(
        HttpCallRecord(
            trace.started_at,
            0,
            "GET",
            "https://example.test",
            {},
            None,
            response_status=200,
            response_headers={
                "x-api-key": "response-header-secret",
                "mirror": "response-header-secret",
            },
            response_body=json.dumps(
                {
                    "client_secret": "response-body-secret",
                    "mirror": "response-body-secret",
                }
            ).encode(),
        )
    )
    document = _trace_document(
        TraceEnvelope(
            trace,
            {
                "secret": "final-result-secret",
                "mirror": "final-result-secret",
            },
            trace.started_at,
        )
    )
    encoded = json.dumps(document)
    assert all(secret not in encoded for secret in secrets)
    assert encoded.count("[REDACTED]") >= len(secrets) * 2


def test_trace_document_redacts_presigned_urls_and_container_secrets() -> None:
    secret = "ephemeral-secret"
    signed_url = (
        "https://bucket.example.test/object?"
        "X-Amz-Credential=access-id%2Fscope&"
        "X-Amz-Signature=reusable-signature&"
        "X-Amz-Security-Token=session-token&public=yes"
    )
    trace = _envelope().trace
    trace.record_provider_start("alpha", {"query": "query"})
    trace.record_provider_complete(
        "alpha",
        {
            "url": signed_url,
            "mirror": "reusable-signature",
            "token": [secret],
            "echo": secret,
        },
        1,
    )
    document = _trace_document(
        TraceEnvelope(trace, {"url": signed_url}, trace.started_at)
    )
    encoded = json.dumps(document)
    assert "access-id" not in encoded
    assert "reusable-signature" not in encoded
    assert "session-token" not in encoded
    assert secret not in encoded
    assert "public=yes" in encoded


async def test_http_hook_does_not_decode_preloaded_content_twice() -> None:
    decoded_body = b'{"decoded":true}'
    trace = SearchTrace("query", ["alpha"])
    trace.record_provider_start("alpha", {})
    trace_token = activate_trace(trace)
    provider_token = activate_provider("alpha")
    request = httpx.Request("GET", "https://provider.example.test")
    try:
        await record_http_request(request)
        response = httpx.Response(
            200,
            request=request,
            headers={"Content-Encoding": "gzip"},
            content=gzip.compress(decoded_body),
        )
        assert response.content == decoded_body
        await record_http_response(response)
    finally:
        reset_provider(provider_token)
        reset_trace(trace_token)

    call = trace.providers["alpha"].http_calls[0]
    assert call.response_body == decoded_body
    assert call.response_size_bytes == len(decoded_body)
    assert call.response_body_truncated is False


def test_serialization_helpers_cover_dataclasses_enums_and_secret_rules() -> (
    None
):
    class State(Enum):
        OK = "ok"

    @dataclass
    class Value:
        state: State
        deadline_exceeded: bool

    @dataclass
    class CredentialContainer:
        token: object

    assert _jsonable(Value(State.OK, True)) == {"state": "ok"}
    assert _jsonable({"items": (State.OK, 2)}) == {"items": ["ok", 2]}
    assert _sensitive_values("plain") == set()
    assert _sensitive_values(
        {
            "token": "abc",
            "nested": [{"client_secret": "long-secret"}],
        }
    ) == {"long-secret"}
    assert _sensitive_values(
        {"token": ["container-secret"], "echo": "container-secret"}
    ) == {"container-secret"}
    assert _sensitive_string_values("token", {"nested": "mapping-secret"}) == {
        "mapping-secret"
    }
    assert _sensitive_values(CredentialContainer(["dataclass-secret"])) == {
        "dataclass-secret"
    }
    assert _sensitive_string_values(
        "token", CredentialContainer("leaf-secret")
    ) == {"leaf-secret"}
    assert _sensitive_string_values("token", 3) == set()
    assert _sensitive_string_values(
        "Authorization", "Bearer bearer-secret"
    ) == {"Bearer bearer-secret", "bearer-secret"}
    assert _sensitive_string_values(
        "proxyAuthorization", "Basic encoded-secret"
    ) == {"Basic encoded-secret", "encoded-secret"}
    assert _sensitive_string_values(
        "Authorization",
        "AWS4-HMAC-SHA256 Credential=access-id/scope, Signature=signature",
    ) == {
        "AWS4-HMAC-SHA256 Credential=access-id/scope, Signature=signature",
        "Credential=access-id/scope, Signature=signature",
        "access-id",
        "access-id/scope",
        "signature",
    }
    assert _sensitive_string_values("Authorization", "none") == {"none"}
    quoted_secret = "  'quoted-secret'  "
    assert _sensitive_string_values("api_key", quoted_secret) == {
        quoted_secret,
        "quoted-secret",
    }
    assert _sensitive_string_values("public", "Bearer visible") == set()
    assert _scrub_strings(
        {
            "long-secret-key": "prefix long-secret",
            "items": ["long-secret", 2],
        },
        {"long-secret"},
    ) == {
        "[REDACTED]-key": "prefix [REDACTED]",
        "items": ["[REDACTED]", 2],
    }


def test_bounded_snapshot_covers_binary_container_and_unknown_values() -> None:
    binary_budget = _SnapshotBudget(1)
    assert _bounded_snapshot(b"a", binary_budget) == b"a"
    assert binary_budget.truncated is True
    assert _bounded_snapshot(bytearray(b"abc"), _SnapshotBudget(10)) == b"abc"
    exhausted_budget = _SnapshotBudget(0)
    assert _bounded_snapshot(["omitted"], exhausted_budget) == []
    assert exhausted_budget.truncated is True
    unknown = _bounded_snapshot(object(), _SnapshotBudget(10))
    assert unknown == "[UNSERIALIZABLE:object]"


def test_prepared_trace_rejects_serialized_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(delivery_module, "_MAX_SERIALIZED_TRACE_BYTES", 1)
    with pytest.raises(ValueError, match="serialized size limit"):
        _prepare_trace(_envelope())


def test_object_key_supports_prefixed_and_root_layouts() -> None:
    envelope = _envelope()
    assert _object_key(_settings(), envelope) == (
        "request_traces/tool=web_search/date=2026-09-10/"
        "hour=03/trace_id=trace-1.json"
    )
    assert (
        _object_key(_settings(JASA_TRACE_S3_PREFIX="/"), envelope)
        == "tool=web_search/date=2026-09-10/hour=03/trace_id=trace-1.json"
    )


@pytest.mark.parametrize(
    ("force_path_style", "addressing_style"), [(True, "path"), (False, "auto")]
)
def test_s3_client_uses_generic_endpoint_and_addressing_style(
    monkeypatch: pytest.MonkeyPatch,
    force_path_style: bool,
    addressing_style: str,
) -> None:
    captured: dict[str, object] = {}

    def client(service: str, **kwargs: object) -> object:
        captured["service"] = service
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(boto3, "client", client)
    result = _build_s3_client(
        _settings(JASA_TRACE_S3_FORCE_PATH_STYLE=force_path_style)
    )
    assert result is not None
    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "https://objects.example.test"
    config = cast(Config, captured["config"])
    assert config.s3 == {"addressing_style": addressing_style}


async def test_sink_disabled_unstarted_overflow_and_upload_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert build_trace_sink(TraceSettings()) is None
    assert isinstance(build_trace_sink(_settings()), S3TraceSink)
    sink = S3TraceSink(_settings(JASA_TRACE_S3_QUEUE_CAPACITY=1))
    assert sink.submit(_envelope().trace, {}) is False
    await sink.close()
    upload_started = threading.Event()
    release_upload = threading.Event()

    def blocking_upload(_prepared: _PreparedTrace) -> None:
        upload_started.set()
        release_upload.wait(timeout=2)

    class Uncopyable:
        copy_attempts = 0

        def __deepcopy__(self, _memo: object) -> object:
            self.copy_attempts += 1
            raise ValueError("cannot snapshot")

    overflow = S3TraceSink(
        _settings(JASA_TRACE_S3_QUEUE_CAPACITY=1), blocking_upload
    )
    overflow.start()
    assert overflow.submit(_envelope().trace, {}) is True
    assert await asyncio.to_thread(upload_started.wait, 1)
    assert overflow.submit(_envelope().trace, {}) is True
    saturated_value = Uncopyable()
    with caplog.at_level(
        logging.WARNING, logger="jasa.observability.trace_delivery"
    ):
        assert overflow.submit(_envelope().trace, saturated_value) is False
        assert saturated_value.copy_attempts == 0
        assert not any("queue saturation" in item for item in caplog.messages)
        release_upload.set()
        await overflow.close()
    assert "Trace queue saturation dropped_count=1" in caplog.messages

    monkeypatch.setattr(delivery_module, "_MAX_QUEUED_TRACE_BYTES", 1)
    byte_bounded = S3TraceSink(_settings())
    byte_bounded.start()
    assert byte_bounded.submit(_envelope().trace, {}) is False

    uncopyable = Uncopyable()
    assert byte_bounded.submit(_envelope().trace, uncopyable) is False
    assert uncopyable.copy_attempts == 0
    await byte_bounded.close()
    assert "Trace queue saturation dropped_count=2" in caplog.messages
    assert byte_bounded._accepted_trace_bytes == 0
    monkeypatch.setattr(
        delivery_module, "_MAX_QUEUED_TRACE_BYTES", 32 * 1024 * 1024
    )

    def fail(_prepared: _PreparedTrace) -> None:
        raise OSError("offline")

    failing = S3TraceSink(_settings(), fail)
    failing.start()
    failing.start()
    assert failing.submit(_envelope().trace, {}) is True
    with caplog.at_level(
        logging.WARNING, logger="jasa.observability.trace_delivery"
    ):
        await failing.close()
    assert "S3 trace upload failed error_type=OSError" in caplog.messages


async def test_snapshot_and_encoding_never_block_the_event_loop() -> None:
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    snapshot_threads: list[int] = []
    uploaded: list[_PreparedTrace] = []

    class SlowMapping(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            if key != "value":
                raise KeyError(key)
            return "visible"

        def __iter__(self) -> Iterator[str]:
            snapshot_threads.append(threading.get_ident())
            snapshot_started.set()
            release_snapshot.wait(timeout=2)
            return iter(("value",))

        def __len__(self) -> int:
            return 1

    sink = S3TraceSink(_settings(), uploaded.append)
    sink.start()
    event_loop_thread = threading.get_ident()
    assert sink.submit(_envelope().trace, SlowMapping())
    assert await asyncio.to_thread(snapshot_started.wait, 1)
    heartbeat_completed = False

    async def heartbeat() -> None:
        nonlocal heartbeat_completed
        await asyncio.sleep(0)
        heartbeat_completed = True

    await asyncio.wait_for(heartbeat(), timeout=0.1)
    assert heartbeat_completed is True
    assert snapshot_threads
    assert all(thread != event_loop_thread for thread in snapshot_threads)
    release_snapshot.set()
    await sink.close()
    assert json.loads(uploaded[0].body)["final_result"] == {"value": "visible"}


def test_prepared_trace_bounds_provider_output_and_cached_final_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(delivery_module, "_MAX_SNAPSHOT_BYTES", 1024)
    trace = _envelope().trace
    trace.record_provider_start("alpha", {"query": "query"})
    trace.providers["alpha"].http_calls.append(
        HttpCallRecord(
            trace.started_at,
            0,
            "GET",
            "https://provider.example.test",
            {},
            None,
        )
    )
    trace.record_provider_complete("alpha", [{"snippet": "x" * 4096}], 1)
    prepared = _prepare_trace(
        TraceEnvelope(trace, {"results": ["y" * 4096]}, trace.started_at)
    )
    document = json.loads(prepared.body)
    assert document["trace_truncated"] is True
    assert "x" * 4096 not in prepared.body.decode()
    assert "y" * 4096 not in prepared.body.decode()
    assert len(prepared.body) <= delivery_module._MAX_SERIALIZED_TRACE_BYTES
    providers = cast(dict[str, dict[str, object]], document["providers"])
    assert providers["alpha"]["http_calls"] == []

    cache_trace = SearchTrace("cached", [])
    cache_trace.cache_hit = True
    cached = _prepare_trace(
        TraceEnvelope(
            cache_trace,
            {"results": ["z" * 4096]},
            cache_trace.started_at,
        )
    )
    cached_document = json.loads(cached.body)
    assert cached_document["cache_hit"] is True
    assert cached_document["trace_truncated"] is True
    assert "z" * 4096 not in cached.body.decode()


async def test_sink_closes_s3_client_off_loop_and_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_threads: list[int] = []
    client = MagicMock()
    client.close.side_effect = lambda: close_threads.append(
        threading.get_ident()
    )
    monkeypatch.setattr(
        delivery_module, "_build_s3_client", MagicMock(return_value=client)
    )
    sink = S3TraceSink(_settings())
    sink.start()
    assert sink.submit(_envelope().trace, {})
    event_loop_thread = threading.get_ident()
    await sink.close()
    client.close.assert_called_once_with()
    assert close_threads[0] != event_loop_thread
    assert sink._client is None


async def test_sink_close_without_client_and_close_failure_are_fail_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unused = S3TraceSink(_settings())
    unused.start()
    await unused.close()
    assert unused._client is None

    class FailingClient:
        def close(self) -> None:
            raise OSError("close failed")

    failing = S3TraceSink(_settings())
    failing._client = cast(object, FailingClient())
    failing.start()
    with caplog.at_level(
        logging.WARNING, logger="jasa.observability.trace_delivery"
    ):
        await failing.close()
    assert failing._client is None
    assert "S3 trace client close failed error_type=OSError" in caplog.messages


async def test_sink_snapshot_failure_releases_reserved_capacity(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_snapshot(
        _envelope: TraceEnvelope, _secrets: object
    ) -> _PreparedTrace:
        raise ValueError("snapshot failed")

    monkeypatch.setattr(delivery_module, "_prepare_trace", fail_snapshot)
    sink = S3TraceSink(_settings())
    sink.start()
    with caplog.at_level(
        logging.WARNING, logger="jasa.observability.trace_delivery"
    ):
        assert sink.submit(_envelope().trace, {})
        await sink.close()
    assert sink._accepted_submissions == 0
    assert sink._accepted_trace_bytes == 0
    assert "Trace snapshot failed error_type=ValueError" in caplog.messages
    assert "Trace queue saturation dropped_count=1" in caplog.messages


@pytest.mark.parametrize(
    "settings",
    [
        _settings(JASA_TRACE_S3_ENDPOINT="http://objects.example.test"),
        _settings(JASA_TRACE_S3_BUCKET=""),
    ],
)
def test_trace_sink_rejects_unsafe_or_incomplete_settings(
    settings: TraceSettings,
) -> None:
    with pytest.raises(ValueError):
        build_trace_sink(settings)


def test_sync_upload_builds_client_once_and_writes_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    build_client = MagicMock(return_value=client)
    monkeypatch.setattr(delivery_module, "_build_s3_client", build_client)
    sink = S3TraceSink(_settings(), configured_secrets={"cache-secret"})
    cached_envelope = _envelope()
    cached_envelope = TraceEnvelope(
        cached_envelope.trace,
        {"echo": "cache-secret"},
        cached_envelope.completed_at,
    )
    sink._upload_sync(_prepare_trace(cached_envelope, {"cache-secret"}))
    sink._upload_sync(_prepare_trace(_envelope()))
    build_client.assert_called_once()
    assert client.put_object.call_count == 2
    call = client.put_object.call_args.kwargs
    assert call["Bucket"] == "traces"
    assert call["ContentType"] == "application/json"
    assert json.loads(call["Body"])["trace_id"] == "trace-1"
    first_body = json.loads(client.put_object.call_args_list[0].kwargs["Body"])
    assert first_body["final_result"] == {"echo": "[REDACTED]"}


def test_trace_sink_scrubs_raw_and_provider_normalized_credentials() -> None:
    raw_secret = "  'cache-secret'  "
    raw_access_key = "  'destination-access'  "
    raw_destination_secret = '  "destination-secret"  '
    sink = build_trace_sink(
        _settings(
            JASA_TRACE_S3_ACCESS_KEY_ID=raw_access_key,
            JASA_TRACE_S3_SECRET_ACCESS_KEY=raw_destination_secret,
        ),
        {"ALPHA_API_KEY": raw_secret},
    )
    assert sink is not None
    assert {
        raw_secret,
        "cache-secret",
        raw_access_key,
        "destination-access",
        raw_destination_secret,
        "destination-secret",
    } <= sink._configured_secrets
    cache_trace = SearchTrace("cached", ["alpha"])
    cache_trace.cache_hit = True
    document = _trace_document(
        TraceEnvelope(
            cache_trace,
            {"echo": ("cache-secret destination-access destination-secret")},
            cache_trace.started_at,
        ),
        sink._configured_secrets,
    )
    assert document["final_result"] == {
        "echo": "[REDACTED] [REDACTED] [REDACTED]"
    }


async def test_search_returns_while_trace_upload_blocks_on_worker_thread() -> (
    None
):
    upload_started = threading.Event()
    release_upload = threading.Event()
    documents: list[dict[str, object]] = []

    def upload(prepared: _PreparedTrace) -> None:
        upload_started.set()
        release_upload.wait(timeout=2)
        documents.append(json.loads(prepared.body))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json={"ok": True})

    sink = S3TraceSink(_settings(), upload)
    sink.start()
    async with httpx.AsyncClient(
        base_url="https://provider.example.test",
        transport=httpx.MockTransport(handler),
        event_hooks={
            "request": [record_http_request],
            "response": [record_http_response],
        },
    ) as client:
        provider = _HttpProvider(client)
        outcome = await run_search(
            {"alpha": provider},
            MemoryCache(),
            "query",
            options=SearchOptions(trace_sink=sink),
        )
        assert outcome.web_results
        assert await asyncio.to_thread(upload_started.wait, 1)
        await asyncio.sleep(0)
        assert documents == []
        release_upload.set()
        await sink.close()
    assert documents[0]["providers_succeeded"] == ["alpha"]
    provider_document = cast(dict[str, object], documents[0]["providers"])
    calls = cast(dict[str, object], provider_document["alpha"])["http_calls"]
    assert len(cast(list[object], calls)) == 1


async def test_submit_snapshots_mutable_trace_state_before_worker_upload() -> (
    None
):
    upload_started = threading.Event()
    release_upload = threading.Event()
    documents: list[dict[str, object]] = []

    def upload(prepared: _PreparedTrace) -> None:
        upload_started.set()
        release_upload.wait(timeout=2)
        documents.append(json.loads(prepared.body))

    trace = _envelope().trace
    trace.record_provider_start("alpha", {"query": "before"})
    trace.record_provider_complete("alpha", ["before"], 1)
    final_result = {"items": ["before"]}
    sink = S3TraceSink(_settings(), upload)
    sink.start()
    assert sink.submit(trace, final_result)
    assert await asyncio.to_thread(upload_started.wait, 1)
    trace.providers["alpha"].output = ["after"]
    final_result["items"].append("after")
    release_upload.set()
    await sink.close()

    provider = cast(dict[str, object], documents[0]["providers"])["alpha"]
    assert cast(dict[str, object], provider)["output"] == ["before"]
    assert documents[0]["final_result"] == {"items": ["before"]}


async def test_submit_freezes_trace_before_background_snapshot() -> None:
    uploaded: list[_PreparedTrace] = []
    trace = _envelope().trace
    trace.record_provider_start("alpha", {"query": "before"})
    trace.record_provider_error("alpha", "timed out", 10)
    trace.record_decision("before", {"state": "complete"})
    sink = S3TraceSink(_settings(), uploaded.append)
    sink.start()

    assert sink.submit(trace, {"items": ["before"]})
    assert trace.frozen is True
    assert trace.reserve_response_capture(1) is False
    trace.release_response_capture(1)
    trace.record_provider_complete("alpha", ["late success"], 20)
    trace.record_provider_start("late", {"query": "late"})
    trace.record_provider_error("late", "late error", 30)
    trace.record_decision("late", {"state": "mutated"})
    await sink.close()

    document = json.loads(uploaded[0].body)
    provider = cast(dict[str, object], document["providers"])["alpha"]
    assert cast(dict[str, object], provider)["success"] is False
    assert cast(dict[str, object], provider)["output"] is None
    assert document["providers_hit"] == ["alpha"]
    orchestrator = cast(dict[str, object], document["orchestrator"])
    decisions = cast(list[dict[str, object]], orchestrator["decisions"])
    assert [decision["action"] for decision in decisions] == ["before"]


async def test_non_cacheable_waiter_trace_records_in_process_flight() -> None:
    uploaded: list[_PreparedTrace] = []
    gate = asyncio.Event()
    waiter_coalesced = asyncio.Event()
    provider = _HttpProvider(gate=gate, cache_allowed=False)
    sink = S3TraceSink(_settings(), uploaded.append)
    sink.start()
    flights = SearchFlightRegistry()
    cache = MemoryCache()

    async def report_progress(
        _progress: float, _total: float | None, message: str | None
    ) -> None:
        if message is not None and message.startswith("Waiting for"):
            waiter_coalesced.set()

    options = SearchOptions(
        flights=flights,
        progress_reporter=report_progress,
        trace_sink=sink,
    )
    leader = asyncio.create_task(
        run_search({"alpha": provider}, cache, "query", options=options)
    )
    while provider.calls == 0:
        await asyncio.sleep(0)
    waiter = asyncio.create_task(
        run_search({"alpha": provider}, cache, "query", options=options)
    )
    await asyncio.wait_for(waiter_coalesced.wait(), timeout=1)
    gate.set()
    await asyncio.gather(leader, waiter)
    await sink.close()

    documents = [json.loads(prepared.body) for prepared in uploaded]
    strategies = [
        cast(dict[str, object], document["orchestrator"])["strategy"]
        for document in documents
    ]
    assert strategies.count("parallel_fanout") == 1
    assert strategies.count("in_process_flight") == 1
    flight_document = documents[strategies.index("in_process_flight")]
    assert flight_document["cache_hit"] is False
    assert flight_document["providers_hit"] == []
    orchestrator = cast(dict[str, object], flight_document["orchestrator"])
    assert any(
        decision["action"] == "coalesced_result"
        and decision["details"] == {"source": "in_process_flight"}
        for decision in cast(list[dict[str, object]], orchestrator["decisions"])
    )


async def test_trace_dispatch_duration_excludes_cache_read_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded: list[_PreparedTrace] = []
    now = [0.0]

    class SlowCache(MemoryCache):
        async def get(self, key: str) -> str | None:
            now[0] = 2.0
            return await super().get(key)

    async def dispatch(*_args: object, **_kwargs: object) -> DispatchResult:
        now[0] = 5.0
        return DispatchResult(
            {
                "alpha": [
                    SearchResult(
                        "Title",
                        "https://result.example.test",
                        "long result snippet " * 5,
                        "alpha",
                    )
                ]
            },
            [ProviderSuccess("alpha", 1)],
            [],
        )

    monkeypatch.setattr(service_module, "dispatch_to_providers", dispatch)
    sink = S3TraceSink(_settings(), uploaded.append)
    sink.start()
    await run_search(
        {"alpha": _HttpProvider()},
        SlowCache(),
        "query",
        options=SearchOptions(trace_sink=sink),
        knobs=_FanoutKnobs(clock=lambda: now[0]),
    )
    await sink.close()

    document = json.loads(uploaded[0].body)
    orchestrator = cast(dict[str, object], document["orchestrator"])
    dispatch_complete = next(
        decision
        for decision in cast(list[dict[str, object]], orchestrator["decisions"])
        if decision["action"] == "dispatch_complete"
    )
    assert (
        cast(dict[str, object], dispatch_complete["details"])[
            "dispatch_duration_ms"
        ]
        == 3000
    )


async def test_trace_records_cache_hit_parent_error_and_timeout() -> None:
    uploaded: list[_PreparedTrace] = []
    sink = S3TraceSink(_settings(), uploaded.append)
    sink.start()
    cache = MemoryCache()
    provider = _HttpProvider()
    await run_search({"alpha": provider}, cache, "cached")
    await run_search(
        {"alpha": provider},
        cache,
        "cached",
        options=SearchOptions(trace_sink=sink),
    )
    parent = SearchTrace("parent", ["parent"], trace_id="parent-id")
    token = activate_trace(parent)
    try:
        failing = _HttpProvider(
            error=ProviderError(ErrorType.API_ERROR, "provider failed", "alpha")
        )
        with pytest.raises(
            SearchError, match="All configured search providers"
        ):
            await run_search(
                {"alpha": failing},
                MemoryCache(),
                "failure",
                options=SearchOptions(trace_sink=sink),
            )
        assert active_trace() is parent
    finally:
        reset_trace(token)
    with pytest.raises(SearchError) as timeout_error:
        await run_search(
            {"alpha": _HttpProvider(delay=0.05)},
            MemoryCache(),
            "timeout",
            options=SearchOptions(
                trace_sink=sink, timeout_ms=1, fanout_timeout_ms=1
            ),
        )
    assert timeout_error.value.kind == "deadline_exceeded"
    await sink.close()
    documents = [json.loads(prepared.body) for prepared in uploaded]
    assert documents[0]["cache_hit"] is True
    assert documents[0]["parent_trace_id"] is None
    assert documents[1]["parent_trace_id"] == "parent-id"
    providers = cast(dict[str, dict[str, object]], documents[1]["providers"])
    assert providers["alpha"]["error"] == "ProviderError"
    assert documents[1]["final_result"] == {"error_type": "SearchError"}
    timeout_providers = cast(
        dict[str, dict[str, object]], documents[2]["providers"]
    )
    assert timeout_providers["alpha"]["error"] == "TimeoutError"

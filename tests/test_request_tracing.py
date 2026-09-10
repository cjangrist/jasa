"""Request-trace capture, redaction, background delivery, and search wiring."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import threading
from collections.abc import AsyncIterator
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
from jasa.cache.memory import MemoryCache
from jasa.config import TraceSettings
from jasa.observability.trace_delivery import (
    _build_s3_client,
    _jsonable,
    _object_key,
    _scrub_strings,
    _sensitive_string_values,
    _sensitive_values,
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
from jasa.search.providers.base import SearchProvider, SearchRequest
from jasa.search.ranking import SearchResult
from jasa.search.service import run_search, SearchError, SearchOptions
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
    ) -> None:
        self.client = client
        self.error = error
        self.delay = delay
        self.calls = 0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
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
    assert failed.error == "RuntimeError"
    assert compressed.response_body == decoded_body
    assert compressed.response_size_bytes == len(decoded_body)


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
    assert preloaded.response_body is None
    assert preloaded.response_body_truncated is True


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
    assert trace.providers["alpha"].http_calls[0].response_body == b""


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


def test_serialization_helpers_cover_dataclasses_enums_and_secret_rules() -> (
    None
):
    class State(Enum):
        OK = "ok"

    @dataclass
    class Value:
        state: State
        deadline_exceeded: bool

    assert _jsonable(Value(State.OK, True)) == {"state": "ok"}
    assert _jsonable({"items": (State.OK, 2)}) == {"items": ["ok", 2]}
    assert _sensitive_values("plain") == set()
    assert _sensitive_values(
        {
            "token": "abc",
            "nested": [{"client_secret": "long-secret"}],
        }
    ) == {"long-secret"}
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
    assert _sensitive_string_values("public", "Bearer visible") == set()
    assert _scrub_strings(
        {"text": "prefix long-secret", "items": ["long-secret", 2]},
        {"long-secret"},
    ) == {"text": "prefix [REDACTED]", "items": ["[REDACTED]", 2]}


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

    def blocking_upload(_envelope: TraceEnvelope) -> None:
        upload_started.set()
        release_upload.wait(timeout=2)

    overflow = S3TraceSink(
        _settings(JASA_TRACE_S3_QUEUE_CAPACITY=1), blocking_upload
    )
    overflow.start()
    assert overflow.submit(_envelope().trace, {}) is True
    assert await asyncio.to_thread(upload_started.wait, 1)
    assert overflow.submit(_envelope().trace, {}) is True
    with caplog.at_level(
        logging.WARNING, logger="jasa.observability.trace_delivery"
    ):
        assert overflow.submit(_envelope().trace, {}) is False
        assert not any("queue saturation" in item for item in caplog.messages)
        release_upload.set()
        await overflow.close()
    assert "Trace queue saturation dropped_count=1" in caplog.messages

    monkeypatch.setattr(delivery_module, "_MAX_QUEUED_CAPTURE_BYTES", 1)
    byte_bounded = S3TraceSink(_settings())
    byte_bounded.start()
    oversized_trace = _envelope().trace
    oversized_trace.captured_response_bytes = 2
    assert byte_bounded.submit(oversized_trace, {}) is False

    class Uncopyable:
        def __deepcopy__(self, _memo: object) -> object:
            raise ValueError("cannot snapshot")

    assert byte_bounded.submit(_envelope().trace, Uncopyable()) is False
    await byte_bounded.close()
    assert "Trace queue saturation dropped_count=2" in caplog.messages
    assert byte_bounded._accepted_capture_bytes == 0

    def fail(_envelope: TraceEnvelope) -> None:
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
    sink._upload_sync(cached_envelope)
    sink._upload_sync(_envelope())
    build_client.assert_called_once()
    assert client.put_object.call_count == 2
    call = client.put_object.call_args.kwargs
    assert call["Bucket"] == "traces"
    assert call["ContentType"] == "application/json"
    assert json.loads(call["Body"])["trace_id"] == "trace-1"
    first_body = json.loads(client.put_object.call_args_list[0].kwargs["Body"])
    assert first_body["final_result"] == {"echo": "[REDACTED]"}


async def test_search_returns_while_trace_upload_blocks_on_worker_thread() -> (
    None
):
    upload_started = threading.Event()
    release_upload = threading.Event()
    documents: list[dict[str, object]] = []

    def upload(envelope: TraceEnvelope) -> None:
        upload_started.set()
        release_upload.wait(timeout=2)
        documents.append(_trace_document(envelope))

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

    def upload(envelope: TraceEnvelope) -> None:
        upload_started.set()
        release_upload.wait(timeout=2)
        documents.append(_trace_document(envelope))

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


async def test_trace_records_cache_hit_parent_error_and_timeout() -> None:
    uploaded: list[TraceEnvelope] = []
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
    assert uploaded[0].trace.cache_hit is True
    assert uploaded[0].trace.parent_trace_id is None
    assert uploaded[1].trace.parent_trace_id == "parent-id"
    assert uploaded[1].trace.providers["alpha"].error == "ProviderError"
    assert uploaded[1].final_result == {"error_type": "SearchError"}
    assert uploaded[2].trace.providers["alpha"].error == "TimeoutError"

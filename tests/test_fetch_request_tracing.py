"""Public web-fetch trace shape, delivery, and transport wiring."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, UTC
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from starlette.testclient import TestClient

import jasa.observability.trace_delivery as delivery_module
from jasa.config import load_config, TraceSettings
from jasa.observability.trace_delivery import (
    _object_key,
    _prepare_trace,
    _PreparedTrace,
    _sensitive_values,
    _snapshot_mapping,
    _snapshot_model,
    _SnapshotBudget,
    _trace_document,
    FetchTraceEnvelope,
    S3TraceSink,
)
from jasa.observability.traces import FetchTrace, FetchTraceMiddleware
from jasa.server import build_composition, build_composition_async
from omnifetch.fetch.engine.race import (
    FetchExhaustionDetails,
    ProviderAttemptFailure,
)
from omnifetch.fetch.shared.types import ErrorType, ProviderError
from omnifetch.schemas import (
    FetchAlternative,
    FetchProviderFailure,
    FetchResponse,
)


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
        "JASA_TRACE_S3_QUEUE_CAPACITY": 4,
    }
    values.update(overrides)
    return TraceSettings.model_validate(values)


def _fetch_trace(**overrides: object) -> FetchTrace:
    values = {
        "request_environment": {
            "url": "https://example.test/article",
            "skip_providers": ["alpha"],
        },
        "active_providers": ("alpha", "beta", "gamma"),
        "transport": "mcp",
        "trace_id": "fetch-trace-1",
        "started_at": datetime(2026, 9, 10, 3, 4, 5, tzinfo=UTC),
    }
    values.update(overrides)
    return FetchTrace(**cast(Any, values))


def _fetch_response(**overrides: object) -> FetchResponse:
    values = {
        "url": "https://example.test/article",
        "title": "Example",
        "content": "Fetched content",
        "source_provider": "beta",
        "total_duration_ms": 12,
        "providers_attempted": ["alpha", "beta", "gamma"],
        "providers_failed": [
            FetchProviderFailure(
                provider="alpha",
                error="failed",
                duration_ms=4,
                error_type="API_ERROR",
            )
        ],
        "alternative_results": [
            FetchAlternative(
                url="https://example.test/article",
                title="Alternative",
                content="Alternative content",
                source_provider="gamma",
            )
        ],
    }
    values.update(overrides)
    return FetchResponse.model_validate(values)


def _envelope(
    trace: FetchTrace | None = None,
    result: object | None = None,
    *,
    error: str | None = None,
) -> FetchTraceEnvelope:
    resolved = trace or _fetch_trace()
    completed = resolved.started_at + timedelta(milliseconds=25)
    return FetchTraceEnvelope(
        resolved,
        result if result is not None else _fetch_response(),
        completed,
        error=error,
    )


def _enable_trace_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASA_TRACE_S3_ENABLED", "true")
    monkeypatch.setenv("JASA_TRACE_S3_ENDPOINT", "https://objects.example.test")
    monkeypatch.setenv("JASA_TRACE_S3_BUCKET", "traces")
    monkeypatch.setenv("JASA_TRACE_S3_ACCESS_KEY_ID", "access-id")
    monkeypatch.setenv("JASA_TRACE_S3_SECRET_ACCESS_KEY", "secret-key")


def test_fetch_document_matches_search_envelope_and_provider_evidence() -> None:
    document = _trace_document(_envelope())
    assert set(document) == {
        "trace_id",
        "tool",
        "parent_trace_id",
        "started_at",
        "completed_at",
        "total_duration_ms",
        "cache_hit",
        "provider_evidence",
        "request_environment",
        "orchestrator",
        "providers_hit",
        "providers_succeeded",
        "providers_failed",
        "providers",
        "final_result",
    }
    assert document["tool"] == "web_fetch"
    assert document["cache_hit"] is None
    assert document["provider_evidence"] == {
        "scope": "returned_result_origin",
        "current_request_execution": "unknown",
    }
    assert document["request_environment"] == {
        "transport": "mcp",
        "arguments": {
            "url": "https://example.test/article",
            "skip_providers": ["alpha"],
        },
    }
    assert document["providers_hit"] == ["alpha", "beta", "gamma"]
    assert document["providers_succeeded"] == ["beta", "gamma"]
    providers = cast(dict[str, dict[str, object]], document["providers"])
    assert providers["alpha"]["success"] is False
    assert providers["alpha"]["error"] == "failed"
    assert providers["beta"]["success"] is True
    assert providers["gamma"]["success"] is True


def test_fetch_document_records_exception_class_without_message() -> None:
    document = _trace_document(
        _envelope(result={"password": "must-not-appear"}, error="TimeoutError")
    )
    assert document["providers_hit"] == []
    assert document["providers_succeeded"] == []
    assert document["providers_failed"] == []
    assert document["final_result"] == {"error": "TimeoutError"}
    assert "must-not-appear" not in json.dumps(document)


def test_fetch_trace_redacts_signed_urls_and_duplicate_credentials() -> None:
    secret = "fetch-secret-value"
    fragment_secret = "fragment-secret-value"
    userinfo_secret = "userinfo-secret-value"
    password_secret = "password-secret-value"
    signed_url = (
        f"  HTTPS://{userinfo_secret}:{password_secret}@example.test/article?"
        "X-Amz-Credential=access-id%2Fscope&"
        "X-Amz-Signature=signed-value&public=yes"
        f"#access_token={fragment_secret}  "
    )
    trace = _fetch_trace(
        request_environment={"url": signed_url, "api_key": secret}
    )
    response = _fetch_response(
        content=f"content echo {secret} and {fragment_secret}",
        metadata={"mirror": "signed-value", "token": secret},
    )
    encoded = json.dumps(_trace_document(_envelope(trace, response)))
    assert secret not in encoded
    assert fragment_secret not in encoded
    assert userinfo_secret not in encoded
    assert password_secret not in encoded
    assert "signed-value" not in encoded
    assert "access-id" not in encoded
    assert "public=yes" in encoded
    assert "  HTTPS://" not in encoded
    assert "#" not in encoded
    assert "[REDACTED]" in encoded
    assert "model-secret" in _sensitive_values(
        {"token": _fetch_response(content="model-secret")}
    )


def test_fetch_object_key_and_prepared_tool_use_partition_layout() -> None:
    envelope = _envelope()
    assert _object_key(_settings(), envelope) == (
        "request_traces/tool=web_fetch/date=2026-09-10/"
        "hour=03/trace_id=fetch-trace-1.json"
    )
    prepared = _prepare_trace(envelope)
    assert prepared.tool == "web_fetch"
    assert json.loads(prepared.body)["tool"] == "web_fetch"


def test_fetch_snapshot_bounds_pydantic_response_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(delivery_module, "_MAX_SNAPSHOT_BYTES", 1024)
    response = _fetch_response(
        content="x" * 4096,
        metadata={"payload": "y" * 4096},
        alternative_results=None,
    )
    prepared = _prepare_trace(_envelope(result=response))
    document = json.loads(prepared.body)
    assert document["trace_truncated"] is True
    assert document["providers_hit"] == ["alpha", "beta", "gamma"]
    assert document["providers_succeeded"] == ["beta"]
    assert len(document["providers_failed"]) == 1
    assert document["final_result"]["source_provider"] == "beta"
    assert document["final_result"]["providers_attempted"] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert "x" * 4096 not in prepared.body.decode()
    assert len(prepared.body) <= delivery_module._MAX_SERIALIZED_TRACE_BYTES
    mapping_prepared = _prepare_trace(
        _envelope(result=response.model_dump(mode="python"))
    )
    mapping_document = json.loads(mapping_prepared.body)
    assert mapping_document["trace_truncated"] is True
    assert mapping_document["providers_hit"] == ["alpha", "beta", "gamma"]
    assert mapping_document["providers_succeeded"] == ["beta"]
    assert mapping_document["final_result"]["source_provider"] == "beta"
    assert mapping_document["final_result"]["providers_attempted"] == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_fetch_snapshot_marks_exact_budget_field_omission() -> None:
    budget = _SnapshotBudget(25)
    snapshot = _snapshot_model(_fetch_response(), budget)
    assert snapshot == {"status": "success"}
    assert budget.remaining_bytes == 0
    assert budget.truncated is True
    mapping_budget = _SnapshotBudget(21)
    mapping_snapshot = _snapshot_mapping(
        {"status": "success", "content": "x"}, mapping_budget
    )
    assert mapping_snapshot == {"status": "success"}
    assert mapping_budget.remaining_bytes == 0
    assert mapping_budget.truncated is True


def test_mapping_snapshot_stops_iterating_at_byte_budget() -> None:
    class GuardedMapping(Mapping[object, object]):
        iterations = 0

        def __getitem__(self, key: object) -> object:
            return key

        def __iter__(self) -> Iterator[object]:
            for index in range(1_000_000):
                self.iterations += 1
                if self.iterations > 30:
                    raise AssertionError("mapping traversal exceeded budget")
                yield str(index)

        def __len__(self) -> int:
            return 1_000_000

    mapping = GuardedMapping()
    budget = _SnapshotBudget(64)
    snapshot = _snapshot_mapping(mapping, budget)
    assert snapshot
    assert mapping.iterations < 30
    assert budget.remaining_bytes == 0
    assert budget.truncated is True


def test_mapping_snapshot_bounds_deferred_field_traversal() -> None:
    class DeferredKey:
        def __str__(self) -> str:
            return "content"

    class DeferredMapping(Mapping[object, object]):
        iterations = 0

        def __getitem__(self, key: object) -> object:
            return key

        def __iter__(self) -> Iterator[object]:
            for _index in range(1_000_000):
                self.iterations += 1
                if self.iterations > 10:
                    raise AssertionError("deferred traversal was not bounded")
                yield DeferredKey()

        def __len__(self) -> int:
            return 1_000_000

    mapping = DeferredMapping()
    budget = _SnapshotBudget(64)
    snapshot = _snapshot_mapping(mapping, budget)
    assert snapshot
    assert mapping.iterations == 3
    assert budget.truncated is True


def test_mapping_snapshot_prioritizes_metadata_over_earlier_content() -> None:
    budget = _SnapshotBudget(64)
    snapshot = _snapshot_mapping(
        {
            "content": "x" * 4096,
            "metadata": {"source_provider": "beta"},
        },
        budget,
    )
    assert snapshot["metadata"] == {"source_provider": "beta"}
    assert cast(str, snapshot["content"]).endswith("[TRUNCATED]")
    assert budget.remaining_bytes == 0
    assert budget.truncated is True


async def test_fetch_middleware_submits_success_error_and_cancellation() -> (
    None
):
    sink = MagicMock()
    sink.submit_fetch.return_value = True
    middleware = FetchTraceMiddleware(sink, ["alpha", "beta"])
    context = SimpleNamespace(
        message=SimpleNamespace(
            name="web_fetch",
            arguments={"url": "https://example.test"},
        )
    )
    success = ToolResult(
        structured_content=_fetch_response().model_dump(mode="json")
    )

    async def return_success(_context: object) -> ToolResult:
        return success

    assert (
        await middleware.on_call_tool(
            cast(Any, context), cast(Any, return_success)
        )
        is success
    )
    submitted_trace, submitted_result = sink.submit_fetch.call_args.args
    assert submitted_trace.transport == "mcp"
    assert submitted_trace.active_providers == ("alpha", "beta")
    assert submitted_result["source_provider"] == "beta"

    invalid = ToolResult(
        content=[TextContent(type="text", text="invalid")], is_error=True
    )

    async def return_invalid(_context: object) -> ToolResult:
        return invalid

    await middleware.on_call_tool(cast(Any, context), cast(Any, return_invalid))
    invalid_payload = sink.submit_fetch.call_args.args[1]
    assert invalid_payload["is_error"] is True

    async def raise_error(_context: object) -> ToolResult:
        raise ValueError("secret message")

    with pytest.raises(ValueError, match="secret message"):
        await middleware.on_call_tool(
            cast(Any, context), cast(Any, raise_error)
        )
    assert sink.submit_fetch.call_args.kwargs == {"error": "ValueError"}

    async def cancel(_context: object) -> ToolResult:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await middleware.on_call_tool(cast(Any, context), cast(Any, cancel))
    assert sink.submit_fetch.call_args.kwargs == {"error": "CancelledError"}


async def test_fetch_middleware_ignores_other_tools() -> None:
    sink = MagicMock()
    middleware = FetchTraceMiddleware(sink, [])
    context = SimpleNamespace(
        message=SimpleNamespace(name="web_search", arguments={"query": "x"})
    )

    async def call_next(_context: object) -> str:
        return "unchanged"

    assert (
        await middleware.on_call_tool(cast(Any, context), cast(Any, call_next))
        == "unchanged"
    )
    sink.submit_fetch.assert_not_called()


async def test_fetch_snapshot_and_upload_do_not_block_event_loop() -> None:
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    upload_started = threading.Event()
    release_upload = threading.Event()
    worker_threads: list[int] = []
    uploaded: list[_PreparedTrace] = []

    class SlowMapping(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            if key != "url":
                raise KeyError(key)
            return "https://example.test"

        def __iter__(self) -> Iterator[str]:
            worker_threads.append(threading.get_ident())
            snapshot_started.set()
            release_snapshot.wait(timeout=2)
            return iter(("url",))

        def __len__(self) -> int:
            return 1

    def upload(prepared: _PreparedTrace) -> None:
        worker_threads.append(threading.get_ident())
        upload_started.set()
        release_upload.wait(timeout=2)
        uploaded.append(prepared)

    sink = S3TraceSink(_settings(), upload)
    sink.start()
    event_loop_thread = threading.get_ident()
    trace = _fetch_trace(request_environment=SlowMapping())
    assert sink.submit_fetch(trace, _fetch_response())
    assert await asyncio.to_thread(snapshot_started.wait, 1)
    await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
    release_snapshot.set()
    assert await asyncio.to_thread(upload_started.wait, 1)
    await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
    assert uploaded == []
    assert all(thread != event_loop_thread for thread in worker_threads)
    release_upload.set()
    await sink.close()
    assert json.loads(uploaded[0].body)["tool"] == "web_fetch"


async def test_mcp_fetch_traces_success_terminal_failure_and_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_trace_environment(monkeypatch)
    monkeypatch.setenv("TAVILY_API_KEY", "provider-key")
    uploaded: list[_PreparedTrace] = []
    responses: list[FetchResponse | BaseException] = [
        _fetch_response(alternative_results=None),
        ProviderError(ErrorType.PROVIDER_ERROR, "failed", "waterfall"),
    ]

    async def execute(
        _engine: object,
        _url: str,
        *,
        provider: str | None = None,
        skip_providers: object = None,
    ) -> FetchResponse:
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr("omnifetch.tools.fetch.execute_web_fetch", execute)
    composition = await build_composition_async(load_config())
    assert composition.trace_sink is not None
    composition.trace_sink._sync_uploader = uploaded.append
    async with Client(composition.server) as client:
        success = await client.call_tool(
            "web_fetch", {"url": "https://example.test/article"}
        )
        terminal = await client.call_tool(
            "web_fetch", {"url": "https://example.test/missing"}
        )
        with pytest.raises(ToolError):
            await client.call_tool("web_fetch", {"url": ""})
    assert success.structured_content["status"] == "success"
    assert terminal.structured_content["status"] == "unavailable"
    documents = [json.loads(item.body) for item in uploaded]
    assert len(documents) == 3
    assert all(document["tool"] == "web_fetch" for document in documents)
    documents_by_outcome = {
        document["final_result"].get("status")
        or document["final_result"].get("error"): document
        for document in documents
    }
    assert set(documents_by_outcome) == {
        "success",
        "unavailable",
        "ValidationError",
    }
    success_document = documents_by_outcome["success"]
    unavailable_document = documents_by_outcome["unavailable"]
    validation_document = documents_by_outcome["ValidationError"]
    assert success_document["providers_succeeded"] == ["beta"]
    assert unavailable_document["final_result"]["status"] == "unavailable"
    assert validation_document["final_result"] == {"error": "ValidationError"}


def test_rest_fetch_traces_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_trace_environment(monkeypatch)
    uploaded: list[_PreparedTrace] = []
    exhaustion = FetchExhaustionDetails(
        providers_attempted=("alpha", "beta"),
        providers_failed=(
            ProviderAttemptFailure(
                provider="alpha",
                error="failed",
                duration_ms=4,
                error_type=ErrorType.API_ERROR,
            ),
            ProviderAttemptFailure(
                provider="beta",
                error="missing",
                duration_ms=7,
                error_type=ErrorType.NOT_FOUND,
            ),
        ),
    )
    responses: list[FetchResponse | BaseException] = [
        _fetch_response(alternative_results=None),
        ProviderError(
            ErrorType.API_ERROR,
            "failed",
            "waterfall",
            details=exhaustion,
        ),
        RuntimeError("unhandled message"),
    ]

    async def execute(
        _engine: object,
        _url: str,
        *,
        skip_providers: object = None,
    ) -> FetchResponse:
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr("omnifetch.tools.fetch.execute_web_fetch", execute)
    composition = build_composition(load_config())
    assert composition.trace_sink is not None
    composition.trace_sink._sync_uploader = uploaded.append
    with TestClient(composition.server.http_app()) as client:
        success = client.post(
            "/fetch", json={"url": "https://example.test/article"}
        )
        failure = client.post(
            "/fetch", json={"url": "https://example.test/failure"}
        )
        with pytest.raises(RuntimeError, match="unhandled message"):
            client.post(
                "/fetch", json={"url": "https://example.test/unhandled"}
            )
    assert success.status_code == 200
    assert failure.status_code == 502
    documents = [json.loads(item.body) for item in uploaded]
    assert len(documents) == 3
    assert all(
        item["request_environment"]["transport"] == "rest" for item in documents
    )
    documents_by_url = {
        item["request_environment"]["arguments"]["url"]: item
        for item in documents
    }
    success_document = documents_by_url["https://example.test/article"]
    failure_document = documents_by_url["https://example.test/failure"]
    unhandled_document = documents_by_url["https://example.test/unhandled"]
    assert success_document["providers_succeeded"] == ["beta"]
    assert failure_document["providers_hit"] == ["alpha", "beta"]
    assert failure_document["providers_succeeded"] == []
    assert len(failure_document["providers_failed"]) == 2
    assert failure_document["providers"]["alpha"]["error"] == "failed"
    assert failure_document["providers"]["beta"]["error"] == "missing"
    assert failure_document["final_result"] == {"error": "ProviderError"}
    assert unhandled_document["final_result"] == {"error": "RuntimeError"}

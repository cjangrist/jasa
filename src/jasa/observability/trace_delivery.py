"""Serialize redacted traces and deliver them to S3-compatible storage."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections import deque
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast
from urllib.parse import parse_qsl, unquote, urlsplit

from pydantic import BaseModel

from jasa.config import TraceSettings
from jasa.logging import get_logger
from jasa.observability.traces import (
    _decode_body,
    _is_http_url,
    _iso_timestamp,
    _redact,
    _sensitive_name,
    _utc_now,
    FetchTrace,
    HttpCallRecord,
    ProviderRecord,
    SearchTrace,
)

_LOGGER = get_logger("observability.trace_delivery")
_SENTINEL = object()
_MINIMUM_SECRET_LENGTH = 4
_MAX_SNAPSHOT_BYTES = 1024 * 1024
_MAX_SERIALIZED_TRACE_BYTES = 8 * 1024 * 1024
_MAX_QUEUED_TRACE_BYTES = 32 * 1024 * 1024
_MAX_SNAPSHOT_STRING_BYTES = 256 * 1024
_MAX_SNAPSHOT_CONTENT_BYTES = 64 * 1024
_MAX_PARTIAL_JSON_VALUES = 128
_MAX_PARTIAL_JSON_VALUE_BYTES = 64 * 1024
_MAX_PARTIAL_JSON_NESTING_DEPTH = 256
_JSON_UNICODE_ESCAPE_DIGITS = 4
_HIGH_SURROGATE_MINIMUM = 0xD800
_HIGH_SURROGATE_MAXIMUM = 0xDBFF
_LOW_SURROGATE_MINIMUM = 0xDC00
_LOW_SURROGATE_MAXIMUM = 0xDFFF
_DEFERRED_MODEL_FIELDS = frozenset({"content", "metadata"})
_REDACTED = "[REDACTED]"
_TRUNCATED = "[TRUNCATED]"
_PROTECTED_SENTINELS = (_REDACTED, _TRUNCATED)
_SecretMatcher = tuple[
    tuple[dict[str, int], ...], tuple[int, ...], tuple[int, ...]
]


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
    snapshot_truncated: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FetchTraceEnvelope:
    """Completed fetch trace waiting for off-loop serialization and upload."""

    trace: FetchTrace
    final_result: object
    completed_at: datetime
    captured_response_bytes: int = 0
    snapshot_truncated: bool = False
    error: str | None = None


TraceEnvelopeRecord = TraceEnvelope | FetchTraceEnvelope


@dataclass(frozen=True, slots=True)
class _PreparedTrace:
    """Redacted, capped JSON ready for an off-loop S3 upload."""

    trace_id: str
    completed_at: datetime
    body: bytes
    tool: str = "web_search"


@dataclass(slots=True)
class _SnapshotBudget:
    """Bound retained trace data before it reaches the delivery queue."""

    remaining_bytes: int = _MAX_SNAPSHOT_BYTES
    truncated: bool = False

    def consume(self, size_bytes: int) -> bool:
        if size_bytes <= self.remaining_bytes:
            self.remaining_bytes -= size_bytes
            return True
        self.truncated = True
        return False


def _snapshot_string(
    value: str,
    budget: _SnapshotBudget,
    maximum_bytes: int = _MAX_SNAPSHOT_STRING_BYTES,
) -> str:
    encoded = value.encode("utf-8")
    retained_limit = min(len(encoded), budget.remaining_bytes, maximum_bytes)
    if retained_limit == len(encoded) and budget.consume(len(encoded) + 2):
        return value
    budget.truncated = True
    marker = _TRUNCATED.encode("utf-8")
    content_limit = max(0, retained_limit - len(marker))
    retained = encoded[:content_limit].decode("utf-8", errors="ignore")
    budget.consume(min(budget.remaining_bytes, retained_limit + 2))
    return retained + _TRUNCATED


def _snapshot_bytes(value: bytes, budget: _SnapshotBudget) -> bytes:
    retained_limit = min(
        len(value), budget.remaining_bytes, _MAX_SNAPSHOT_STRING_BYTES
    )
    if retained_limit == len(value) and budget.consume(len(value) + 2):
        return value
    budget.truncated = True
    budget.consume(min(budget.remaining_bytes, retained_limit + 2))
    return value[:retained_limit]


def _snapshot_model(value: BaseModel, budget: _SnapshotBudget) -> object:
    budget.consume(16)
    declared_names = tuple(type(value).model_fields)
    ordered_names = tuple(
        name for name in declared_names if name not in _DEFERRED_MODEL_FIELDS
    ) + tuple(name for name in declared_names if name in _DEFERRED_MODEL_FIELDS)
    snapshot: dict[str, object] = {}
    for name in ordered_names:
        if budget.remaining_bytes <= 0:
            budget.truncated = True
            break
        field_value = getattr(value, name)
        snapshot[name] = (
            _snapshot_string(field_value, budget, _MAX_SNAPSHOT_CONTENT_BYTES)
            if name == "content" and isinstance(field_value, str)
            else _bounded_snapshot(field_value, budget)
        )
    return {name: snapshot[name] for name in declared_names if name in snapshot}


def _bounded_snapshot(value: object, budget: _SnapshotBudget) -> object:
    if isinstance(value, BaseModel):
        return _snapshot_model(value, budget)
    return _bounded_non_model_snapshot(value, budget)


def _snapshot_mapping_item(
    snapshot: dict[object, object],
    key: object,
    item: object,
    budget: _SnapshotBudget,
) -> None:
    snapshot_key = _bounded_snapshot(key, budget)
    snapshot[snapshot_key] = (
        _snapshot_string(item, budget, _MAX_SNAPSHOT_CONTENT_BYTES)
        if str(key) == "content" and isinstance(item, str)
        else _bounded_snapshot(item, budget)
    )
    budget.consume(2)


def _snapshot_mapping(
    value: Mapping[object, object], budget: _SnapshotBudget
) -> dict[object, object]:
    budget.consume(2)
    snapshot: dict[object, object] = {}
    deferred_items: list[tuple[bool, object, object]] = []
    for key, item in value.items():
        if budget.remaining_bytes <= 0:
            budget.truncated = True
            break
        key_name = str(key)
        if key_name in _DEFERRED_MODEL_FIELDS:
            if len(deferred_items) >= len(_DEFERRED_MODEL_FIELDS):
                budget.truncated = True
                break
            deferred_items.append((key_name == "content", key, item))
            continue
        _snapshot_mapping_item(snapshot, key, item, budget)
    for _, key, item in sorted(deferred_items, key=lambda entry: entry[0]):
        if budget.remaining_bytes <= 0:
            budget.truncated = True
            break
        _snapshot_mapping_item(snapshot, key, item, budget)
    return snapshot


def _bounded_non_model_snapshot(
    value: object, budget: _SnapshotBudget
) -> object:
    if isinstance(value, str):
        snapshot: object = _snapshot_string(value, budget)
    elif isinstance(value, bytes):
        snapshot = _snapshot_bytes(value, budget)
    elif isinstance(value, bytearray):
        snapshot = _snapshot_bytes(bytes(value), budget)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        budget.consume(16)
        snapshot = type(value)(
            **{
                item.name: _bounded_snapshot(getattr(value, item.name), budget)
                for item in dataclasses.fields(value)
                if item.init
            }
        )
    elif isinstance(value, Mapping):
        snapshot = _snapshot_mapping(value, budget)
    elif isinstance(value, Sequence):
        budget.consume(2)
        snapshot_items: list[object] = []
        for item in value:
            if budget.remaining_bytes == 0:
                budget.truncated = True
                break
            snapshot_items.append(_bounded_snapshot(item, budget))
            budget.consume(1)
        snapshot = (
            tuple(snapshot_items)
            if isinstance(value, tuple)
            else snapshot_items
        )
    elif value is None or isinstance(
        value, bool | int | float | Enum | datetime
    ):
        budget.consume(len(str(value)) + 1)
        snapshot = value
    else:
        budget.truncated = True
        snapshot = f"[UNSERIALIZABLE:{type(value).__name__}]"
    return snapshot


def _snapshot_http_call(
    call: HttpCallRecord, budget: _SnapshotBudget
) -> HttpCallRecord:
    method = cast(str, _bounded_snapshot(call.method, budget))
    url = cast(str, _bounded_snapshot(call.url, budget))
    request_headers = cast(
        dict[str, str], _bounded_snapshot(call.request_headers, budget)
    )
    request_body = cast(
        bytes | None, _bounded_snapshot(call.request_body, budget)
    )
    response_headers = cast(
        dict[str, str], _bounded_snapshot(call.response_headers, budget)
    )
    source_response_body = _retained_response_body(call)
    response_body = cast(
        bytes | None, _bounded_snapshot(source_response_body, budget)
    )
    response_was_truncated = (
        source_response_body is not None
        and response_body is not None
        and len(response_body) < len(source_response_body)
    )
    return HttpCallRecord(
        timestamp=call.timestamp,
        started_monotonic=call.started_monotonic,
        method=method,
        url=url,
        request_headers=request_headers,
        request_body=request_body,
        response_status=call.response_status,
        response_headers=response_headers,
        response_body=response_body,
        response_size_bytes=call.response_size_bytes,
        response_body_truncated=(
            call.response_body_truncated or response_was_truncated
        ),
        duration_ms=call.duration_ms,
        error=cast(str | None, _bounded_snapshot(call.error, budget)),
    )


def _retained_response_body(call: HttpCallRecord) -> bytes | None:
    if call.response_body is not None:
        return call.response_body
    if call._response_body_chunks:
        return b"".join(call._response_body_chunks)
    return None


def _snapshot_provider(
    record: ProviderRecord, budget: _SnapshotBudget
) -> ProviderRecord:
    provider_input = _bounded_snapshot(record.input, budget)
    provider_output = _bounded_snapshot(record.output, budget)
    provider_error = cast(str | None, _bounded_snapshot(record.error, budget))
    http_calls: list[HttpCallRecord] = []
    for call in record.http_calls:
        if budget.remaining_bytes == 0:
            budget.truncated = True
            break
        http_calls.append(_snapshot_http_call(call, budget))
    return ProviderRecord(
        started_at=record.started_at,
        input=provider_input,
        duration_ms=record.duration_ms,
        success=record.success,
        output=provider_output,
        error=provider_error,
        http_calls=http_calls,
    )


def _snapshot_envelope(envelope: TraceEnvelopeRecord) -> TraceEnvelopeRecord:
    budget = _SnapshotBudget(_MAX_SNAPSHOT_BYTES)
    trace = envelope.trace
    if isinstance(trace, SearchTrace):
        snapshot_trace: SearchTrace | FetchTrace = SearchTrace(
            query=cast(str, _bounded_snapshot(trace.query, budget)),
            active_providers=cast(
                list[str], _bounded_snapshot(trace.active_providers, budget)
            ),
            trace_id=trace.trace_id,
            started_at=trace.started_at,
            parent_trace_id=cast(
                str | None, _bounded_snapshot(trace.parent_trace_id, budget)
            ),
            cache_hit=trace.cache_hit,
            providers={
                cast(str, _bounded_snapshot(name, budget)): _snapshot_provider(
                    record, budget
                )
                for name, record in trace.providers.items()
                if budget.remaining_bytes > 0
            },
            decisions=cast(
                list[Any], _bounded_snapshot(trace.decisions, budget)
            ),
            captured_response_bytes=min(
                trace.captured_response_bytes, _MAX_SNAPSHOT_BYTES
            ),
            orchestrator_strategy=cast(
                str, _bounded_snapshot(trace.orchestrator_strategy, budget)
            ),
        )
    else:
        snapshot_trace = FetchTrace(
            request_environment=_bounded_snapshot(
                trace.request_environment, budget
            ),
            active_providers=cast(
                tuple[str, ...],
                _bounded_snapshot(trace.active_providers, budget),
            ),
            transport=cast(str, _bounded_snapshot(trace.transport, budget)),
            trace_id=trace.trace_id,
            started_at=trace.started_at,
            parent_trace_id=cast(
                str | None, _bounded_snapshot(trace.parent_trace_id, budget)
            ),
            orchestrator_strategy=cast(
                str, _bounded_snapshot(trace.orchestrator_strategy, budget)
            ),
        )
    final_result = _bounded_snapshot(envelope.final_result, budget)
    error = cast(str | None, _bounded_snapshot(envelope.error, budget))
    captured_response_bytes = min(
        envelope.captured_response_bytes, _MAX_SNAPSHOT_BYTES
    )
    if isinstance(snapshot_trace, FetchTrace):
        return FetchTraceEnvelope(
            snapshot_trace,
            final_result,
            envelope.completed_at,
            captured_response_bytes,
            budget.truncated,
            error,
        )
    return TraceEnvelope(
        snapshot_trace,
        final_result,
        envelope.completed_at,
        captured_response_bytes,
        budget.truncated,
        error,
    )


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
            if item.name != "deadline_exceeded"
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return {
            name: _jsonable(getattr(value, name))
            for name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [_jsonable(item) for item in value]
    return value


def _sensitive_values(value: object) -> set[str]:
    if isinstance(value, str) and _is_http_url(value):
        return _url_sensitive_values(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            secret
            for item in dataclasses.fields(value)
            for secret in (
                _sensitive_string_values(item.name, getattr(value, item.name))
                | _sensitive_values(getattr(value, item.name))
            )
        }
    if isinstance(value, BaseModel):
        return {
            secret
            for name in type(value).model_fields
            for secret in (
                _sensitive_string_values(name, getattr(value, name))
                | _sensitive_values(getattr(value, name))
            )
        }
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
    if not _sensitive_name(name):
        return set()
    candidates = _string_leaves(value)
    normalized_name = "".join(
        character for character in str(name).lower() if character.isalnum()
    )
    if normalized_name in {"authorization", "proxyauthorization"}:
        for candidate in tuple(candidates):
            scheme_and_value = candidate.split(maxsplit=1)
            if scheme_and_value[1:]:
                payload = scheme_and_value[-1].strip()
                candidates.add(payload)
                for parameter in payload.split(","):
                    _, separator, parameter_value = parameter.partition("=")
                    if separator:
                        unquoted = parameter_value.strip().strip("\"'")
                        candidates.update(
                            {unquoted, unquoted.partition("/")[0]}
                        )
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


def _string_leaves(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            leaf
            for item in dataclasses.fields(value)
            for leaf in _string_leaves(getattr(value, item.name))
        }
    if isinstance(value, BaseModel):
        return {
            leaf
            for name in type(value).model_fields
            for leaf in _string_leaves(getattr(value, name))
        }
    if isinstance(value, Mapping):
        return {
            leaf for item in value.values() for leaf in _string_leaves(item)
        }
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return {leaf for item in value for leaf in _string_leaves(item)}
    return set()


def _url_sensitive_values(raw_url: str) -> set[str]:
    """Return URL credentials and signatures that need global scrubbing."""
    try:
        parts = urlsplit(raw_url.strip())
    except ValueError:
        return set()
    userinfo_values = {
        value
        for item in (parts.username, parts.password)
        if item is not None
        for value in (item, unquote(item))
        if len(value) >= _MINIMUM_SECRET_LENGTH
    }
    query_values = _sensitive_url_parameter_values(parts.query)
    fragment_parameters = (
        parts.fragment.split("?", maxsplit=1)[-1] if parts.fragment else ""
    ).lstrip("/")
    fragment_values = _sensitive_url_parameter_values(fragment_parameters)
    return userinfo_values | query_values | fragment_values


def _partial_json_string_variants(raw_value: str) -> set[str]:
    candidates = {raw_value}
    try:
        decoded_value = json.loads(f'"{raw_value}"')
    except json.JSONDecodeError:
        decoded_value = _decoded_prefix_before_incomplete_unicode(raw_value)
    if isinstance(decoded_value, str):
        candidates.add(decoded_value)
        encodable_prefix = _utf8_encodable_prefix(decoded_value)
        if encodable_prefix is not None:
            candidates.add(encodable_prefix)
    return {
        candidate
        for candidate in candidates
        if len(candidate) >= _MINIMUM_SECRET_LENGTH
        and _is_utf8_encodable(candidate)
    }


def _decoded_prefix_before_incomplete_unicode(raw_value: str) -> str | None:
    index = 0
    pending_high_surrogate: int | None = None
    invalid_unicode_start: int | None = None
    while index < len(raw_value):
        if raw_value[index] != "\\":
            pending_high_surrogate, invalid_unicode_start = (
                _close_pending_high_surrogate(
                    pending_high_surrogate, invalid_unicode_start
                )
            )
            index += 1
            continue
        if index + 1 >= len(raw_value):
            return None
        if raw_value[index + 1] != "u":
            pending_high_surrogate, invalid_unicode_start = (
                _close_pending_high_surrogate(
                    pending_high_surrogate, invalid_unicode_start
                )
            )
            index += 2
            continue
        escape_end = index + 6
        digits = raw_value[index + 2 : escape_end]
        if escape_end > len(raw_value) and all(
            character in "0123456789abcdefABCDEF" for character in digits
        ):
            prefix_end = (
                invalid_unicode_start
                if invalid_unicode_start is not None
                else pending_high_surrogate
                if pending_high_surrogate is not None
                else index
            )
            try:
                prefix = json.loads(f'"{raw_value[:prefix_end]}"')
            except json.JSONDecodeError:
                return None
            return cast(str, prefix)
        pending_high_surrogate, invalid_unicode_start = _unicode_escape_state(
            digits, index, pending_high_surrogate, invalid_unicode_start
        )
        index = escape_end
    return None


def _close_pending_high_surrogate(
    pending: int | None, invalid: int | None
) -> tuple[None, int | None]:
    if pending is not None and invalid is None:
        invalid = pending
    return None, invalid


def _unicode_escape_state(
    digits: str,
    index: int,
    pending: int | None,
    invalid: int | None,
) -> tuple[int | None, int | None]:
    if len(digits) != _JSON_UNICODE_ESCAPE_DIGITS or not all(
        character in "0123456789abcdefABCDEF" for character in digits
    ):
        return _close_pending_high_surrogate(pending, invalid)
    code_unit = int(digits, 16)
    if _LOW_SURROGATE_MINIMUM <= code_unit <= _LOW_SURROGATE_MAXIMUM:
        if pending is None and invalid is None:
            invalid = index
        return None, invalid
    if _HIGH_SURROGATE_MINIMUM <= code_unit <= _HIGH_SURROGATE_MAXIMUM:
        _, invalid = _close_pending_high_surrogate(pending, invalid)
        return index, invalid
    return _close_pending_high_surrogate(pending, invalid)


def _is_utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _utf8_encodable_prefix(value: str) -> str | None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        return value[: error.start]
    return None


def _add_partial_json_variants(values: set[str], raw_value: str) -> None:
    additions = _partial_json_string_variants(raw_value) - values
    if (
        len(values) + len(additions) > _MAX_PARTIAL_JSON_VALUES
        or sum(len(value.encode("utf-8")) for value in values | additions)
        > _MAX_PARTIAL_JSON_VALUE_BYTES
    ):
        raise ValueError("partial JSON value scrub limit exceeded")
    values.update(additions)


def _json_value_expected(stack: list[tuple[str, str]], root_state: str) -> bool:
    if not stack:
        return root_state == "value"
    kind, state = stack[-1]
    return state == ("value" if kind == "object" else "value_or_end")


def _consume_json_value(stack: list[tuple[str, str]], root_state: str) -> str:
    if not stack:
        return "end"
    kind, _ = stack[-1]
    stack[-1] = (kind, "comma_or_end")
    return root_state


def _advance_json_structure(
    token: str, stack: list[tuple[str, str]], root_state: str
) -> str:
    if token in {"{", "["}:
        if len(stack) >= _MAX_PARTIAL_JSON_NESTING_DEPTH:
            raise ValueError("partial JSON nesting limit exceeded")
        if _json_value_expected(stack, root_state):
            root_state = _consume_json_value(stack, root_state)
        stack.append(
            ("object", "key_or_end")
            if token == "{"
            else ("array", "value_or_end")
        )
    elif token in {"}", "]"}:
        expected_kind = "object" if token == "}" else "array"
        if stack and stack[-1][0] == expected_kind:
            stack.pop()
    elif token == ":" and stack and stack[-1] == ("object", "colon"):
        stack[-1] = ("object", "value")
    elif token == "," and stack:
        kind, _ = stack[-1]
        stack[-1] = (
            ("object", "key_or_end")
            if kind == "object"
            else ("array", "value_or_end")
        )
    return root_state


def _record_partial_json_string(
    values: set[str],
    raw_value: str,
    stack: list[tuple[str, str]],
    root_state: str,
    followed_by_colon: bool,
) -> str:
    if stack and stack[-1] == ("object", "key_or_end") and followed_by_colon:
        stack[-1] = ("object", "colon")
        return root_state
    _add_partial_json_variants(values, raw_value)
    if _json_value_expected(stack, root_state):
        return _consume_json_value(stack, root_state)
    return root_state


def _followed_by_colon(text: str, index: int) -> bool:
    while index < len(text) and text[index].isspace():
        index += 1
    return index < len(text) and text[index] == ":"


def _partial_json_tokens(text: str) -> Iterator[tuple[str, str, bool, bool]]:
    index = 0
    while index < len(text):
        character = text[index]
        if character in "{}[],:":
            yield "structure", character, False, False
            index += 1
            continue
        if character != '"':
            start = index
            while index < len(text) and text[index] not in '"{}[],:':
                index += 1
            yield (
                "literal",
                text[start:index],
                False,
                _followed_by_colon(text, index),
            )
            continue
        start = index + 1
        index = start
        while index < len(text):
            if text[index] == '"':
                yield (
                    "string",
                    text[start:index],
                    False,
                    _followed_by_colon(text, index + 1),
                )
                index += 1
                break
            if text[index] == "\\":
                if index + 1 == len(text):
                    yield "unclosed", text[start:index], True, False
                    return
                index += 2
                continue
            index += 1
        else:
            yield "unclosed", text[start:index], False, False
            return


def _add_malformed_truncated_json_text_values(
    values: set[str], text: str
) -> None:
    stack: list[tuple[str, str]] = []
    root_state = "value"
    for kind, token, dangling, followed_by_colon in _partial_json_tokens(text):
        if kind == "structure":
            root_state = _advance_json_structure(token, stack, root_state)
        elif kind == "literal":
            if not token.strip():
                continue
            if (
                stack
                and stack[-1] == ("object", "key_or_end")
                and followed_by_colon
            ):
                stack[-1] = ("object", "colon")
            elif _json_value_expected(stack, root_state):
                root_state = _consume_json_value(stack, root_state)
        elif kind == "string":
            root_state = _record_partial_json_string(
                values, token, stack, root_state, followed_by_colon
            )
        else:
            _add_partial_json_variants(values, token)
            if dangling:
                _add_partial_json_variants(values, token + "\\")


def _strict_utf8_prefix(body: bytes) -> str | None:
    try:
        body.decode("utf-8")
    except UnicodeDecodeError as error:
        return body[: error.start].decode("utf-8")
    return None


def _add_malformed_truncated_json_values(
    values: set[str], body: bytes | None
) -> None:
    if not body:
        return
    text = body.decode("utf-8", errors="replace")
    if not isinstance(_decode_body(body), str):
        return
    _add_malformed_truncated_json_text_values(values, text)
    utf8_prefix = _strict_utf8_prefix(body)
    if utf8_prefix is not None:
        _add_malformed_truncated_json_text_values(values, utf8_prefix)


def _malformed_truncated_json_values(body: bytes | None) -> set[str]:
    """Conservatively scrub every value string from partial JSON bodies."""
    values: set[str] = set()
    _add_malformed_truncated_json_values(values, body)
    return values


def _sensitive_url_parameter_values(raw_parameters: str) -> set[str]:
    """Return decoded sensitive values from query-style URL parameters."""
    return {
        value
        for key, item in parse_qsl(raw_parameters, keep_blank_values=True)
        if _sensitive_name(key)
        for value in (item, unquote(item))
        if len(value) >= _MINIMUM_SECRET_LENGTH
    }


def _build_secret_matcher(secrets: set[str]) -> _SecretMatcher | None:
    candidates = {secret for secret in secrets if secret}
    if not candidates:
        return None
    transitions: list[dict[str, int]] = [{}]
    failures = [0]
    output_lengths = [0]
    for secret in candidates:
        state = 0
        for character in secret:
            if character not in transitions[state]:
                transitions[state][character] = len(transitions)
                transitions.append({})
                failures.append(0)
                output_lengths.append(0)
            state = transitions[state][character]
        output_lengths[state] = max(output_lengths[state], len(secret))
    pending = deque(transitions[0].values())
    while pending:
        state = pending.popleft()
        for character, child in transitions[state].items():
            pending.append(child)
            fallback = failures[state]
            while fallback and character not in transitions[fallback]:
                fallback = failures[fallback]
            failures[child] = transitions[fallback].get(character, 0)
            output_lengths[child] = max(
                output_lengths[child], output_lengths[failures[child]]
            )
    return tuple(transitions), tuple(failures), tuple(output_lengths)


def _scrub_plain_text(value: str, matcher: _SecretMatcher) -> str:
    transitions, failures, output_lengths = matcher
    parts: list[str] = []
    preceding_end = 0
    merged_start: int | None = None
    merged_end = 0
    state = 0
    for end, character in enumerate(value, start=1):
        while state and character not in transitions[state]:
            state = failures[state]
        state = transitions[state].get(character, 0)
        match_length = output_lengths[state]
        if not match_length:
            continue
        start = end - match_length
        if merged_start is None:
            merged_start, merged_end = start, end
        elif start <= merged_end:
            merged_start = min(merged_start, start)
            merged_end = max(merged_end, end)
        else:
            parts.extend((value[preceding_end:merged_start], _REDACTED))
            preceding_end = merged_end
            merged_start, merged_end = start, end
    if merged_start is None:
        return value
    parts.extend(
        (value[preceding_end:merged_start], _REDACTED, value[merged_end:])
    )
    return "".join(parts)


def _scrub_text(
    value: str,
    matcher: _SecretMatcher | None,
    *,
    protect_sentinels: bool,
) -> str:
    if matcher is None:
        return value
    if not protect_sentinels:
        return _scrub_plain_text(value, matcher)
    parts: list[str] = []
    preceding_end = 0
    index = 0
    while index < len(value):
        sentinel = next(
            (
                candidate
                for candidate in _PROTECTED_SENTINELS
                if value.startswith(candidate, index)
            ),
            None,
        )
        if sentinel is None:
            index += 1
            continue
        parts.append(_scrub_plain_text(value[preceding_end:index], matcher))
        parts.append(sentinel)
        index += len(sentinel)
        preceding_end = index
    if not parts:
        return _scrub_plain_text(value, matcher)
    parts.append(_scrub_plain_text(value[preceding_end:], matcher))
    return "".join(parts)


def _scrub_with_matcher(
    value: object,
    matcher: _SecretMatcher | None,
    *,
    scrub_keys: bool,
    protect_sentinels: bool,
) -> object:
    if isinstance(value, str):
        return _scrub_text(value, matcher, protect_sentinels=protect_sentinels)
    if isinstance(value, Mapping):
        return {
            (
                _scrub_text(key, matcher, protect_sentinels=protect_sentinels)
                if scrub_keys and isinstance(key, str)
                else key
            ): _scrub_with_matcher(
                item,
                matcher,
                scrub_keys=scrub_keys,
                protect_sentinels=protect_sentinels,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [
            _scrub_with_matcher(
                item,
                matcher,
                scrub_keys=scrub_keys,
                protect_sentinels=protect_sentinels,
            )
            for item in value
        ]
    return value


def _scrub_strings(
    value: object, secrets: set[str], *, scrub_keys: bool = True
) -> object:
    sentinel_secrets = {
        secret
        for secret in secrets
        if any(
            _secret_intersects_sentinel_boundary(secret, sentinel)
            for sentinel in _PROTECTED_SENTINELS
        )
    }
    sentinel_scrubbed = _scrub_with_matcher(
        value,
        _build_secret_matcher(sentinel_secrets),
        scrub_keys=scrub_keys,
        protect_sentinels=False,
    )
    return _scrub_with_matcher(
        sentinel_scrubbed,
        _build_secret_matcher(secrets - sentinel_secrets),
        scrub_keys=scrub_keys,
        protect_sentinels=True,
    )


def _secret_intersects_sentinel_boundary(secret: str, sentinel: str) -> bool:
    if sentinel in secret:
        return True
    maximum_overlap = min(len(secret), len(sentinel)) - 1
    return any(
        (
            secret.startswith(sentinel[-overlap:])
            or secret.endswith(sentinel[:overlap])
        )
        for overlap in range(1, maximum_overlap + 1)
    )


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
        for value in (record.input, record.output)
        for secret in _sensitive_values(value)
    }
    call_secrets = {
        secret
        for call in record.http_calls
        for secret in _http_call_secrets(call)
    }
    return direct_secrets | call_secrets


def _trace_secrets(envelope: TraceEnvelopeRecord) -> set[str]:
    trace = envelope.trace
    if isinstance(trace, FetchTrace):
        return _sensitive_values(trace.request_environment) | _sensitive_values(
            envelope.final_result
        )
    provider_secrets = {
        secret
        for record in trace.providers.values()
        for secret in _provider_secrets(record)
    }
    result_secrets = _sensitive_values(envelope.final_result)
    decision_secrets = {
        secret
        for decision in trace.decisions
        for secret in _sensitive_values(decision.details)
    }
    return provider_secrets | result_secrets | decision_secrets


def _trace_partial_body_values(envelope: TraceEnvelopeRecord) -> set[str]:
    if isinstance(envelope.trace, FetchTrace):
        return set()
    values: set[str] = set()
    for record in envelope.trace.providers.values():
        for call in record.http_calls:
            if call.response_body_truncated:
                _add_malformed_truncated_json_values(
                    values, _retained_response_body(call)
                )
    return values


def _search_trace_document(
    envelope: TraceEnvelopeRecord,
    configured_secrets: Collection[str] = (),
    partial_body_values: Collection[str] = (),
) -> dict[str, object]:
    trace = cast(SearchTrace, envelope.trace)
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
    if envelope.snapshot_truncated:
        document["trace_truncated"] = True
    full_secrets = _trace_secrets(envelope) | set(configured_secrets)
    value_scrubbed = _scrub_strings(
        document,
        full_secrets | set(partial_body_values),
        scrub_keys=False,
    )
    return cast(
        dict[str, object],
        _scrub_strings(value_scrubbed, full_secrets),
    )


def _string_items(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(
        value, str | bytes | bytearray
    ):
        return []
    return [item for item in value if isinstance(item, str)]


def _fetch_result_fields(result: object) -> Mapping[str, object]:
    converted = _jsonable(result)
    return converted if isinstance(converted, Mapping) else {}


def _fetch_provider_summary(
    provider: str,
    trace: FetchTrace,
    succeeded: set[str],
    failures: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    failure = failures.get(provider)
    document: dict[str, object] = {
        "started_at": _iso_timestamp(trace.started_at),
        "duration_ms": failure.get("duration_ms", 0) if failure else 0,
        "success": provider in succeeded,
        "input": _redact(_jsonable(trace.request_environment)),
        "output": {"source_provider": provider}
        if provider in succeeded
        else None,
        "http_calls": [],
    }
    if failure is not None:
        document["error"] = failure.get("error", "unknown")
    return document


def _fetch_trace_document(
    envelope: TraceEnvelopeRecord, configured_secrets: Collection[str] = ()
) -> dict[str, object]:
    trace = cast(FetchTrace, envelope.trace)
    result = _fetch_result_fields(envelope.final_result)
    attempted = _string_items(result.get("providers_attempted"))
    source_provider = result.get("source_provider")
    alternatives = result.get("alternative_results")
    alternative_names = (
        [
            item.get("source_provider")
            for item in alternatives
            if isinstance(item, Mapping)
            and isinstance(item.get("source_provider"), str)
        ]
        if isinstance(alternatives, Sequence)
        else []
    )
    succeeded_names = list(
        dict.fromkeys(
            name
            for name in [source_provider, *alternative_names]
            if isinstance(name, str) and name
        )
    )
    succeeded = set(succeeded_names)
    raw_failures = result.get("providers_failed")
    failure_items = (
        [item for item in raw_failures if isinstance(item, Mapping)]
        if isinstance(raw_failures, Sequence)
        and not isinstance(raw_failures, str | bytes | bytearray)
        else []
    )
    failures_by_provider = {
        cast(str, item["provider"]): item
        for item in failure_items
        if isinstance(item.get("provider"), str)
    }
    providers_hit = list(dict.fromkeys([*attempted, *succeeded_names]))
    request_environment = {
        "transport": trace.transport,
        "arguments": _redact(_jsonable(trace.request_environment)),
    }
    final_result: object = envelope.final_result
    if envelope.error is not None:
        final_result = {"error": envelope.error}
    document = {
        "trace_id": trace.trace_id,
        "tool": "web_fetch",
        "parent_trace_id": trace.parent_trace_id,
        "started_at": _iso_timestamp(trace.started_at),
        "completed_at": _iso_timestamp(envelope.completed_at),
        "total_duration_ms": int(
            (envelope.completed_at - trace.started_at).total_seconds() * 1000
        ),
        "cache_hit": None,
        "provider_evidence": {
            "scope": "returned_result_origin",
            "current_request_execution": "unknown",
        },
        "request_environment": request_environment,
        "orchestrator": {
            "strategy": trace.orchestrator_strategy,
            "active_providers": list(trace.active_providers),
            "decisions": [],
        },
        "providers_hit": providers_hit,
        "providers_succeeded": succeeded_names,
        "providers_failed": failure_items,
        "providers": {
            name: _fetch_provider_summary(
                name, trace, succeeded, failures_by_provider
            )
            for name in providers_hit
        },
        "final_result": _redact(_jsonable(final_result)),
    }
    if envelope.snapshot_truncated:
        document["trace_truncated"] = True
    return cast(
        dict[str, object],
        _scrub_strings(
            document, _trace_secrets(envelope) | set(configured_secrets)
        ),
    )


def _trace_document(
    envelope: TraceEnvelopeRecord,
    configured_secrets: Collection[str] = (),
    partial_body_values: Collection[str] = (),
) -> dict[str, object]:
    if isinstance(envelope.trace, FetchTrace):
        return _fetch_trace_document(envelope, configured_secrets)
    return _search_trace_document(
        envelope, configured_secrets, partial_body_values
    )


def _prepare_trace(
    envelope: TraceEnvelopeRecord, configured_secrets: Collection[str] = ()
) -> _PreparedTrace:
    discovered_secrets = _trace_secrets(envelope) | set(configured_secrets)
    partial_body_values = _trace_partial_body_values(envelope)
    snapshot = _snapshot_envelope(envelope)
    if isinstance(snapshot.trace, SearchTrace):
        for record in snapshot.trace.providers.values():
            for call in record.http_calls:
                if call.response_body_truncated:
                    _add_malformed_truncated_json_values(
                        partial_body_values, _retained_response_body(call)
                    )
    body = json.dumps(
        _trace_document(snapshot, discovered_secrets, partial_body_values),
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")
    if len(body) > _MAX_SERIALIZED_TRACE_BYTES:
        raise ValueError("bounded trace exceeded serialized size limit")
    tool = _trace_tool(envelope.trace)
    return _PreparedTrace(
        envelope.trace.trace_id, envelope.completed_at, body, tool
    )


def _trace_tool(trace: SearchTrace | FetchTrace) -> str:
    return "web_fetch" if isinstance(trace, FetchTrace) else "web_search"


def _object_key(settings: TraceSettings, envelope: TraceEnvelopeRecord) -> str:
    completed_at = envelope.completed_at
    prefix = settings.prefix.strip("/")
    suffix = (
        f"tool={_trace_tool(envelope.trace)}/"
        f"date={completed_at:%Y-%m-%d}/"
        f"hour={completed_at:%H}/trace_id={envelope.trace.trace_id}.json"
    )
    return f"{prefix}/{suffix}" if prefix else suffix


def _prepared_object_key(
    settings: TraceSettings, prepared: _PreparedTrace
) -> str:
    prefix = settings.prefix.strip("/")
    suffix = (
        f"tool={prepared.tool}/date={prepared.completed_at:%Y-%m-%d}/"
        f"hour={prepared.completed_at:%H}/trace_id={prepared.trace_id}.json"
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
        sync_uploader: Callable[[_PreparedTrace], None] | None = None,
        configured_secrets: Collection[str] = (),
    ) -> None:
        """Create an inactive sink without starting network work."""
        self.settings = settings
        self._queue: asyncio.Queue[_PreparedTrace | object] = asyncio.Queue(
            maxsize=settings.queue_capacity
        )
        self._sync_uploader = sync_uploader
        self._client: object | None = None
        self._worker: asyncio.Task[None] | None = None
        self._preparations: set[asyncio.Task[None]] = set()
        self._closing = False
        self._dropped_submissions = 0
        self._accepted_submissions = 0
        self._accepted_trace_bytes = 0
        self._configured_secrets = frozenset(configured_secrets)

    def start(self) -> None:
        """Create the lightweight coordinator on the active event loop."""
        if self._worker is None:
            self._closing = False
            self._worker = asyncio.create_task(
                self._run(), name="jasa-s3-trace-uploader"
            )

    def submit(self, trace: SearchTrace, final_result: object) -> bool:
        """Reserve bounded delivery work without copying on the event loop."""
        envelope = TraceEnvelope(
            trace, final_result, _utc_now(), trace.captured_response_bytes
        )
        return self._submit_envelope(envelope)

    def submit_fetch(
        self,
        trace: FetchTrace,
        final_result: object,
        *,
        error: str | None = None,
    ) -> bool:
        """Reserve one public-fetch trace without serializing on the loop."""
        return self._submit_envelope(
            FetchTraceEnvelope(trace, final_result, _utc_now(), error=error)
        )

    def _submit_envelope(self, envelope: TraceEnvelopeRecord) -> bool:
        """Reserve bounded delivery work shared by search and fetch."""
        if self._worker is None or self._closing:
            return False
        submission_capacity = self.settings.queue_capacity + 1
        if (
            self._accepted_submissions >= submission_capacity
            or _MAX_QUEUED_TRACE_BYTES - self._accepted_trace_bytes
            < _MAX_SERIALIZED_TRACE_BYTES
        ):
            self._dropped_submissions += 1
            return False
        self._accepted_submissions += 1
        self._accepted_trace_bytes += _MAX_SERIALIZED_TRACE_BYTES
        if isinstance(envelope.trace, SearchTrace):
            envelope.trace.freeze()
        preparation = asyncio.create_task(
            self._prepare_and_enqueue(envelope),
            name=f"jasa-trace-snapshot-{envelope.trace.trace_id}",
        )
        self._preparations.add(preparation)
        preparation.add_done_callback(self._preparations.discard)
        return True

    async def close(self) -> None:
        """Drain accepted traces during orderly process shutdown."""
        worker = self._worker
        if worker is None:
            return
        self._closing = True
        if self._preparations:
            await asyncio.gather(*tuple(self._preparations))
        await self._queue.put(_SENTINEL)
        await worker
        self._worker = None
        await self._close_client()

    async def _prepare_and_enqueue(self, envelope: TraceEnvelopeRecord) -> None:
        try:
            prepared = await asyncio.to_thread(
                _prepare_trace, envelope, self._configured_secrets
            )
            del envelope
            await self._queue.put(prepared)
        except Exception as error:
            self._release_submission()
            self._dropped_submissions += 1
            await asyncio.to_thread(
                _LOGGER.warning,
                "Trace snapshot failed error_type=%s",
                type(error).__name__,
            )

    def _release_submission(self) -> None:
        self._accepted_submissions -= 1
        self._accepted_trace_bytes -= _MAX_SERIALIZED_TRACE_BYTES

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            accepted_submission = False
            try:
                if item is _SENTINEL:
                    return
                prepared = cast(_PreparedTrace, item)
                accepted_submission = True
                await asyncio.to_thread(self._upload_sync, prepared)
            except Exception as error:
                await asyncio.to_thread(
                    _LOGGER.warning,
                    "S3 trace upload failed error_type=%s",
                    type(error).__name__,
                )
            finally:
                self._queue.task_done()
                if accepted_submission:
                    self._release_submission()
                await self._report_dropped_submissions()

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            await asyncio.to_thread(cast(Any, client).close)
        except Exception as error:
            await asyncio.to_thread(
                _LOGGER.warning,
                "S3 trace client close failed error_type=%s",
                type(error).__name__,
            )

    async def _report_dropped_submissions(self) -> None:
        dropped_submissions = self._dropped_submissions
        self._dropped_submissions = 0
        if dropped_submissions:
            await asyncio.to_thread(
                _LOGGER.warning,
                "Trace queue saturation dropped_count=%s",
                dropped_submissions,
            )

    def _upload_sync(self, prepared: _PreparedTrace) -> None:
        if self._sync_uploader is not None:
            self._sync_uploader(prepared)
            return
        if self._client is None:
            self._client = _build_s3_client(self.settings)
        client = cast(Any, self._client)
        client.put_object(
            Bucket=self.settings.bucket,
            Key=_prepared_object_key(self.settings, prepared),
            Body=prepared.body,
            ContentType="application/json",
        )
        _LOGGER.debug("S3 trace uploaded trace_id=%s", prepared.trace_id)


def build_trace_sink(
    settings: TraceSettings,
    secret_environment: Mapping[str, str] | None = None,
) -> S3TraceSink | None:
    """Return an enabled sink after bootstrap validated its settings."""
    validate_trace_settings(settings)
    if not settings.enabled:
        return None
    destination_secrets = {
        "JASA_TRACE_S3_ACCESS_KEY_ID": settings.access_key_id,
        "JASA_TRACE_S3_SECRET_ACCESS_KEY": settings.secret_access_key,
    }
    configured_secrets = _sensitive_values(secret_environment or {})
    configured_secrets.update(_sensitive_values(destination_secrets))
    return S3TraceSink(settings, configured_secrets=configured_secrets)

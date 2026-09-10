"""Serialize redacted traces and deliver them to S3-compatible storage."""

from __future__ import annotations

import asyncio
import codecs
import dataclasses
import json
from collections import deque
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast
from urllib.parse import (
    parse_qsl,
    unquote,
    unquote_plus,
    urlsplit,
)

from pydantic import BaseModel

from jasa.config import TraceSettings
from jasa.logging import get_logger
from jasa.observability.traces import (
    _decode_body,
    _is_http_url,
    _iso_timestamp,
    _sanitize_url,
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
_MAX_SECRET_MATCHER_VALUES = 512
_MAX_SECRET_MATCHER_CHARACTERS = 128 * 1024
_MAX_URL_DECODE_PASSES = 8
_URL_ESCAPE_DIGITS = 2
_JSON_UNICODE_ESCAPE_DIGITS = 4
_HIGH_SURROGATE_MINIMUM = 0xD800
_HIGH_SURROGATE_MAXIMUM = 0xDBFF
_LOW_SURROGATE_MINIMUM = 0xDC00
_LOW_SURROGATE_MAXIMUM = 0xDFFF
_DEFERRED_MODEL_FIELDS = frozenset({"content", "metadata"})
_REDACTED = "[REDACTED]"
_URL_REDACTED = "%5BREDACTED%5D"
_TRUNCATED = "[TRUNCATED]"
_SecretMatcher = tuple[
    tuple[dict[str, int], ...], tuple[int, ...], tuple[int, ...]
]


class _TruncatedText(str):
    """String whose generated truncation suffix is not source content."""

    protected_start: int

    def __new__(cls, value: str, protected_start: int) -> _TruncatedText:
        instance = super().__new__(cls, value)
        instance.protected_start = protected_start
        return instance


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
    return _TruncatedText(retained + _TRUNCATED, len(retained))


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
        return {
            key if isinstance(key, str) else str(key): _jsonable(item)
            for key, item in value.items()
        }
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


def _source_text(value: str) -> str:
    return (
        value[: value.protected_start]
        if isinstance(value, _TruncatedText)
        else value
    )


def _string_leaves(value: object) -> set[str]:
    if isinstance(value, str):
        return {_source_text(value)}
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
    source = _source_text(raw_url)
    try:
        parts = urlsplit(source.strip())
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


def _add_partial_json_variants(
    values: set[str], raw_value: str, retained_bytes: int
) -> int:
    additions = _partial_json_string_variants(raw_value) - values
    if not additions:
        return retained_bytes
    added_bytes = sum(len(value.encode("utf-8")) for value in additions)
    if (
        len(values) + len(additions) > _MAX_PARTIAL_JSON_VALUES
        or retained_bytes + added_bytes > _MAX_PARTIAL_JSON_VALUE_BYTES
    ):
        raise ValueError("partial JSON value scrub limit exceeded")
    values.update(additions)
    return retained_bytes + added_bytes


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


def _advance_malformed_json_structure(
    token: str,
    stack: list[tuple[str, str]],
    sensitive_containers: list[bool],
    sensitive_key_depths: set[int],
    root_state: str,
) -> str:
    depth = len(stack)
    inherited_sensitivity = depth in sensitive_key_depths or (
        bool(sensitive_containers) and sensitive_containers[-1]
    )
    updated_root_state = _advance_json_structure(token, stack, root_state)
    if len(stack) > depth:
        sensitive_containers.append(inherited_sensitivity)
        sensitive_key_depths.discard(depth)
    elif len(stack) < depth:
        sensitive_containers.pop()
        sensitive_key_depths.discard(depth)
    elif token == ",":
        sensitive_key_depths.discard(depth)
    return updated_root_state


def _partial_value_is_sensitive(
    stack: list[tuple[str, str]],
    sensitive_containers: list[bool],
    sensitive_key_depths: set[int],
) -> bool:
    return len(stack) in sensitive_key_depths or (
        bool(sensitive_containers) and sensitive_containers[-1]
    )


def _add_malformed_truncated_json_text_values(
    values: set[str], text: str, retained_bytes: int
) -> int:
    stack: list[tuple[str, str]] = []
    sensitive_containers: list[bool] = []
    sensitive_key_depths: set[int] = set()
    root_state = "value"
    for kind, token, dangling, followed_by_colon in _partial_json_tokens(text):
        if kind == "structure":
            root_state = _advance_malformed_json_structure(
                token,
                stack,
                sensitive_containers,
                sensitive_key_depths,
                root_state,
            )
        elif kind == "literal":
            stripped_token = token.strip()
            if not stripped_token:
                continue
            if (
                stack
                and stack[-1] == ("object", "key_or_end")
                and followed_by_colon
            ):
                _set_sensitive_key_depth(
                    sensitive_key_depths, len(stack), stripped_token
                )
                stack[-1] = ("object", "colon")
            elif _json_value_expected(stack, root_state):
                if _partial_value_is_sensitive(
                    stack, sensitive_containers, sensitive_key_depths
                ) and _is_unquoted_secret_candidate(stripped_token):
                    retained_bytes = _add_partial_json_variants(
                        values, stripped_token, retained_bytes
                    )
                sensitive_key_depths.discard(len(stack))
                root_state = _consume_json_value(stack, root_state)
        elif kind == "string":
            if (
                stack
                and stack[-1] == ("object", "key_or_end")
                and followed_by_colon
            ):
                _set_sensitive_key_depth(
                    sensitive_key_depths, len(stack), token
                )
                stack[-1] = ("object", "colon")
                continue
            retained_bytes = _add_partial_json_variants(
                values, token, retained_bytes
            )
            if _json_value_expected(stack, root_state):
                sensitive_key_depths.discard(len(stack))
                root_state = _consume_json_value(stack, root_state)
        else:
            retained_bytes = _add_partial_json_variants(
                values, token, retained_bytes
            )
            if dangling:
                retained_bytes = _add_partial_json_variants(
                    values, token + "\\", retained_bytes
                )
    return retained_bytes


def _set_sensitive_key_depth(
    sensitive_key_depths: set[int], depth: int, raw_key: str
) -> None:
    candidates = {raw_key} | _partial_json_string_variants(raw_key)
    if any(map(_sensitive_name, candidates)):
        sensitive_key_depths.add(depth)
    else:
        sensitive_key_depths.discard(depth)


def _is_unquoted_secret_candidate(value: str) -> bool:
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return True
    return False


def _utf8_recovery_text(body: bytes) -> str | None:
    try:
        body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("utf-8", errors="ignore")
    return None


def _add_malformed_truncated_json_values(
    values: set[str], body: bytes | None, retained_bytes: int
) -> int:
    if not body:
        return retained_bytes
    text = body.decode("utf-8", errors="replace")
    recovery_text = _utf8_recovery_text(body)
    if recovery_text is None and not isinstance(_decode_body(body), str):
        return retained_bytes
    retained_bytes = _add_malformed_truncated_json_text_values(
        values, text, retained_bytes
    )
    if recovery_text is not None:
        retained_bytes = _add_malformed_truncated_json_text_values(
            values, recovery_text, retained_bytes
        )
    return retained_bytes


def _malformed_truncated_json_values(body: bytes | None) -> set[str]:
    """Conservatively scrub every value string from partial JSON bodies."""
    values: set[str] = set()
    _add_malformed_truncated_json_values(values, body, 0)
    return values


def _sensitive_url_parameter_values(raw_parameters: str) -> set[str]:
    """Return nested-decoded sensitive query-style parameter values."""
    return {
        value
        for field in raw_parameters.split("&")
        for key, separator, item in (field.partition("="),)
        if separator and _url_query_key_is_sensitive(key)
        for value in _decoded_url_parameter_values(item)
        if len(value) >= _MINIMUM_SECRET_LENGTH
    }


def _decoded_url_parameter_values(value: str) -> set[str]:
    values = {value}
    decoded = value
    for _ in range(_MAX_URL_DECODE_PASSES):
        next_decoded = unquote_plus(decoded)
        if next_decoded == decoded:
            break
        values.add(next_decoded)
        decoded = next_decoded
    if unquote_plus(decoded) != decoded:
        raise ValueError("sensitive URL parameter decode limit exceeded")
    return values


def _build_secret_matcher(secrets: set[str]) -> _SecretMatcher | None:
    candidates = {secret for secret in secrets if secret}
    if not candidates:
        return None
    if (
        len(candidates) > _MAX_SECRET_MATCHER_VALUES
        or sum(map(len, candidates)) > _MAX_SECRET_MATCHER_CHARACTERS
    ):
        raise ValueError("secret matcher input limit exceeded")
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


def _matching_secret_spans(
    value: str, matcher: _SecretMatcher
) -> list[tuple[int, int]]:
    transitions, failures, output_lengths = matcher
    spans: list[tuple[int, int]] = []
    state = 0
    protected_start = (
        value.protected_start
        if isinstance(value, _TruncatedText)
        else len(value)
    )
    for end, character in enumerate(value, start=1):
        while state and character not in transitions[state]:
            state = failures[state]
        state = transitions[state].get(character, 0)
        match_length = output_lengths[state]
        if not match_length:
            continue
        start = end - match_length
        if end > protected_start:
            continue
        merged_start = start
        while spans and merged_start <= spans[-1][1]:
            merged_start = min(merged_start, spans.pop()[0])
        spans.append((merged_start, end))
    return spans


def _scrub_plain_text(
    value: str,
    matcher: _SecretMatcher,
    replacement: str = _REDACTED,
) -> str:
    spans = _matching_secret_spans(value, matcher)
    if not spans:
        return value
    return _replace_spans(value, spans, replacement)


def _replace_spans(
    value: str, spans: list[tuple[int, int]], replacement: str
) -> str:
    """Replace sorted or unsorted source spans after merging overlaps."""
    spans.sort()
    parts: list[str] = []
    preceding_end = 0
    merged_start, merged_end = spans[0]
    for start, end in spans[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        parts.extend((value[preceding_end:merged_start], replacement))
        preceding_end = merged_end
        merged_start, merged_end = start, end
    parts.extend(
        (value[preceding_end:merged_start], replacement, value[merged_end:])
    )
    return "".join(parts)


def _combined_origin(
    origins: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    """Project an ordered contiguous decoded range from its boundaries."""
    return origins[start][0], origins[end - 1][1]


def _decode_url_component_layer(
    value: str,
    origins: Sequence[tuple[int, int]],
    *,
    plus_as_space: bool,
) -> tuple[str, list[tuple[int, int]], bool]:
    encoded = bytearray()
    byte_origins: list[tuple[int, int]] = []
    changed = False
    index = 0
    while index < len(value):
        character = value[index]
        digits = value[index + 1 : index + 3]
        if (
            character == "%"
            and len(digits) == _URL_ESCAPE_DIGITS
            and all(item in "0123456789abcdefABCDEF" for item in digits)
        ):
            encoded.append(int(digits, 16))
            byte_origins.append(_combined_origin(origins, index, index + 3))
            changed = True
            index += 3
            continue
        if plus_as_space and character == "+":
            encoded.append(ord(" "))
            byte_origins.append(origins[index])
            changed = True
            index += 1
            continue
        character_bytes = character.encode("utf-8", errors="replace")
        encoded.extend(character_bytes)
        byte_origins.extend([origins[index]] * len(character_bytes))
        index += 1
    if not changed:
        return value, list(origins), False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    decoded_characters: list[str] = []
    decoded_origins: list[tuple[int, int]] = []
    pending_origins: list[tuple[int, int]] = []
    for byte, origin in zip(encoded, byte_origins, strict=True):
        pending_origins.append(origin)
        emitted = decoder.decode(bytes((byte,)), final=False)
        if emitted:
            combined = _combined_origin(
                pending_origins, 0, len(pending_origins)
            )
            decoded_characters.extend(emitted)
            decoded_origins.extend([combined] * len(emitted))
            pending_origins.clear()
    emitted = decoder.decode(b"", final=True)
    if emitted:
        combined = _combined_origin(pending_origins, 0, len(pending_origins))
        decoded_characters.extend(emitted)
        decoded_origins.extend([combined] * len(emitted))
    return "".join(decoded_characters), decoded_origins, True


def _mapped_secret_spans(
    value: str,
    origins: Sequence[tuple[int, int]],
    matcher: _SecretMatcher,
) -> list[tuple[int, int]]:
    return [
        _combined_origin(origins, start, end)
        for start, end in _matching_secret_spans(value, matcher)
    ]


def _encoded_url_secret_spans(
    value: str, matcher: _SecretMatcher, *, plus_as_space: bool
) -> list[tuple[int, int]] | None:
    origins = [(index, index + 1) for index in range(len(value))]
    spans = _mapped_secret_spans(value, origins, matcher)
    decoded = value
    for _ in range(_MAX_URL_DECODE_PASSES):
        decoded, origins, changed = _decode_url_component_layer(
            decoded, origins, plus_as_space=plus_as_space
        )
        if not changed:
            return spans
        spans.extend(_mapped_secret_spans(decoded, origins, matcher))
    _, _, still_encoded = _decode_url_component_layer(
        decoded, origins, plus_as_space=plus_as_space
    )
    return None if still_encoded else spans


def _encoded_url_replacement_spans(
    value: str,
    matcher: _SecretMatcher | None,
    *,
    plus_as_space: bool = False,
) -> list[tuple[int, int]]:
    if matcher is None:
        return []
    spans = _encoded_url_secret_spans(
        value, matcher, plus_as_space=plus_as_space
    )
    return [(0, len(value))] if spans is None else spans


def _url_path_replacement_spans(
    path: str, matcher: _SecretMatcher | None
) -> list[tuple[int, int]]:
    spans = _encoded_url_replacement_spans(path, matcher)
    if not path.startswith("/"):
        return spans
    return [(max(1, start), end) for start, end in spans if end > 1]


def _url_query_key_is_sensitive(key: str) -> bool:
    decoded = key
    for _ in range(_MAX_URL_DECODE_PASSES):
        next_decoded = unquote_plus(decoded)
        if next_decoded == decoded:
            return _sensitive_name(decoded)
        decoded = next_decoded
    return (
        True if unquote_plus(decoded) != decoded else _sensitive_name(decoded)
    )


def _offset_spans(
    spans: Sequence[tuple[int, int]], offset: int
) -> list[tuple[int, int]]:
    return [(start + offset, end + offset) for start, end in spans]


def _url_query_replacement_spans(
    query: str, matcher: _SecretMatcher | None
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    field_offset = 0
    for field in query.split("&"):
        key, separator, item = field.partition("=")
        spans.extend(
            _offset_spans(
                _encoded_url_replacement_spans(
                    key, matcher, plus_as_space=True
                ),
                field_offset,
            )
        )
        item_offset = field_offset + len(key) + len(separator)
        item_spans = (
            [(0, len(item))]
            if separator and _url_query_key_is_sensitive(key)
            else _encoded_url_replacement_spans(
                item, matcher, plus_as_space=True
            )
        )
        spans.extend(_offset_spans(item_spans, item_offset))
        field_offset += len(field) + 1
    return spans


def _url_source_offsets(
    source: str, netloc: str, path: str
) -> tuple[int, int | None]:
    authority_start = source.find("//") + 2
    path_start = authority_start + len(netloc)
    query_marker = path_start + len(path)
    query_start = (
        query_marker + 1
        if source[query_marker : query_marker + 1] == "?"
        else None
    )
    return path_start, query_start


def _scrub_decoded_url_components(
    value: str, matcher: _SecretMatcher | None
) -> str:
    """Scrub encoded URL data without rewriting unmatched source spans."""
    source = _source_text(value)
    try:
        source = source.strip()
        parts = urlsplit(source)
        _ = parts.port
    except ValueError:
        return _REDACTED
    spans = _matching_secret_spans(source, matcher) if matcher else []
    path_start, query_start = _url_source_offsets(
        source, parts.netloc, parts.path
    )
    spans.extend(
        _offset_spans(
            _url_path_replacement_spans(parts.path, matcher), path_start
        )
    )
    if query_start is not None:
        spans.extend(
            _offset_spans(
                _url_query_replacement_spans(parts.query, matcher),
                query_start,
            )
        )
    scrubbed = _replace_spans(source, spans, _URL_REDACTED) if spans else source
    if not isinstance(value, _TruncatedText):
        return scrubbed
    suffix = value[value.protected_start :]
    return _TruncatedText(scrubbed + suffix, len(scrubbed))


def _sanitize_traced_url(value: str) -> str:
    """Sanitize a URL without treating a generated suffix as source text."""
    if not isinstance(value, _TruncatedText):
        return _sanitize_url(value)
    source = value[: value.protected_start]
    sanitized = _sanitize_url(source)
    if sanitized == _REDACTED:
        return sanitized
    return sanitized + value[value.protected_start :]


def _scrub_text(value: str, matcher: _SecretMatcher | None) -> str:
    if not _is_http_url(value):
        return value if matcher is None else _scrub_plain_text(value, matcher)
    component_scrubbed = _scrub_decoded_url_components(value, matcher)
    if not _is_http_url(component_scrubbed):
        return _REDACTED
    return _sanitize_traced_url(component_scrubbed)


def _scrub_with_matchers(
    value: object,
    value_matcher: _SecretMatcher | None,
    key_matcher: _SecretMatcher | None,
) -> object:
    if isinstance(value, str):
        return _scrub_text(value, value_matcher)
    if isinstance(value, Mapping):
        return {
            (_scrub_text(key, key_matcher) if isinstance(key, str) else key): (
                _REDACTED
                if _sensitive_name(key)
                else _scrub_with_matchers(item, value_matcher, key_matcher)
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [
            _scrub_with_matchers(item, value_matcher, key_matcher)
            for item in value
        ]
    return value


def _scrub_strings(
    value: object, secrets: set[str], *, scrub_keys: bool = True
) -> object:
    matcher = _build_secret_matcher(secrets)
    return _scrub_with_matchers(value, matcher, matcher if scrub_keys else None)


def _http_document(call: HttpCallRecord) -> dict[str, object]:
    document: dict[str, object] = {
        "timestamp": _iso_timestamp(call.timestamp),
        "method": call.method,
        "url": call.url,
        "request_headers": call.request_headers,
        "request_body": _decode_body(call.request_body),
        "response_status": call.response_status,
        "response_headers": call.response_headers,
        "response_body": _decode_body(call.response_body),
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
        "input": _jsonable(record.input),
        "output": _jsonable(record.output),
        "http_calls": [_http_document(call) for call in record.http_calls],
    }
    if record.error is not None:
        document["error"] = record.error
    return document


def _http_call_secrets(call: HttpCallRecord) -> set[str]:
    source_url = _source_text(call.url)
    structured_secrets = {
        secret
        for value in (
            call.request_headers,
            _decode_body(call.request_body),
            dict(parse_qsl(urlsplit(source_url).query, keep_blank_values=True)),
            call.response_headers,
            _decode_body(call.response_body),
        )
        for secret in _sensitive_values(value)
    }
    return structured_secrets | _url_sensitive_values(source_url)


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


def _trace_partial_body_values(
    envelope: TraceEnvelopeRecord,
) -> tuple[set[str], int]:
    if isinstance(envelope.trace, FetchTrace):
        return set(), 0
    values: set[str] = set()
    retained_bytes = 0
    for record in envelope.trace.providers.values():
        for call in record.http_calls:
            if call.response_body_truncated:
                retained_bytes = _add_malformed_truncated_json_values(
                    values,
                    _retained_response_body(call),
                    retained_bytes,
                )
    return values, retained_bytes


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
                    "details": _jsonable(item.details),
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
        "final_result": _jsonable(envelope.final_result),
    }
    if envelope.snapshot_truncated:
        document["trace_truncated"] = True
    full_secrets = _trace_secrets(envelope) | set(configured_secrets)
    scrubbed = _scrub_with_matchers(
        document,
        _build_secret_matcher(full_secrets | set(partial_body_values)),
        _build_secret_matcher(full_secrets),
    )
    return cast(dict[str, object], scrubbed)


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
        "input": _jsonable(trace.request_environment),
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
        "arguments": _jsonable(trace.request_environment),
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
        "final_result": _jsonable(final_result),
    }
    if envelope.snapshot_truncated:
        document["trace_truncated"] = True
    secrets = _trace_secrets(envelope) | set(configured_secrets)
    return cast(dict[str, object], _scrub_strings(document, secrets))


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
    partial_body_values, partial_body_value_bytes = _trace_partial_body_values(
        envelope
    )
    snapshot = _snapshot_envelope(envelope)
    if isinstance(snapshot.trace, SearchTrace):
        for record in snapshot.trace.providers.values():
            for call in record.http_calls:
                if call.response_body_truncated:
                    partial_body_value_bytes = (
                        _add_malformed_truncated_json_values(
                            partial_body_values,
                            _retained_response_body(call),
                            partial_body_value_bytes,
                        )
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

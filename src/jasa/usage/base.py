"""Usage-probe contract, bounded raw JSON requests, and redaction."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast, TypeAlias

import httpx

from omnifetch.fetch.shared.config import ProviderSecrets

JsonValue: TypeAlias = (
    bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
)
UsageFetcher: TypeAlias = Callable[
    [httpx.AsyncClient, ProviderSecrets],
    Awaitable[dict[str, JsonValue]],
]

REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RESPONSE_BYTES = 1_048_576
REDACTED = "[REDACTED]"
SENSITIVE_KEYS = frozenset(
    {
        "account_email",
        "account_id",
        "allowed_networks",
        "api_key",
        "business_id",
        "email",
        "id",
        "key",
        "name",
        "password",
        "secret",
        "token",
        "user",
        "user_id",
    }
)
_SECRET_ENV_SUFFIXES = (
    "_API_KEY",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
    "_USERNAME",
)


@dataclass(frozen=True, slots=True)
class UsageProbe:
    """One provider's free usage endpoint and its credential requirements."""

    required_secrets: tuple[str, ...]
    fetch: UsageFetcher


class UsageResponseError(Exception):
    """A non-success provider response retaining its cleaned raw dictionary."""

    def __init__(
        self,
        status_code: int,
        raw: dict[str, JsonValue],
    ) -> None:
        """Retain the upstream status and cleaned response dictionary."""
        super().__init__(f"usage endpoint returned HTTP {status_code}")
        self.status_code = status_code
        self.raw = raw


def _normalized_key(key: str) -> str:
    """Return a snake-like field name for conservative redaction checks."""
    with_boundaries = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    with_boundaries = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", with_boundaries)
    return re.sub(r"[^a-z0-9]+", "_", with_boundaries.lower()).strip("_")


def _sensitive_key(key: str) -> bool:
    """Return whether a field holds a credential or account identity."""
    normalized = _normalized_key(key)
    return normalized in SENSITIVE_KEYS or normalized.endswith("_id")


def secret_values(secrets: ProviderSecrets) -> tuple[str, ...]:
    """Return non-empty secret values without exposing their representation."""
    return tuple(
        value
        for name, value in secrets.values.items()
        if value and name.endswith(_SECRET_ENV_SUFFIXES)
    )


def redact_string(value: str, secrets: tuple[str, ...]) -> str:
    """Remove every raw and quote-stripped configured credential from text."""
    redacted = value
    candidates = {
        candidate
        for secret in secrets
        for candidate in (secret, secret.strip().strip('"').strip("'"))
        if candidate
    }
    for candidate in candidates:
        redacted = redacted.replace(candidate, REDACTED)
    return redacted


def _clean_sensitive_value(
    value: Any,
    secrets: tuple[str, ...],
) -> JsonValue:
    """Redact a sensitive value while retaining container structure."""
    if isinstance(value, Mapping):
        return {
            str(key): clean_provider_value(
                child,
                secrets,
                field_name=str(key),
            )
            for key, child in value.items()
        }
    if isinstance(value, list | tuple):
        return [REDACTED for _item in value]
    return REDACTED


def clean_provider_value(
    value: Any,
    secrets: tuple[str, ...],
    *,
    field_name: str | None = None,
) -> JsonValue:
    """Recursively retain JSON shape while redacting secrets and identities."""
    if field_name is not None and _sensitive_key(field_name):
        return _clean_sensitive_value(value, secrets)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redact_string(value, secrets)
    if isinstance(value, list | tuple):
        return [clean_provider_value(item, secrets) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): clean_provider_value(
                child,
                secrets,
                field_name=str(key),
            )
            for key, child in value.items()
        }
    return redact_string(str(value), secrets)


async def request_usage_json(
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, JsonValue]:
    """Request bounded JSON and return its cleaned raw dictionary."""
    async with client.stream(
        method,
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        **kwargs,
    ) as response:
        content = await _read_bounded_content(response)
        try:
            parsed = json.loads(content)
        except ValueError:
            encoding = response.encoding or "utf-8"
            try:
                body = content.decode(encoding, errors="replace")
            except LookupError:
                body = content.decode("utf-8", errors="replace")
            parsed = {"body": body}
        cleaned = clean_provider_value(parsed, secret_values(secrets))
        raw = cast(
            dict[str, JsonValue],
            cleaned if isinstance(cleaned, dict) else {"value": cleaned},
        )
        if not response.is_success:
            raise UsageResponseError(response.status_code, raw)
    return raw


async def _read_bounded_content(response: httpx.Response) -> bytes:
    """Read at most one mebibyte without buffering an oversized response."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise ValueError("usage response exceeded the 1 MiB limit")
        chunks.append(chunk)
    return b"".join(chunks)

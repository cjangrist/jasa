"""Shared usage snapshot collection, cache reuse, and MCP refresh hooks."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, cast

import httpx
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from jasa.logging import get_logger
from jasa.search.providers import PROVIDER_CLASSES
from jasa.usage.base import (
    clean_provider_value,
    JsonValue,
    redact_string,
    secret_values,
    UsageProbe,
    UsageResponseError,
)
from jasa.usage.providers import PROVIDER_USAGE_PROBES
from omnifetch.cache import CacheBackend
from omnifetch.fetch.providers.registry import import_all_providers
from omnifetch.fetch.shared.config import ProviderSecrets

_LOGGER = get_logger("usage")
_CACHE_KEY = "jasa:usage:v1"
_SCHEMA_VERSION = 1
_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "catalog_fingerprint",
        "refreshed_at",
        "expires_at",
        "ttl_seconds",
        "search",
        "fetch",
    }
)


def _provider_requirements() -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    """Return complete search and fetch provider credential requirements."""
    search: dict[str, tuple[str, ...]] = {
        provider.name: (provider.secret_env,) for provider in PROVIDER_CLASSES
    }
    fetch = cast(
        dict[str, tuple[str, ...]],
        {
            name: provider.required_secrets
            for name, provider in import_all_providers().items()
        },
    )
    return search, fetch


def _catalog_fingerprint(
    search: Mapping[str, tuple[str, ...]],
    fetch: Mapping[str, tuple[str, ...]],
    secrets: ProviderSecrets,
) -> str:
    """Hash provider order, probes, and configured state for safe reuse."""
    identity = {
        "fetch": [
            (name, requirements, bool(secrets.require_all(*requirements)))
            for name, requirements in fetch.items()
        ],
        "probes": {
            name: {
                "configured": bool(
                    secrets.require_all(*probe.required_secrets)
                ),
                "fetch": probe.fetch.__name__,
                "required_secrets": probe.required_secrets,
            }
            for name, probe in PROVIDER_USAGE_PROBES.items()
        },
        "search": [
            (name, requirements, bool(secrets.require_all(*requirements)))
            for name, requirements in search.items()
        ],
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _configured(
    requirements: tuple[str, ...],
    secrets: ProviderSecrets,
) -> bool:
    return bool(secrets.require_all(*requirements))


def _provider_record(
    name: str,
    configured: bool,
    collected: Mapping[str, dict[str, JsonValue]],
    secrets: ProviderSecrets,
) -> dict[str, JsonValue]:
    """Build one stable provider record around an optional raw response."""
    probe = PROVIDER_USAGE_PROBES.get(name)
    if probe is None:
        return {
            "configured": configured,
            "status": "not_implemented",
            "supported": False,
        }
    if not configured:
        return {
            "configured": False,
            "status": "unconfigured",
            "supported": True,
        }
    missing = [
        env_name
        for env_name in probe.required_secrets
        if not secrets.get(env_name)
    ]
    if missing:
        missing_json: list[JsonValue] = list(missing)
        return {
            "configured": True,
            "missing_usage_credentials": missing_json,
            "status": "usage_credentials_missing",
            "supported": True,
        }
    result = collected[name]
    return {
        "configured": True,
        "status": result["status"],
        "supported": True,
        **({"raw": result["raw"]} if "raw" in result else {}),
        **({"error": result["error"]} if "error" in result else {}),
    }


async def _collect_one(
    name: str,
    probe: UsageProbe,
    client: httpx.AsyncClient,
    secrets: ProviderSecrets,
) -> tuple[str, dict[str, JsonValue]]:
    """Collect one provider without allowing its failure to cancel siblings."""
    try:
        raw = await probe.fetch(client, secrets)
    except UsageResponseError as error:
        return name, {
            "status": "error",
            "raw": error.raw,
            "error": {
                "type": "http_error",
                "status_code": error.status_code,
            },
        }
    except Exception as error:
        message = redact_string(str(error), secret_values(secrets))
        return name, {
            "status": "error",
            "error": {
                "type": type(error).__name__,
                "message": message,
            },
        }
    return name, {"status": "ok", "raw": raw}


def _records_match_catalog(
    value: object,
    requirements: Mapping[str, tuple[str, ...]],
) -> bool:
    """Return whether records are dictionaries in exact catalog order."""
    return (
        isinstance(value, dict)
        and list(value) == list(requirements)
        and all(isinstance(record, dict) for record in value.values())
    )


@dataclass(slots=True)
class UsageRuntime:
    """Shared usage collector, cache reader, and background refresh owner."""

    client: httpx.AsyncClient
    cache: CacheBackend
    secrets: ProviderSecrets
    ttl_seconds: int
    clock: Callable[[], float] = time.time
    search_requirements: dict[str, tuple[str, ...]] = field(init=False)
    fetch_requirements: dict[str, tuple[str, ...]] = field(init=False)
    catalog_fingerprint: str = field(init=False)
    _local_snapshot: dict[str, JsonValue] | None = field(
        default=None, init=False
    )
    _refresh_task: asyncio.Task[dict[str, JsonValue]] | None = field(
        default=None, init=False
    )
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Snapshot the complete provider registries in canonical order."""
        search, fetch = _provider_requirements()
        self.search_requirements = search
        self.fetch_requirements = fetch
        self.catalog_fingerprint = _catalog_fingerprint(
            search, fetch, self.secrets
        )

    def _fresh_local(self) -> dict[str, JsonValue] | None:
        snapshot = self._local_snapshot
        if snapshot is None:
            return None
        expires_at = snapshot.get("expires_at")
        if (
            not isinstance(expires_at, int | float)
            or isinstance(expires_at, bool)
            or expires_at <= self.clock()
        ):
            self._local_snapshot = None
            return None
        return copy.deepcopy(snapshot)

    def trigger_refresh(self) -> None:
        """Start a non-blocking refresh check unless recent data is local."""
        if self._closed or self._fresh_local() is not None:
            return
        self._ensure_refresh_task()

    def _ensure_refresh_task(self) -> asyncio.Task[dict[str, JsonValue]]:
        task = self._refresh_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(self._refresh_if_missing())
        self._refresh_task = task
        task.add_done_callback(self._refresh_finished)
        return task

    def _refresh_finished(
        self, task: asyncio.Task[dict[str, JsonValue]]
    ) -> None:
        if self._refresh_task is task:
            self._refresh_task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _LOGGER.warning("Usage refresh failed (%s)", type(error).__name__)

    async def get_snapshot(self) -> dict[str, JsonValue]:
        """Return recent usage, awaiting one coalesced refresh on a miss."""
        local = self._fresh_local()
        if local is not None:
            return local
        task = self._ensure_refresh_task()
        return copy.deepcopy(await asyncio.shield(task))

    async def _read_cache(self) -> dict[str, JsonValue] | None:
        try:
            raw = await self.cache.get(_CACHE_KEY)
        except Exception as error:
            _LOGGER.warning(
                "Usage cache read failed (%s)", type(error).__name__
            )
            return None
        try:
            parsed = json.loads(raw) if isinstance(raw, str | bytes) else raw
        except (TypeError, ValueError):
            parsed = None
        if not self._cache_record_is_current(parsed):
            return None
        cleaned = clean_provider_value(parsed, secret_values(self.secrets))
        return cast(dict[str, JsonValue], cleaned)

    def _cache_record_is_current(self, parsed: object) -> bool:
        """Return whether a cached snapshot matches the exact live catalog."""
        if not isinstance(parsed, dict) or set(parsed) != _SNAPSHOT_KEYS:
            return False
        expires_at = parsed.get("expires_at")
        search = parsed.get("search")
        fetch = parsed.get("fetch")
        return (
            type(parsed.get("schema_version")) is int
            and parsed.get("schema_version") == _SCHEMA_VERSION
            and parsed.get("catalog_fingerprint") == self.catalog_fingerprint
            and isinstance(parsed.get("refreshed_at"), str)
            and isinstance(expires_at, int | float)
            and not isinstance(expires_at, bool)
            and expires_at > self.clock()
            and type(parsed.get("ttl_seconds")) is int
            and parsed.get("ttl_seconds") == self.ttl_seconds
            and _records_match_catalog(search, self.search_requirements)
            and _records_match_catalog(fetch, self.fetch_requirements)
        )

    async def _refresh_if_missing(self) -> dict[str, JsonValue]:
        cached = await self._read_cache()
        if cached is not None:
            self._local_snapshot = cached
            return cached
        snapshot = await self._collect_snapshot()
        self._local_snapshot = snapshot
        try:
            stored = await self.cache.set(
                _CACHE_KEY,
                json.dumps(snapshot, separators=(",", ":")),
                self.ttl_seconds,
            )
            if stored is False:
                _LOGGER.warning("Usage cache write was rejected")
        except Exception as error:
            _LOGGER.warning(
                "Usage cache write failed (%s)", type(error).__name__
            )
        return snapshot

    async def _collect_snapshot(self) -> dict[str, JsonValue]:
        active_names = {
            name
            for requirements in (
                self.search_requirements,
                self.fetch_requirements,
            )
            for name, required in requirements.items()
            if _configured(required, self.secrets)
        }
        probes = {
            name: probe
            for name, probe in PROVIDER_USAGE_PROBES.items()
            if name in active_names
            and _configured(probe.required_secrets, self.secrets)
        }
        pairs = await asyncio.gather(
            *(
                _collect_one(name, probe, self.client, self.secrets)
                for name, probe in probes.items()
            )
        )
        collected = dict(pairs)
        now = self.clock()
        search: dict[str, JsonValue] = {
            name: _provider_record(
                name,
                _configured(requirements, self.secrets),
                collected,
                self.secrets,
            )
            for name, requirements in self.search_requirements.items()
        }
        fetch: dict[str, JsonValue] = {
            name: _provider_record(
                name,
                _configured(requirements, self.secrets),
                collected,
                self.secrets,
            )
            for name, requirements in self.fetch_requirements.items()
        }
        return {
            "schema_version": _SCHEMA_VERSION,
            "catalog_fingerprint": self.catalog_fingerprint,
            "refreshed_at": datetime.fromtimestamp(now, tz=UTC).isoformat(),
            "expires_at": now + self.ttl_seconds,
            "ttl_seconds": self.ttl_seconds,
            "search": search,
            "fetch": fetch,
        }

    async def close(self) -> None:
        """Cancel and observe an in-flight refresh before shared shutdown."""
        self._closed = True
        task = self._refresh_task
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class UsageRefreshMiddleware(Middleware):
    """Trigger background usage refreshes for both public MCP tools."""

    def __init__(self, usage: UsageRuntime) -> None:
        """Retain the shared usage runtime used by both public tools."""
        self._usage = usage

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        """Refresh on web_search/web_fetch without delaying tool execution."""
        if getattr(context.message, "name", None) in _TOOL_NAMES:
            self._usage.trigger_refresh()
        return await call_next(context)

"""Usage snapshot coalescing, strict cache validation, and fail-open I/O."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import Any, cast

import httpx
import pytest

import jasa.usage.runtime as runtime_module
from jasa.usage.base import REDACTED, UsageProbe
from jasa.usage.runtime import _CACHE_KEY, _provider_record
from omnifetch.cache import CacheBackend
from omnifetch.fetch.shared.config import ProviderSecrets
from tests.usage_helpers import build_usage_runtime, UsageCache


def test_configured_provider_reports_separate_missing_usage_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unused(
        _client: httpx.AsyncClient, _secrets: ProviderSecrets
    ) -> dict[str, Any]:
        raise AssertionError("probe must not run")

    monkeypatch.setattr(
        runtime_module,
        "PROVIDER_USAGE_PROBES",
        {"tavily": UsageProbe(("USAGE_ONLY_KEY",), unused)},
    )
    record = _provider_record(
        "tavily",
        True,
        {},
        ProviderSecrets({"TAVILY_API_KEY": "configured"}),
    )
    assert record == {
        "configured": True,
        "missing_usage_credentials": ["USAGE_ONLY_KEY"],
        "status": "usage_credentials_missing",
        "supported": True,
    }


async def test_concurrent_snapshot_misses_coalesce_and_return_copies(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fetch(
        _client: httpx.AsyncClient, _secrets: ProviderSecrets
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"remaining": 9}

    monkeypatch.setattr(
        runtime_module,
        "PROVIDER_USAGE_PROBES",
        {"tavily": UsageProbe(("TAVILY_API_KEY",), fetch)},
    )
    usage = build_usage_runtime(http_client, secrets={"TAVILY_API_KEY": "key"})
    first_task = asyncio.create_task(usage.get_snapshot())
    second_task = asyncio.create_task(usage.get_snapshot())
    async with asyncio.timeout(1):
        await started.wait()
    release.set()
    first, second = await asyncio.gather(first_task, second_task)
    cast(dict[str, object], first["search"])["changed"] = True

    assert calls == 1
    assert "changed" not in cast(dict[str, object], second["search"])
    assert "changed" not in cast(
        dict[str, object], (await usage.get_snapshot())["search"]
    )


async def test_shared_cache_hit_avoids_second_provider_call_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    calls = 0

    async def fetch(
        _client: httpx.AsyncClient, _secrets: ProviderSecrets
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"remaining": 9}

    monkeypatch.setattr(
        runtime_module,
        "PROVIDER_USAGE_PROBES",
        {"tavily": UsageProbe(("TAVILY_API_KEY",), fetch)},
    )
    cache = UsageCache()
    first = build_usage_runtime(
        http_client,
        cache=cache,
        secrets={"TAVILY_API_KEY": "secret"},
    )
    first_snapshot = await first.get_snapshot()
    second = build_usage_runtime(
        http_client,
        cache=cache,
        secrets={"TAVILY_API_KEY": "secret"},
    )
    second_snapshot = await second.get_snapshot()

    assert calls == 1
    assert second_snapshot == first_snapshot
    assert list(cast(dict[str, object], second_snapshot["search"])) == list(
        second.search_requirements
    )
    assert cache.set_calls[0][0::2] == (_CACHE_KEY, 600)


async def test_local_and_shared_ttl_expiry_refreshes_provider(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    now = [1_000.0]
    calls = 0

    async def fetch(
        _client: httpx.AsyncClient, _secrets: ProviderSecrets
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"call": calls}

    monkeypatch.setattr(
        runtime_module,
        "PROVIDER_USAGE_PROBES",
        {"tavily": UsageProbe(("TAVILY_API_KEY",), fetch)},
    )
    usage = build_usage_runtime(
        http_client,
        cache=UsageCache(),
        secrets={"TAVILY_API_KEY": "secret"},
        clock=lambda: now[0],
    )
    await usage.get_snapshot()
    now[0] += 601
    refreshed = await usage.get_snapshot()

    assert calls == 2
    record = cast(dict[str, Any], refreshed["search"])["tavily"]
    assert record["raw"] == {"call": 2}


def _invalid_cache_variants(valid: dict[str, Any]) -> list[object]:
    field_changes: dict[str, list[object]] = {
        "schema_version": [2, True],
        "catalog_fingerprint": ["wrong"],
        "refreshed_at": [1_000],
        "expires_at": [True],
        "ttl_seconds": [300, True],
        "search": [[]],
        "fetch": [[]],
    }
    changed = [
        {**copy.deepcopy(valid), field: value}
        for field, values in field_changes.items()
        for value in values
    ]
    wrong_search_order = copy.deepcopy(valid)
    wrong_search_order["search"] = dict(
        reversed(list(wrong_search_order["search"].items()))
    )
    wrong_fetch_order = copy.deepcopy(valid)
    wrong_fetch_order["fetch"] = dict(
        reversed(list(wrong_fetch_order["fetch"].items()))
    )
    wrong_provider_record = copy.deepcopy(valid)
    wrong_provider_record["search"]["tavily"] = "invalid"
    missing_field = copy.deepcopy(valid)
    missing_field.pop("refreshed_at")
    return [
        "not json",
        b"not json",
        42,
        missing_field,
        {**valid, "unexpected": True},
        wrong_search_order,
        wrong_fetch_order,
        wrong_provider_record,
        *changed,
    ]


async def test_corrupt_or_incompatible_cache_records_are_misses(
    http_client: httpx.AsyncClient,
) -> None:
    seed = build_usage_runtime(http_client)
    valid = cast(dict[str, Any], await seed._collect_snapshot())

    for invalid in _invalid_cache_variants(valid):
        cache = UsageCache(value=invalid)
        await build_usage_runtime(http_client, cache=cache).get_snapshot()
        assert len(cache.set_calls) == 1


async def test_cache_hit_redacts_configured_secrets_again(
    http_client: httpx.AsyncClient,
) -> None:
    secret = "tvly-cache-secret"
    seed = build_usage_runtime(http_client)
    snapshot = cast(dict[str, Any], await seed._collect_snapshot())
    consumer = build_usage_runtime(
        http_client,
        secrets={"TAVILY_API_KEY": secret},
    )
    snapshot["catalog_fingerprint"] = consumer.catalog_fingerprint
    snapshot["search"]["tavily"] = {
        "configured": True,
        "supported": True,
        "status": "error",
        "raw": {"detail": f"leaked {secret}", "user_id": "123"},
    }
    cache = UsageCache(value=json.dumps(snapshot).encode())
    consumer.cache = cast(CacheBackend, cache)
    result = await consumer.get_snapshot()
    raw = cast(dict[str, Any], result["search"])["tavily"]["raw"]
    assert raw == {"detail": f"leaked {REDACTED}", "user_id": REDACTED}
    assert cache.set_calls == []


async def test_shared_cache_rejects_a_different_configured_provider_set(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    calls = 0

    async def fetch(
        _client: httpx.AsyncClient, _secrets: ProviderSecrets
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"remaining": 2}

    monkeypatch.setattr(
        runtime_module,
        "PROVIDER_USAGE_PROBES",
        {"tavily": UsageProbe(("TAVILY_API_KEY",), fetch)},
    )
    cache = UsageCache()
    await build_usage_runtime(http_client, cache=cache).get_snapshot()
    configured = build_usage_runtime(
        http_client,
        cache=cache,
        secrets={"TAVILY_API_KEY": "secret"},
    )
    snapshot = await configured.get_snapshot()

    assert calls == 1
    record = cast(dict[str, Any], snapshot["search"])["tavily"]
    assert record["raw"] == {"remaining": 2}
    assert len(cache.set_calls) == 2


@pytest.mark.parametrize("failure", ["read", "write", "rejected"])
async def test_cache_failures_fail_open_and_are_observable(
    failure: str,
    caplog: pytest.LogCaptureFixture,
    http_client: httpx.AsyncClient,
) -> None:
    cache = UsageCache(
        get_error=OSError("read failed") if failure == "read" else None,
        set_error=OSError("write failed") if failure == "write" else None,
        set_result=failure != "rejected",
    )
    with caplog.at_level(logging.WARNING, logger="jasa.usage"):
        snapshot = await build_usage_runtime(
            http_client, cache=cache
        ).get_snapshot()

    assert snapshot["schema_version"] == 1
    expected = {
        "read": "Usage cache read failed (OSError)",
        "write": "Usage cache write failed (OSError)",
        "rejected": "Usage cache write was rejected",
    }
    assert expected[failure] in caplog.messages


async def test_refresh_deadline_releases_a_blocked_cache_read(
    monkeypatch: pytest.MonkeyPatch,
    http_client: httpx.AsyncClient,
) -> None:
    async def blocked_get(_cache: UsageCache, _key: str) -> object | None:
        await asyncio.Event().wait()
        return None

    monkeypatch.setattr(runtime_module, "_REFRESH_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(UsageCache, "get", blocked_get)
    usage = build_usage_runtime(http_client, cache=UsageCache())

    with pytest.raises(TimeoutError):
        await usage.get_snapshot()
    async with asyncio.timeout(1):
        while usage._refresh_task is not None:
            await asyncio.sleep(0)

    assert usage._refresh_task is None


async def test_cache_write_timeout_returns_the_completed_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    http_client: httpx.AsyncClient,
) -> None:
    async def blocked_set(
        _cache: UsageCache,
        _key: str,
        _value: object,
        _ttl_seconds: int,
    ) -> bool:
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(runtime_module, "_CACHE_WRITE_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(UsageCache, "set", blocked_set)
    usage = build_usage_runtime(http_client, cache=UsageCache())

    with caplog.at_level(logging.WARNING, logger="jasa.usage"):
        snapshot = await usage.get_snapshot()

    assert snapshot["schema_version"] == 1
    assert usage._refresh_task is None
    assert "Usage cache write timed out" in caplog.messages

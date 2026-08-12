"""Cache key, write gate, and the memory/disk backends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jasa.cache.base import (
    KEY_PREFIX,
    make_cache_key,
    should_cache,
)
from jasa.cache.disk import DiskCache
from jasa.cache.memory import MemoryCache


def _expected_key(value: str) -> str:
    return KEY_PREFIX + hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_cache_key_is_sha256_of_query_plus_mode_suffixes() -> None:
    assert make_cache_key("query") == _expected_key("query")
    assert make_cache_key("query", skip_quality_filter=True) == _expected_key(
        "query\0sqf=true"
    )
    assert make_cache_key("query", grounding=True) == _expected_key(
        "query\0gnd=true"
    )
    assert make_cache_key(
        "query", skip_quality_filter=True, grounding=True
    ) == _expected_key("query\0sqf=true\0gnd=true")


def test_should_cache_only_for_complete_non_transient_fanout() -> None:
    assert should_cache(
        providers_succeeded=2, providers_failed=0, want_grounding=False
    )
    assert not should_cache(
        providers_succeeded=0, providers_failed=0, want_grounding=False
    )
    assert not should_cache(
        providers_succeeded=2, providers_failed=1, want_grounding=False
    )
    assert should_cache(
        providers_succeeded=2,
        providers_failed=0,
        want_grounding=True,
        transient_failures=0,
    )
    assert not should_cache(
        providers_succeeded=2,
        providers_failed=0,
        want_grounding=True,
        transient_failures=1,
    )


async def test_memory_round_trip_and_expiry() -> None:
    ticks = [100.0]
    cache = MemoryCache(clock=lambda: ticks[0])
    await cache.set("k", "v", ttl_seconds=50)
    assert await cache.get("k") == "v"
    ticks[0] = 200.0
    assert await cache.get("k") is None


async def test_memory_miss_is_none() -> None:
    assert await MemoryCache().get("absent") is None


async def test_memory_close_clears() -> None:
    cache = MemoryCache()
    await cache.set("k", "v", ttl_seconds=10)
    await cache.close()
    assert await cache.get("k") is None


async def test_memory_set_sweeps_expired_entries_before_eviction() -> None:
    ticks = [100.0]
    cache = MemoryCache(clock=lambda: ticks[0], max_entries=2)
    await cache.set("expired", "old", ttl_seconds=1)
    await cache.set("kept", "current", ttl_seconds=100)
    ticks[0] = 102.0
    await cache.set("new", "latest", ttl_seconds=100)
    assert await cache.get("expired") is None
    assert await cache.get("kept") == "current"
    assert await cache.get("new") == "latest"


async def test_memory_evicts_oldest_entry_at_capacity() -> None:
    cache = MemoryCache(max_entries=2)
    await cache.set("oldest", "one", ttl_seconds=100)
    await cache.set("newer", "two", ttl_seconds=100)
    await cache.set("newest", "three", ttl_seconds=100)
    assert await cache.get("oldest") is None
    assert await cache.get("newer") == "two"
    assert await cache.get("newest") == "three"


def test_memory_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        MemoryCache(max_entries=0)


async def test_disk_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    await cache.set("k", "v", ttl_seconds=3600)
    assert await cache.get("k") == "v"


async def test_disk_corrupt_file_is_miss(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    (tmp_path / "k").write_text("not json", encoding="utf-8")
    assert await cache.get("k") is None


async def test_disk_missing_file_is_miss(tmp_path: Path) -> None:
    assert await DiskCache(str(tmp_path)).get("absent") is None


async def test_disk_expiry_is_miss(tmp_path: Path) -> None:
    ticks = [1000.0]
    cache = DiskCache(str(tmp_path), clock=lambda: ticks[0])
    await cache.set("k", "v", ttl_seconds=10)
    ticks[0] = 5000.0
    assert await cache.get("k") is None


async def test_disk_write_failure_is_swallowed(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    cache._dir = Path("/nonexistent-jasa-cache-test/dir")
    await cache.set("k", "v", ttl_seconds=10)


async def test_disk_legacy_shape_is_miss(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    (tmp_path / "k").write_text(json.dumps({"value": "v"}), encoding="utf-8")
    assert await cache.get("k") is None


async def test_disk_non_dict_record_is_miss(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    (tmp_path / "k").write_text("[1, 2, 3]", encoding="utf-8")
    assert await cache.get("k") is None


async def test_disk_close_is_a_noop(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    await cache.set("k", "v", ttl_seconds=10)
    await cache.close()
    assert await cache.get("k") == "v"

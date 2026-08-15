"""Cache key, write gate, and the memory/disk backends."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

import pytest

from jasa.cache.base import (
    KEY_PREFIX,
    make_cache_key,
    SearchCacheIdentity,
    should_cache,
)
from jasa.cache.disk import _OPERATION_QUEUE_LIMIT, DiskCache
from jasa.cache.memory import MemoryCache


def _identity(
    *,
    query: str = "query",
    raw: bool = False,
    grounding: bool = False,
    providers: tuple[str, ...] = ("alpha", "beta"),
    grounding_fingerprint: str | None = None,
) -> SearchCacheIdentity:
    return SearchCacheIdentity(
        query=query,
        skip_quality_filter=raw,
        grounding=grounding,
        providers=providers,
        grounding_fingerprint=grounding_fingerprint,
    )


def _expected_key(identity: SearchCacheIdentity) -> str:
    canonical = json.dumps(
        asdict(identity),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return KEY_PREFIX + hashlib.sha256(canonical.encode()).hexdigest()


async def _await_thread_event(event: threading.Event) -> None:
    """Wait for a worker handshake with a bounded failure mode."""
    assert await asyncio.to_thread(event.wait, 5), "worker event was not set"


def test_cache_key_hashes_canonical_v2_identity_without_query_text() -> None:
    identity = _identity(query="private exact query ✓")

    key = make_cache_key(identity)

    assert key == _expected_key(identity)
    assert key.startswith("jasa:search:v2:")
    assert "private" not in key


def test_cache_key_separates_provider_order_modes_and_grounding() -> None:
    baseline = make_cache_key(_identity())
    variants = {
        make_cache_key(_identity(providers=("alpha",))),
        make_cache_key(_identity(providers=("beta", "alpha"))),
        make_cache_key(_identity(raw=True)),
        make_cache_key(_identity(grounding=True)),
        make_cache_key(
            _identity(grounding=True, grounding_fingerprint="fingerprint-a")
        ),
        make_cache_key(
            _identity(grounding=True, grounding_fingerprint="fingerprint-b")
        ),
    }

    assert baseline not in variants
    assert len(variants) == 6


def test_cache_key_is_deterministic_for_identical_identity() -> None:
    assert make_cache_key(_identity()) == make_cache_key(
        SearchCacheIdentity(
            query="query",
            skip_quality_filter=False,
            grounding=False,
            providers=("alpha", "beta"),
            grounding_fingerprint=None,
        )
    )


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


async def test_disk_read_can_be_cancelled_while_file_io_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = DiskCache(str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_read_text = Path.read_text

    def blocking_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        started.set()
        release.wait(timeout=5)
        try:
            return original_read_text(path, encoding=encoding, errors=errors)
        finally:
            finished.set()

    monkeypatch.setattr(Path, "read_text", blocking_read_text)
    task = asyncio.create_task(cache.get("missing"))
    await _await_thread_event(started)

    try:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await task
    finally:
        release.set()

    assert await asyncio.to_thread(finished.wait, 1)


async def test_disk_write_can_be_cancelled_while_file_io_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = DiskCache(str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_fsync = os.fsync
    original_replace = os.replace

    def blocking_fsync(file_descriptor: int) -> None:
        started.set()
        release.wait(timeout=5)
        original_fsync(file_descriptor)

    def tracking_replace(
        source: str | os.PathLike[str], destination: Path
    ) -> None:
        original_replace(source, destination)
        finished.set()

    monkeypatch.setattr(os, "fsync", blocking_fsync)
    monkeypatch.setattr(os, "replace", tracking_replace)
    task = asyncio.create_task(cache.set("k", "v", ttl_seconds=3600))
    await _await_thread_event(started)

    try:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await task
    finally:
        release.set()

    assert await asyncio.to_thread(finished.wait, 1)
    assert await cache.get("k") == "v"


async def test_cancelled_expired_read_cannot_unlink_newer_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = [0.0]
    reader_cache = DiskCache(str(tmp_path), clock=lambda: ticks[0])
    writer_cache = DiskCache(str(tmp_path), clock=lambda: ticks[0])
    await writer_cache.set("k", "old", ttl_seconds=1)
    ticks[0] = 2.0
    read_started = threading.Event()
    release_read = threading.Event()
    write_finished = threading.Event()
    pause_next_read = [True]
    original_read_text = Path.read_text
    original_replace = os.replace

    def paused_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        value = original_read_text(path, encoding=encoding, errors=errors)
        if pause_next_read[0]:
            pause_next_read[0] = False
            read_started.set()
            release_read.wait(timeout=5)
        return value

    def tracked_replace(
        source: str | os.PathLike[str], destination: Path
    ) -> None:
        original_replace(source, destination)
        write_finished.set()

    monkeypatch.setattr(Path, "read_text", paused_read)
    monkeypatch.setattr(os, "replace", tracked_replace)
    read_task = asyncio.create_task(reader_cache.get("k"))
    await _await_thread_event(read_started)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await read_task
    newer_write = asyncio.create_task(
        writer_cache.set("k", "new", ttl_seconds=100)
    )
    try:
        assert not await asyncio.to_thread(write_finished.wait, 0.1)
    finally:
        release_read.set()
    await newer_write
    assert await reader_cache.get("k") == "new"


async def test_cancelled_older_write_cannot_replace_newer_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = DiskCache(str(tmp_path))
    first_replace_started = threading.Event()
    release_first_replace = threading.Event()
    second_replace_finished = threading.Event()
    replace_calls = [0]
    original_replace = os.replace

    def ordered_replace(
        source: str | os.PathLike[str], destination: Path
    ) -> None:
        replace_calls[0] += 1
        call = replace_calls[0]
        if call == 1:
            first_replace_started.set()
            release_first_replace.wait(timeout=5)
        original_replace(source, destination)
        if call == 2:
            second_replace_finished.set()

    monkeypatch.setattr(os, "replace", ordered_replace)
    older_write = asyncio.create_task(cache.set("k", "old", ttl_seconds=100))
    await _await_thread_event(first_replace_started)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await older_write
    newer_write = asyncio.create_task(cache.set("k", "new", ttl_seconds=100))
    try:
        assert not await asyncio.to_thread(second_replace_finished.wait, 0.1)
    finally:
        release_first_replace.set()
    await newer_write
    assert second_replace_finished.is_set()
    assert await cache.get("k") == "new"


async def test_disk_bounds_cancelled_shielded_operations_per_stripe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = DiskCache(str(tmp_path))
    first_write_started = threading.Event()
    release_first_write = threading.Event()
    fsync_calls = [0]
    replace_calls = [0]
    original_fsync = os.fsync
    original_replace = os.replace

    def blocking_first_fsync(file_descriptor: int) -> None:
        fsync_calls[0] += 1
        if fsync_calls[0] == 1:
            first_write_started.set()
            release_first_write.wait(timeout=5)
        original_fsync(file_descriptor)

    def tracking_replace(source: str, destination: Path) -> None:
        replace_calls[0] += 1
        original_replace(source, destination)

    monkeypatch.setattr(os, "fsync", blocking_first_fsync)
    monkeypatch.setattr(os, "replace", tracking_replace)
    first_caller = asyncio.create_task(cache.set("k", "0", 3600))
    await _await_thread_event(first_write_started)
    extra_callers = [
        asyncio.create_task(cache.set("k", str(index), 3600))
        for index in range(_OPERATION_QUEUE_LIMIT + 4)
    ]
    while len(cache._background_operations) < _OPERATION_QUEUE_LIMIT:
        await asyncio.sleep(0)
    assert len(cache._background_operations) == _OPERATION_QUEUE_LIMIT
    assert await cache.get("k") is None
    admitted_callers = [
        first_caller,
        *extra_callers[: _OPERATION_QUEUE_LIMIT - 1],
    ]
    overflow_callers = extra_callers[_OPERATION_QUEUE_LIMIT - 1 :]
    try:
        assert await asyncio.gather(*overflow_callers) == [None] * 5
        for caller in admitted_callers:
            caller.cancel()
        results = await asyncio.gather(
            *admitted_callers, return_exceptions=True
        )
        assert all(
            isinstance(result, asyncio.CancelledError) for result in results
        )
        assert len(cache._background_operations) == _OPERATION_QUEUE_LIMIT
        close_task = asyncio.create_task(cache.close())
        await asyncio.sleep(0)
        assert not close_task.done()
    finally:
        release_first_write.set()
    await close_task
    assert cache._background_operations == set()
    assert replace_calls[0] == _OPERATION_QUEUE_LIMIT


async def test_disk_close_waits_for_shielded_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = DiskCache(str(tmp_path))
    write_started = threading.Event()
    release_write = threading.Event()
    original_fsync = os.fsync

    def blocking_fsync(file_descriptor: int) -> None:
        write_started.set()
        release_write.wait(timeout=5)
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", blocking_fsync)
    caller = asyncio.create_task(cache.set("k", "v", 3600))
    await _await_thread_event(write_started)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    close_task = asyncio.create_task(cache.close())
    await asyncio.sleep(0)
    try:
        assert not close_task.done()
    finally:
        release_write.set()
    await close_task
    assert cache._background_operations == set()
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


async def test_disk_failed_atomic_replace_preserves_existing_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = DiskCache(str(tmp_path))
    await cache.set("k", "old", ttl_seconds=3600)

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    await cache.set("k", "new", ttl_seconds=3600)

    assert await cache.get("k") == "old"
    assert list(tmp_path.glob(".k.*.tmp")) == []


async def test_disk_failed_write_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = DiskCache(str(tmp_path))

    def fail_fsync(file_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    await cache.set("k", "new", ttl_seconds=3600)

    assert await cache.get("k") is None
    assert list(tmp_path.glob(".k.*.tmp")) == []


async def test_disk_legacy_shape_is_miss(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    (tmp_path / "k").write_text(json.dumps({"value": "v"}), encoding="utf-8")
    assert await cache.get("k") is None


async def test_disk_non_dict_record_is_miss(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    (tmp_path / "k").write_text("[1, 2, 3]", encoding="utf-8")
    assert await cache.get("k") is None


async def test_disk_close_preserves_contents(tmp_path: Path) -> None:
    cache = DiskCache(str(tmp_path))
    await cache.set("k", "v", ttl_seconds=10)
    await cache.close()
    assert await cache.get("k") == "v"

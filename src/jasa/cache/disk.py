"""Disk cache backend; survives restart, single-process safe.

Each key is one JSON file (``{value, expires_at}``) named by the cache key.
Reads degrade to a miss on any error (corrupt/legacy/missing); writes never
raise (a write failure is logged and swallowed). Wall-clock expiry makes the
configured TTL elapse in real time across restarts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from weakref import WeakValueDictionary

from jasa.logging import get_logger

_LOGGER = get_logger("cache.disk")
_OPERATION_LOCK_STRIPES = 64
_OPERATION_QUEUE_LIMIT = 16


@dataclass(slots=True, weakref_slot=True)
class _DirectoryCoordinator:
    """Bounded fair operation controls shared by one directory and loop."""

    locks: tuple[asyncio.Lock, ...] = field(
        default_factory=lambda: tuple(
            asyncio.Lock() for _ in range(_OPERATION_LOCK_STRIPES)
        )
    )
    admissions: tuple[asyncio.BoundedSemaphore, ...] = field(
        default_factory=lambda: tuple(
            asyncio.BoundedSemaphore(_OPERATION_QUEUE_LIMIT)
            for _ in range(_OPERATION_LOCK_STRIPES)
        )
    )


_COORDINATORS: WeakValueDictionary[
    tuple[str, asyncio.AbstractEventLoop], _DirectoryCoordinator
] = WeakValueDictionary()
_COORDINATORS_LOCK = threading.Lock()


def _shared_coordinator(directory: str) -> _DirectoryCoordinator:
    """Return one process-local coordinator for this directory and loop."""
    key = (directory, asyncio.get_running_loop())
    with _COORDINATORS_LOCK:
        coordinator = _COORDINATORS.get(key)
        if coordinator is None:
            coordinator = _DirectoryCoordinator()
            _COORDINATORS[key] = coordinator
        return coordinator


class DiskCache:
    """A file-backed string store with wall-clock TTL."""

    def __init__(
        self, path: str, *, clock: Callable[[], float] = time.time
    ) -> None:
        """Create the cache directory and bind an injectable clock."""
        self._dir = Path(path)
        self._clock = clock
        self._dir.mkdir(parents=True, exist_ok=True)
        self._directory_scope = str(self._dir.resolve())
        self._coordinators: dict[
            asyncio.AbstractEventLoop, _DirectoryCoordinator
        ] = {}
        self._background_operations: set[asyncio.Task[object]] = set()

    def _file(self, key: str) -> Path:
        return self._dir / key

    def _operation_controls(
        self, key: str
    ) -> tuple[asyncio.Lock, asyncio.BoundedSemaphore]:
        """Return shared lock and bounded admission for this key stripe."""
        loop = asyncio.get_running_loop()
        coordinator = self._coordinators.get(loop)
        if coordinator is None:
            coordinator = _shared_coordinator(self._directory_scope)
            self._coordinators[loop] = coordinator
        stripe = hash(key) % _OPERATION_LOCK_STRIPES
        return coordinator.locks[stripe], coordinator.admissions[stripe]

    def _track_operation(self, operation: asyncio.Task[object]) -> None:
        """Keep a shielded operation alive and consume its final exception."""
        self._background_operations.add(operation)
        operation.add_done_callback(self._finish_operation)

    def _finish_operation(self, operation: asyncio.Task[object]) -> None:
        """Release the strong task reference after its worker has finished."""
        self._background_operations.discard(operation)
        with contextlib.suppress(asyncio.CancelledError):
            operation.exception()

    def _get_sync(self, key: str) -> str | None:
        """Read and validate one entry on a worker thread."""
        try:
            record = json.loads(self._file(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict):
            return None
        expires_at = record.get("expires_at")
        if not isinstance(expires_at, int | float) or "value" not in record:
            return None
        if expires_at <= self._clock():
            with contextlib.suppress(OSError):
                self._file(key).unlink()
            return None
        return str(record["value"])

    async def _get_serialized(
        self,
        key: str,
        lock: asyncio.Lock,
        admission: asyncio.BoundedSemaphore,
    ) -> str | None:
        """Hold the key stripe until its worker read actually finishes."""
        try:
            async with lock:
                return await asyncio.to_thread(self._get_sync, key)
        finally:
            admission.release()

    async def get(self, key: str) -> str | None:
        """Return one entry without blocking the event loop on file I/O."""
        lock, admission = self._operation_controls(key)
        await admission.acquire()
        operation = asyncio.create_task(
            self._get_serialized(key, lock, admission)
        )
        self._track_operation(operation)
        return await asyncio.shield(operation)

    def _set_sync(self, key: str, value: str, ttl_seconds: int) -> None:
        """Atomically replace one entry on a worker thread."""
        record = json.dumps(
            {"value": value, "expires_at": int(self._clock()) + ttl_seconds}
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._dir,
                prefix=f".{key}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(record)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._file(key))
        except OSError as error:
            if temporary_path is not None:
                with contextlib.suppress(OSError):
                    temporary_path.unlink()
            _LOGGER.warning(
                "Disk cache write failed for %s: %s", key[:12], error
            )

    async def _set_serialized(
        self,
        key: str,
        value: str,
        ttl_seconds: int,
        lock: asyncio.Lock,
        admission: asyncio.BoundedSemaphore,
    ) -> None:
        """Hold the key stripe until its worker write actually finishes."""
        try:
            async with lock:
                await asyncio.to_thread(self._set_sync, key, value, ttl_seconds)
        finally:
            admission.release()

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store one entry without blocking the event loop on file I/O."""
        lock, admission = self._operation_controls(key)
        await admission.acquire()
        operation = asyncio.create_task(
            self._set_serialized(key, value, ttl_seconds, lock, admission)
        )
        self._track_operation(operation)
        await asyncio.shield(operation)

    async def close(self) -> None:
        """Wait for shielded filesystem operations without clearing data."""
        operations = tuple(self._background_operations)
        if operations:
            await asyncio.shield(
                asyncio.gather(*operations, return_exceptions=True)
            )

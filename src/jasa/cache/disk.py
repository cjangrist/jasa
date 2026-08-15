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
import time
from collections.abc import Callable
from pathlib import Path

from jasa.logging import get_logger

_LOGGER = get_logger("cache.disk")
_OPERATION_LOCK_STRIPES = 64


class DiskCache:
    """A file-backed string store with wall-clock TTL."""

    def __init__(
        self, path: str, *, clock: Callable[[], float] = time.time
    ) -> None:
        """Create the cache directory and bind an injectable clock."""
        self._dir = Path(path)
        self._clock = clock
        self._operation_locks = tuple(
            asyncio.Lock() for _ in range(_OPERATION_LOCK_STRIPES)
        )
        self._background_operations: set[asyncio.Task[object]] = set()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _file(self, key: str) -> Path:
        return self._dir / key

    def _operation_lock(self, key: str) -> asyncio.Lock:
        """Return the bounded fair lock that serializes this cache key."""
        return self._operation_locks[hash(key) % _OPERATION_LOCK_STRIPES]

    def _track_operation(self, operation: asyncio.Task[object]) -> None:
        """Keep a shielded operation alive and consume its final exception."""
        self._background_operations.add(operation)
        operation.add_done_callback(self._finish_operation)

    def _finish_operation(self, operation: asyncio.Task[object]) -> None:
        """Release the strong task reference after its worker has finished."""
        self._background_operations.discard(operation)
        if not operation.cancelled():
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

    async def _get_serialized(self, key: str) -> str | None:
        """Hold the key stripe until its worker read actually finishes."""
        async with self._operation_lock(key):
            return await asyncio.to_thread(self._get_sync, key)

    async def get(self, key: str) -> str | None:
        """Return one entry without blocking the event loop on file I/O."""
        operation = asyncio.create_task(self._get_serialized(key))
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
        self, key: str, value: str, ttl_seconds: int
    ) -> None:
        """Hold the key stripe until its worker write actually finishes."""
        async with self._operation_lock(key):
            await asyncio.to_thread(self._set_sync, key, value, ttl_seconds)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store one entry without blocking the event loop on file I/O."""
        operation = asyncio.create_task(
            self._set_serialized(key, value, ttl_seconds)
        )
        self._track_operation(operation)
        await asyncio.shield(operation)

    async def close(self) -> None:
        """No resources to release."""
        return None

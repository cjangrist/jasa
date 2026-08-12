"""Disk cache backend; survives restart, single-process safe.

Each key is one JSON file (``{value, expires_at}``) named by the cache key.
Reads degrade to a miss on any error (corrupt/legacy/missing); writes never
raise (a write failure is logged and swallowed). Wall-clock expiry so a 36-hour
TTL expires in real time across restarts.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from jasa.logging import get_logger

_LOGGER = get_logger("cache.disk")


class DiskCache:
    """A file-backed string store with wall-clock TTL."""

    def __init__(
        self, path: str, *, clock: Callable[[], float] = time.time
    ) -> None:
        """Create the cache directory and bind an injectable clock."""
        self._dir = Path(path)
        self._clock = clock
        self._dir.mkdir(parents=True, exist_ok=True)

    def _file(self, key: str) -> Path:
        return self._dir / key

    async def get(self, key: str) -> str | None:
        """Return the value if the file is valid and unexpired, else None."""
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

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Atomically replace the value; swallow and log write failures."""
        record = json.dumps(
            {"value": value, "expires_at": int(self._clock()) + ttl_seconds}
        )
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._dir,
                prefix=f".{key}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(record)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = temporary_file.name
            os.replace(temporary_path, self._file(key))
        except OSError as error:
            _LOGGER.warning(
                "Disk cache write failed for %s: %s", key[:12], error
            )

    async def close(self) -> None:
        """No resources to release."""
        return None

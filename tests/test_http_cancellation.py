"""Real HTTP pool regressions for overlapping request deadlines.

No provider APIs are used. Pausing HTTP cleanup makes the cancellation race
that strands ACTIVE connections deterministic rather than timing-dependent.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
import pytest
from httpcore._async.http11 import AsyncHTTP11Connection

from jasa.grounding.service import _drain_pending_workers
from jasa.search.fanout import (
    _ABANDONED_PROVIDER_TASKS,
    _cancel_and_drain,
)


async def _serve(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    started: asyncio.Event,
    writers: list[asyncio.StreamWriter],
) -> None:
    writers.append(writer)
    try:
        headers = await reader.readuntil(b"\r\n\r\n")
        if b"/fast " in headers:
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                b"Connection: close\r\n\r\nok"
            )
            await writer.drain()
            return
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nx")
        await writer.drain()
        started.set()
        await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.parametrize("layer", ["search", "grounding"])
async def test_overlapping_cancellation_preserves_shared_http_pool(
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    started = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()
    writers: list[asyncio.StreamWriter] = []
    original_close = AsyncHTTP11Connection._response_closed

    async def paused_close(connection: AsyncHTTP11Connection) -> None:
        closing.set()
        await release.wait()
        await original_close(connection)

    monkeypatch.setattr(AsyncHTTP11Connection, "_response_closed", paused_close)
    monkeypatch.setattr("jasa.search.fanout._CANCELLATION_GRACE_SECONDS", 0)
    listener = await asyncio.start_server(
        lambda reader, writer: _serve(reader, writer, started, writers),
        "127.0.0.1",
        0,
    )
    port = listener.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    try:
        async with (
            listener,
            httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1), timeout=1
            ) as client,
        ):
            pool = cast(Any, client._transport)._pool
            task = asyncio.create_task(client.get(f"{base_url}/slow"))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                await asyncio.wait_for(closing.wait(), timeout=1)
                if layer == "search":
                    await _cancel_and_drain([cast(Any, task)])
                    await _cancel_and_drain([cast(Any, task)])
                else:
                    await _drain_pending_workers(
                        [cast(Any, task)], asyncio.get_running_loop().time()
                    )
                assert task.cancelling() == 1
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not pool.connections
                response = await client.get(f"{base_url}/fast")
                assert response.status_code == 200
                assert response.text == "ok"
                await asyncio.sleep(0)
                assert not _ABANDONED_PROVIDER_TASKS
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
    finally:
        for writer in writers:
            writer.close()
            await writer.wait_closed()

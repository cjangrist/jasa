"""Real HTTP pool regression for grounding's stage-deadline cancellation.

A loopback server accepts each request and never answers, the way a slow
scraping provider holds one open. The grounding stage deadline then fires while
a worker's request is in flight. Nothing in httpx or httpcore is patched: the
test asserts the outcome -- a one-connection pool still serves the next request
-- so any cancellation path that strands the connection fails it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Literal

import httpx
import pytest

from jasa.cache.memory import MemoryCache
from jasa.config import GroundingSettings
from jasa.grounding.flights import GroundingFlightRegistry
from jasa.grounding.service import ground_results, GroundingContext
from jasa.search.ranking import RankedWebResult
from tests.conftest import grounding_engine, resolved_waterfall, tier

_STAGE_SECONDS = 0.25
_PAGE_CONTENT = "Real page content for grounding. " * 10


async def _serve(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    writers: list[asyncio.StreamWriter],
) -> None:
    writers.append(writer)
    try:
        headers = await reader.readuntil(b"\r\n\r\n")
        if b" /fast " in headers:
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                b"Connection: close\r\n\r\nok"
            )
            await writer.drain()
            return
        await reader.read()
    finally:
        writer.close()


@pytest.mark.parametrize("hanging_call", ["fetch", "llm"])
async def test_stage_deadline_leaves_the_shared_pool_usable(
    monkeypatch: pytest.MonkeyPatch,
    hanging_call: Literal["fetch", "llm"],
) -> None:
    writers: list[asyncio.StreamWriter] = []
    listener = await asyncio.start_server(
        lambda reader, writer: _serve(reader, writer, writers),
        "127.0.0.1",
        0,
    )
    base_url = f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
    monkeypatch.setattr(
        "jasa.grounding.service.MIN_WORKER_BUDGET_SECONDS", 0.01
    )
    async with (
        listener,
        httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1), timeout=5
        ) as client,
    ):

        async def fetch(engine: object, url: str) -> SimpleNamespace:
            if hanging_call == "fetch":
                await client.post(f"{base_url}/hang")
            return SimpleNamespace(content=_PAGE_CONTENT, title="t")

        monkeypatch.setattr("jasa.grounding.service.execute_web_fetch", fetch)
        settings = GroundingSettings(top_n=1)
        context = GroundingContext(
            engine=grounding_engine(),
            client=client,
            cache=MemoryCache(),
            cache_write_semaphore=asyncio.Semaphore(1),
            flights=GroundingFlightRegistry(),
            waterfall=resolved_waterfall(
                (tier("hanging", f"{base_url}/v1", "model"),)
            ),
            config=settings,
        )
        deadline_at = asyncio.get_running_loop().time() + _STAGE_SECONDS
        pairs, _stats = await ground_results(
            "q",
            [RankedWebResult("t", f"{base_url}/page", ["agg"], ["p"], 0.1)],
            context,
            deadline_at,
        )
        response = await client.get(
            f"{base_url}/fast", timeout=httpx.Timeout(5, pool=1)
        )

    assert pairs[0][1] == "fallback:pipeline_timeout"
    assert response.text == "ok"

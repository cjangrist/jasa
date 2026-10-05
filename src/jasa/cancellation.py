"""Cancel a task once, and repeat a cancellation that AnyIO absorbed.

Two rules live here so that the second can be deleted on its own.

A task with a cancellation already in progress is never cancelled again.
httpcore closes a cancelled request's connection inside an AnyIO cancel shield,
which stops only AnyIO cancellation: a second raw ``Task.cancel()`` that lands
there aborts the close and leaves an ACTIVE connection in the shared pool for
good.

A cancellation that AnyIO absorbed is repeated once. Until AnyIO ships the fix
for agronholm/anyio#1214 (PR #1330, merged after the 4.15.1 release), a
``Task.cancel()`` that lands in the same event-loop iteration as a cancel
scope's own cancellation is swallowed. ``anyio.connect_tcp`` cancels its group
of connection attempts the moment one succeeds, so a search provider or
grounding request cancelled at that instant carries on as if it never was. A
task that is still running, with no cancellation in progress, shortly after
being cancelled is cancelled once more. Delete ``_cancel_again_if_absorbed``
and its scheduling, together with the canary test in
``tests/test_cancellation.py``, once the locked AnyIO includes that fix; the
canary fails at that point to say so.
"""

from __future__ import annotations

import asyncio
from typing import Any

from jasa.logging import get_logger

_LOGGER = get_logger("cancellation")
ABSORBED_CANCELLATION_RECHECK_SECONDS = 0.1


def cancel_task(task: asyncio.Task[Any]) -> None:
    """Cancel ``task`` unless it is unwinding, then recheck it once later."""
    if task.done():
        return
    if task.cancelling():
        _LOGGER.debug(
            "Not cancelling %s: already unwinding (cancelling=%d)",
            task.get_name(),
            task.cancelling(),
        )
    else:
        _LOGGER.debug("Cancelling %s", task.get_name())
        task.cancel()
    loop = asyncio.get_running_loop()
    loop.call_later(
        ABSORBED_CANCELLATION_RECHECK_SECONDS,
        _cancel_again_if_absorbed,
        task,
        loop.time(),
    )


def _cancel_again_if_absorbed(
    task: asyncio.Task[Any], cancelled_at: float
) -> None:
    """Cancel ``task`` again if it kept running with no cancellation pending."""
    if task.done() or task.cancelling():
        return
    _LOGGER.warning(
        "Task %s still running %.2fs after cancellation with none in "
        "progress; cancelling again (absorbed cancellation, anyio#1214)",
        task.get_name(),
        asyncio.get_running_loop().time() - cancelled_at,
    )
    task.cancel()

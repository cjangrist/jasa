"""Process-local grounding miss flights with shielded deadline-aware waiters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GroundingWait:
    """A follower waiting outside the scarce fetch/LLM worker pool."""

    completion: asyncio.Future[None]


@dataclass(slots=True)
class GroundingFlightRegistry:
    """Composition-owned in-process flights for grounding LLM misses."""

    _flights: dict[str, asyncio.Future[None]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def active_count(self) -> int:
        """Return the number of currently led grounding identities."""
        return len(self._flights)

    def claim(self, key: str) -> tuple[bool, asyncio.Future[None]]:
        """Return whether this caller leads the identity's current flight."""
        existing = self._flights.get(key)
        if existing is not None:
            return False, existing
        completion = asyncio.get_running_loop().create_future()
        self._flights[key] = completion
        return True, completion

    def release(self, key: str, completion: asyncio.Future[None]) -> None:
        """Remove one flight and release every shielded waiter."""
        if self._flights.get(key) is completion:
            del self._flights[key]
        if not completion.done():
            completion.set_result(None)


@dataclass(slots=True)
class GroundingFlightOwnership:
    """Track a leader lease across cancellable coroutine handoffs."""

    registry: GroundingFlightRegistry
    key: str | None = field(default=None, init=False)
    completion: asyncio.Future[None] | None = field(default=None, init=False)

    def hold(self, key: str, completion: asyncio.Future[None]) -> None:
        """Retain a newly claimed flight before the next cancellation point."""
        self.key = key
        self.completion = completion

    def release(self) -> None:
        """Idempotently release the retained flight, if any."""
        if self.key is not None and self.completion is not None:
            self.registry.release(self.key, self.completion)
        self.key = None
        self.completion = None


async def wait_for_grounding_flight(
    completion: asyncio.Future[None],
    deadline_at: float,
) -> bool:
    """Wait without cancelling the leader and report deadline availability."""
    try:
        async with asyncio.timeout_at(deadline_at):
            await asyncio.shield(completion)
    except TimeoutError:
        return False
    return True

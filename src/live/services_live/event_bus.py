"""EventBus seam: fan-out outbound wire messages to connected WebSocket clients.

P1 uses an in-process asyncio implementation (one subscriber queue per socket). P2 swaps in
Redis pub/sub behind the same interface so events reach sockets on other Cloud Run instances.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator

from src.live.schemas_live.outbound import OutboundMessage


class EventBus(ABC):
    @abstractmethod
    async def publish(self, bot_id: str, message: OutboundMessage) -> None: ...

    @abstractmethod
    def subscribe(self, bot_id: str) -> AsyncIterator[OutboundMessage]: ...


class LocalEventBus(EventBus):
    """In-process pub/sub keyed by ``botId``."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[OutboundMessage]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, bot_id: str, message: OutboundMessage) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(bot_id, []))
        for queue in queues:
            queue.put_nowait(message)

    async def subscribe(self, bot_id: str) -> AsyncIterator[OutboundMessage]:
        queue: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        async with self._lock:
            self._subscribers[bot_id].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                subs = self._subscribers.get(bot_id, [])
                if queue in subs:
                    subs.remove(queue)

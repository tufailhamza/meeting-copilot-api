"""EventLog seam: the append-only log of meeting events.

This is one of the stable interfaces of the feature. The in-memory implementation here backs
single-instance operation; a durable Redis Streams implementation will later satisfy the exact
same interface (and the exact same parity test-suite), so swapping storage never changes the
callers.

Semantics (the contract every implementation must honor):
- ``append`` assigns a per-meeting monotonically increasing ``offset`` starting at 0.
- ``read`` returns events with ``offset >= from_offset`` in order.
- Meetings are isolated: events in one ``meeting_id`` never appear in another.
- ``subscribe`` yields existing events from ``from_offset`` and then live events as they are
  appended, until the consumer stops iterating (or the task is cancelled).
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator

from src.live.services_live.events import MeetingEvent, StoredEvent, new_event_id


class EventLog(ABC):
    @abstractmethod
    async def append(self, meeting_id: str, event: MeetingEvent) -> StoredEvent: ...

    @abstractmethod
    async def read(self, meeting_id: str, from_offset: int = 0) -> list[StoredEvent]: ...

    @abstractmethod
    def subscribe(
        self, meeting_id: str, from_offset: int = 0
    ) -> AsyncIterator[StoredEvent]: ...


class InMemoryEventLog(EventLog):
    def __init__(self) -> None:
        self._events: dict[str, list[StoredEvent]] = defaultdict(list)
        self._conditions: dict[str, asyncio.Condition] = defaultdict(asyncio.Condition)

    def _condition(self, meeting_id: str) -> asyncio.Condition:
        return self._conditions[meeting_id]

    async def append(self, meeting_id: str, event: MeetingEvent) -> StoredEvent:
        condition = self._condition(meeting_id)
        async with condition:
            events = self._events[meeting_id]
            stored = StoredEvent(
                offset=len(events),
                event_id=new_event_id(),
                ts=time.time(),
                meeting_id=meeting_id,
                event=event,
            )
            events.append(stored)
            condition.notify_all()
        return stored

    async def read(self, meeting_id: str, from_offset: int = 0) -> list[StoredEvent]:
        condition = self._condition(meeting_id)
        async with condition:
            return list(self._events[meeting_id][from_offset:])

    async def subscribe(
        self, meeting_id: str, from_offset: int = 0
    ) -> AsyncIterator[StoredEvent]:
        condition = self._condition(meeting_id)
        next_offset = from_offset
        while True:
            async with condition:
                events = self._events[meeting_id]
                while len(events) <= next_offset:
                    await condition.wait()
                    events = self._events[meeting_id]
                pending = events[next_offset:]
            for stored in pending:
                yield stored
                next_offset += 1

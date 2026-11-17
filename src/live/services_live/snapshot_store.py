"""SnapshotStore seam: durable memory compaction snapshots per meeting.

Phase 1 uses an in-memory implementation behind this interface. Phase 2 swaps in Redis
without changing callers — the same pattern as ``EventLog``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class MemorySnapshot(BaseModel):
    """Compacted transcript memory for a meeting at a point in time."""

    model_config = ConfigDict(extra="forbid")

    summary_text: str = Field(min_length=1)
    compacted_turn_count: int = Field(ge=0)
    event_offset: int = Field(ge=-1)
    updated_at: float


class SnapshotStore(ABC):
    @abstractmethod
    async def save(self, meeting_id: str, snapshot: MemorySnapshot) -> None: ...

    @abstractmethod
    async def load(self, meeting_id: str) -> MemorySnapshot | None: ...

    @abstractmethod
    async def delete(self, meeting_id: str) -> None: ...


class InMemorySnapshotStore(SnapshotStore):
    def __init__(self) -> None:
        self._snapshots: dict[str, MemorySnapshot] = {}
        self._lock = asyncio.Lock()

    async def save(self, meeting_id: str, snapshot: MemorySnapshot) -> None:
        async with self._lock:
            self._snapshots[meeting_id] = snapshot

    async def load(self, meeting_id: str) -> MemorySnapshot | None:
        async with self._lock:
            return self._snapshots.get(meeting_id)

    async def delete(self, meeting_id: str) -> None:
        async with self._lock:
            self._snapshots.pop(meeting_id, None)

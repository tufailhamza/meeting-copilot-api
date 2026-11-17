"""MemoryCompactor: rolling-summary compaction policy + orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.live.config_live import live_settings
from src.live.services_live.compaction.summarizer import (
    FakeSummarizer,
    SummarizerClient,
    build_summarizer_client,
)
from src.live.services_live.events import TranscriptSummaryUpdated
from src.live.services_live.projection import MeetingSession, TranscriptTurn
from src.live.services_live.snapshot_store import InMemorySnapshotStore, MemorySnapshot, SnapshotStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactionPlan:
    """What to compact, snapshotted on the actor loop before the summarizer runs off it."""

    turns: tuple[TranscriptTurn, ...]
    previous_summary: str | None
    compacted_turn_count: int
    total_turns: int


@dataclass(frozen=True)
class CompactionPolicy:
    memory_window_turns: int
    compaction_buffer_turns: int

    def turns_to_compact(self, session: MeetingSession) -> list[TranscriptTurn]:
        """Return transcript turns that should be folded into the rolling summary."""
        excess = len(session.transcript) - session.compacted_turn_count - self.memory_window_turns
        if excess <= self.compaction_buffer_turns:
            return []
        stop = len(session.transcript) - self.memory_window_turns
        if stop <= session.compacted_turn_count:
            return []
        return list(session.transcript[session.compacted_turn_count : stop])

    def new_compacted_turn_count(self, session: MeetingSession) -> int:
        return len(session.transcript) - self.memory_window_turns


class MemoryCompactor:
    def __init__(
        self,
        *,
        policy: CompactionPolicy,
        summarizer: SummarizerClient,
        snapshot_store: SnapshotStore,
        summary_max_chars: int,
    ) -> None:
        self._policy = policy
        self._summarizer = summarizer
        self._snapshot_store = snapshot_store
        self._summary_max_chars = summary_max_chars

    def plan(self, session: MeetingSession) -> CompactionPlan | None:
        """Decide what to fold, synchronously, before handing it to the off-loop summarizer.

        Slicing the transcript here means the background task never reads live session state
        while the actor is appending to it.
        """
        turns = self._policy.turns_to_compact(session)
        if not turns:
            return None
        return CompactionPlan(
            turns=tuple(turns),
            previous_summary=session.transcript_summary,
            compacted_turn_count=self._policy.new_compacted_turn_count(session),
            total_turns=len(session.transcript),
        )

    async def run_plan(
        self,
        meeting_id: str,
        plan: CompactionPlan,
        event_offset: int,
    ) -> TranscriptSummaryUpdated | None:
        """Summarize off the loop; the caller posts the returned event back to the mailbox."""
        summary = await self._summarizer.summarize(plan.previous_summary, list(plan.turns))
        summary = summary.strip()[: self._summary_max_chars]
        if not summary:
            logger.warning(
                "[live] compaction skipped empty summary meeting_id=%s turns=%s",
                meeting_id,
                len(plan.turns),
            )
            return None

        snapshot = MemorySnapshot(
            summary_text=summary,
            compacted_turn_count=plan.compacted_turn_count,
            event_offset=event_offset,
            updated_at=time.time(),
        )
        await self._snapshot_store.save(meeting_id, snapshot)

        logger.info(
            "[live] memory compacted meeting_id=%s turns_folded=%s compacted_total=%s total_turns=%s",
            meeting_id,
            len(plan.turns),
            plan.compacted_turn_count,
            plan.total_turns,
        )
        return TranscriptSummaryUpdated(
            summary_text=summary,
            compacted_turn_count=plan.compacted_turn_count,
            turns_compacted=len(plan.turns),
        )

    async def maybe_compact(
        self,
        meeting_id: str,
        session: MeetingSession,
        event_offset: int,
    ) -> TranscriptSummaryUpdated | None:
        """Inline convenience path; the actor uses plan()/run_plan() to stay off its loop."""
        turns = self._policy.turns_to_compact(session)
        if not turns:
            return None

        summary = await self._summarizer.summarize(session.transcript_summary, turns)
        summary = summary.strip()[: self._summary_max_chars]
        if not summary:
            logger.warning(
                "[live] compaction skipped empty summary meeting_id=%s turns=%s",
                meeting_id,
                len(turns),
            )
            return None

        compacted_count = self._policy.new_compacted_turn_count(session)
        snapshot = MemorySnapshot(
            summary_text=summary,
            compacted_turn_count=compacted_count,
            event_offset=event_offset,
            updated_at=time.time(),
        )
        await self._snapshot_store.save(meeting_id, snapshot)

        logger.info(
            "[live] memory compacted meeting_id=%s turns_folded=%s compacted_total=%s total_turns=%s",
            meeting_id,
            len(turns),
            compacted_count,
            len(session.transcript),
        )
        return TranscriptSummaryUpdated(
            summary_text=summary,
            compacted_turn_count=compacted_count,
            turns_compacted=len(turns),
        )


def build_default_compactor(
    *,
    memory_window_turns: int | None = None,
    snapshot_store: SnapshotStore | None = None,
    summarizer: SummarizerClient | None | object = ...,
) -> MemoryCompactor | None:
    window = memory_window_turns or live_settings.memory_window_turns
    if summarizer is ...:
        summarizer = build_summarizer_client()
    if summarizer is None:
        return None
    return MemoryCompactor(
        policy=CompactionPolicy(
            memory_window_turns=window,
            compaction_buffer_turns=live_settings.compaction_buffer_turns,
        ),
        summarizer=summarizer,  # type: ignore[arg-type]
        snapshot_store=snapshot_store or InMemorySnapshotStore(),
        summary_max_chars=live_settings.summary_max_chars,
    )


def build_test_compactor(
    *,
    memory_window_turns: int = 40,
    compaction_buffer_turns: int = 10,
    snapshot_store: SnapshotStore | None = None,
) -> MemoryCompactor:
    return MemoryCompactor(
        policy=CompactionPolicy(
            memory_window_turns=memory_window_turns,
            compaction_buffer_turns=compaction_buffer_turns,
        ),
        summarizer=FakeSummarizer(max_chars=live_settings.summary_max_chars),
        snapshot_store=snapshot_store or InMemorySnapshotStore(),
        summary_max_chars=live_settings.summary_max_chars,
    )

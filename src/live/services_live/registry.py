"""MeetingRegistry: per-process directory of live meeting actors (keyed by ``botId``)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from src.live.config_live import live_db_settings, live_settings
from src.live.services_live.actor import MeetingActor
from src.live.services_live.event_bus import EventBus, LocalEventBus
from src.live.services_live.event_log import EventLog, InMemoryEventLog
from src.live.services_live.chat import ChatAgent, build_default_chat_agent
from src.live.services_live.compaction import MemoryCompactor, build_default_compactor
from src.live.services_live.orchestrator import AgentOrchestrator
from src.live.services_live.prep_loader import PostgresPrepLoader, PrepLoader
from src.live.services_live.snapshot_store import InMemorySnapshotStore, SnapshotStore
from src.live.services_live.trigger import TriggerConfig

logger = logging.getLogger(__name__)


def build_default_prep_loader() -> PrepLoader | None:
    if not live_db_settings.configured:
        return None
    return PostgresPrepLoader(
        meetings_table=live_settings.meetings_table,
        call_prep_table=live_settings.call_prep_table,
        db_settings=live_db_settings,
    )


class MeetingRegistry:
    def __init__(
        self,
        event_log: EventLog | None = None,
        event_bus: EventBus | None = None,
        prep_loader: PrepLoader | None | object = ...,
        *,
        trigger_config: TriggerConfig | None = None,
        orchestrator_factory: Callable[[], AgentOrchestrator | None] | None = None,
        chat_agent_factory: Callable[[], ChatAgent | None] | None = None,
        question_answerer_factory: Callable[[], object | None] | None = None,
        snapshot_store: SnapshotStore | None = None,
    ) -> None:
        self._log: EventLog = event_log or InMemoryEventLog()
        self._bus: EventBus = event_bus or LocalEventBus()
        self._snapshot_store: SnapshotStore = snapshot_store or InMemorySnapshotStore()
        if prep_loader is ...:
            prep_loader = build_default_prep_loader()
        self._prep_loader: PrepLoader | None = prep_loader  # type: ignore[assignment]
        self._trigger_config = trigger_config
        self._orchestrator_factory = orchestrator_factory
        self._chat_agent_factory = chat_agent_factory
        self._question_answerer_factory = question_answerer_factory
        self._actors: dict[str, MeetingActor] = {}
        self._lock = asyncio.Lock()

    @property
    def event_log(self) -> EventLog:
        return self._log

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    @property
    def prep_loader(self) -> PrepLoader | None:
        return self._prep_loader

    @property
    def snapshot_store(self) -> SnapshotStore:
        return self._snapshot_store

    def _build_orchestrator(self) -> AgentOrchestrator | None | object:
        if self._orchestrator_factory is not None:
            return self._orchestrator_factory()
        return ...

    def _build_chat_agent(self) -> ChatAgent | None | object:
        if self._chat_agent_factory is not None:
            return self._chat_agent_factory()
        return ...

    def _build_question_answerer(self) -> object | None:
        if self._question_answerer_factory is not None:
            return self._question_answerer_factory()
        return ...

    def _build_memory_compactor(self) -> MemoryCompactor | None | object:
        return build_default_compactor(snapshot_store=self._snapshot_store)

    async def get_or_create(self, bot_id: str) -> MeetingActor:
        async with self._lock:
            actor = self._actors.get(bot_id)
            if actor is None:
                actor = MeetingActor(
                    bot_id,
                    self._log,
                    prep_loader=self._prep_loader,
                    event_bus=self._bus,
                    trigger_config=self._trigger_config,
                    orchestrator=self._build_orchestrator(),
                    chat_agent=self._build_chat_agent(),
                    memory_compactor=self._build_memory_compactor(),
                    question_answerer=self._build_question_answerer(),
                )
                actor.start()
                self._actors[bot_id] = actor
            return actor

    def get(self, bot_id: str) -> MeetingActor | None:
        return self._actors.get(bot_id)

    def active_count(self) -> int:
        return len(self._actors)

    async def remove(self, bot_id: str) -> None:
        async with self._lock:
            actor = self._actors.pop(bot_id, None)
        if actor is not None:
            await actor.stop()
            logger.info("[live] actor removed bot_id=%s", bot_id)


registry = MeetingRegistry()

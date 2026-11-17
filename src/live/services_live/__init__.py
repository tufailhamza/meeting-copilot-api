"""Service layer for the live feature: event log, meeting actor, agent runtime, tools,
and the interface seams (EventLog, EventBus, ToolRegistry, ModelClient, SnapshotStore).
"""

from src.live.services_live.actor import MeetingActor
from src.live.services_live.event_bus import EventBus, LocalEventBus
from src.live.services_live.event_log import EventLog, InMemoryEventLog
from src.live.services_live.events import (
    AgentEvaluationTriggered,
    AgendaRiskEmitted,
    ChatMessageAppended,
    CompanyDetected,
    FactCheckEmitted,
    MeetingEvent,
    MeetingStarted,
    PrepQuestion,
    PrepQuestionsLoaded,
    PrepQuestionAnswered,
    SessionUserRecord,
    StoredEvent,
    SuggestionEmitted,
    TranscriptSegmentAppended,
    TranscriptSummaryUpdated,
    new_event_id,
)
from src.live.services_live.memory import prompt_transcript, recent_transcript
from src.live.services_live.snapshot_store import InMemorySnapshotStore, MemorySnapshot, SnapshotStore
from src.live.services_live.prep_loader import (
    PostgresPrepLoader,
    PrepData,
    PrepLoader,
    StaticPrepLoader,
    parse_prep_payload,
)
from src.live.services_live.projection import (
    ChatEntry,
    EvaluationTriggerRecord,
    FactCheckState,
    MeetingSession,
    QuestionState,
    RiskState,
    SuggestionState,
    TranscriptTurn,
    reduce,
)
from src.live.services_live.registry import MeetingRegistry, registry
from src.live.services_live.orchestrator import (
    AgentOrchestrator,
    OrchestratorResult,
    action_allowed_for_gate,
    build_default_orchestrator,
)
from src.live.services_live.trigger import EvaluationTriggerPolicy, TriggerConfig, TriggerReason
from src.live.services_live.tools import (
    FactCheckTool,
    FlagAgendaRiskTool,
    AnswerPrepQuestionTool,
    SuggestFollowupTool,
    Tool,
    ToolRegistry,
    ToolResult,
    build_default_tool_registry,
    parse_fact_verdict,
)

from src.live.services_live.wire import domain_event_to_wire

from src.live.services_live import agent as live_agent
from src.live.services_live import chat as live_chat
from src.live.services_live import compaction as live_compaction

__all__ = [
    # actor + registry
    "MeetingActor",
    "MeetingRegistry",
    "registry",
    # event log + bus
    "EventBus",
    "EventLog",
    "InMemoryEventLog",
    "InMemorySnapshotStore",
    "LocalEventBus",
    "MemorySnapshot",
    "SnapshotStore",
    # events
    "AgentEvaluationTriggered",
    "AgendaRiskEmitted",
    "ChatMessageAppended",
    "CompanyDetected",
    "FactCheckEmitted",
    "MeetingEvent",
    "MeetingStarted",
    "PrepQuestion",
    "PrepQuestionsLoaded",
    "PrepQuestionAnswered",
    "SessionUserRecord",
    "StoredEvent",
    "SuggestionEmitted",
    "TranscriptSegmentAppended",
    "TranscriptSummaryUpdated",
    "new_event_id",
    # projection
    "ChatEntry",
    "EvaluationTriggerRecord",
    "FactCheckState",
    "MeetingSession",
    "QuestionState",
    "RiskState",
    "SuggestionState",
    "TranscriptTurn",
    "reduce",
    # prep loader
    "PostgresPrepLoader",
    "PrepData",
    "PrepLoader",
    "StaticPrepLoader",
    "parse_prep_payload",
    # orchestrator
    "AgentOrchestrator",
    "OrchestratorResult",
    "action_allowed_for_gate",
    "build_default_orchestrator",
    # memory + trigger
    "EvaluationTriggerPolicy",
    "TriggerConfig",
    "TriggerReason",
    "prompt_transcript",
    "recent_transcript",
    # tools
    "FactCheckTool",
    "FlagAgendaRiskTool",
    "AnswerPrepQuestionTool",
    "SuggestFollowupTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "build_default_tool_registry",
    "parse_fact_verdict",
    "domain_event_to_wire",
    # agent + chat
    "live_agent",
    "live_chat",
    "live_compaction",
]

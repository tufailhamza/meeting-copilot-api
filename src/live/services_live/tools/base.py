"""ToolRegistry seam: execute structured reasoner outcomes as validated domain events.

Tools are pure over ``MeetingSession`` + ``ReasonerOutcome`` — they decide whether an action is
allowed and return events to append. Orchestration (P1.11) dispatches here; the actor appends
whatever events the tool returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.live.services_live.agent.schemas import ReasonerAction, ReasonerOutcome
from src.live.services_live.events import MeetingEvent
from src.live.services_live.projection import MeetingSession


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a single tool invocation."""

    events: tuple[MeetingEvent, ...] = ()
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def applied(self) -> bool:
        return bool(self.events) and not self.skipped


class Tool(ABC):
    """One agent action the orchestrator can invoke."""

    action: ReasonerAction

    @abstractmethod
    def execute(self, session: MeetingSession, outcome: ReasonerOutcome) -> ToolResult: ...


class ToolRegistry:
    """Maps ``ReasonerAction`` values to tool implementations."""

    def __init__(self) -> None:
        self._tools: dict[ReasonerAction, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.action] = tool

    def get(self, action: ReasonerAction) -> Tool | None:
        return self._tools.get(action)

    def execute(self, session: MeetingSession, outcome: ReasonerOutcome) -> ToolResult:
        tool = self._tools.get(outcome.action)
        if tool is None:
            return ToolResult(skipped=True, skip_reason=f"no tool registered for {outcome.action.value}")
        return tool.execute(session, outcome)

    def supported_actions(self) -> frozenset[ReasonerAction]:
        return frozenset(self._tools)


def build_default_tool_registry() -> ToolRegistry:
    """Production registry with core P1 agent tools."""
    from src.live.services_live.tools.fact_check import FactCheckTool
    from src.live.services_live.tools.flag_agenda_risk import FlagAgendaRiskTool
    from src.live.services_live.tools.mark_question import AnswerPrepQuestionTool
    from src.live.services_live.tools.suggest_followup import SuggestFollowupTool

    registry = ToolRegistry()
    registry.register(AnswerPrepQuestionTool())
    registry.register(SuggestFollowupTool())
    registry.register(FactCheckTool())
    registry.register(FlagAgendaRiskTool())
    return registry

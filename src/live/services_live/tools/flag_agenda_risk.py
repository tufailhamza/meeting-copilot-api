"""``flag_agenda_risk`` tool: emit short agenda/scope drift warnings.

Novelty is the model's call. It receives ``prior_risks`` in context and is instructed not to
repeat itself; a second string-similarity check here would only overrule that judgment — rejecting
a genuinely new warning that happens to share words, while waving through a reworded repeat.
"""

from __future__ import annotations

from src.live.config_live import live_settings
from src.live.schemas_live.common import new_id
from src.live.services_live.agent.schemas import ReasonerAction, ReasonerOutcome
from src.live.services_live.events import AgendaRiskEmitted
from src.live.services_live.projection import MeetingSession
from src.live.services_live.tools.base import Tool, ToolResult


class FlagAgendaRiskTool(Tool):
    action = ReasonerAction.FLAG_AGENDA_RISK

    def execute(self, session: MeetingSession, outcome: ReasonerOutcome) -> ToolResult:
        if outcome.action != self.action:
            return ToolResult(skipped=True, skip_reason="wrong action for flag_agenda_risk tool")

        text = (outcome.risk_text or "").strip()
        if not text:
            return ToolResult(skipped=True, skip_reason="missing risk_text")
        # Display constraint, not a quality judgment: the panel shows one short line.
        if len(text) > live_settings.risk_text_max_chars:
            return ToolResult(skipped=True, skip_reason="risk_text too long for the panel")

        return ToolResult(
            events=(
                AgendaRiskEmitted(
                    risk_id=new_id("risk"),
                    text=text,
                ),
            )
        )

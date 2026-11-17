"""``fact_check`` tool: emit ``FactCheckEmitted``, or abstain when the model is not confident.

The claim is grounded by citation — ``evidence_turn_seq`` points at the transcript turn where it
was asserted — so a fabricated claim cannot pass and a correctly-identified one cannot be rejected
for being worded differently from the speaker.

Whether something is an assertion rather than a question, an opinion, or a name-drop is a semantic
judgment made by the gate and the reasoner, which the prompts cover at length. This tool does not
re-litigate it with string inspection.

The one thing it does enforce is a **confidence floor**, because a confidently wrong fact-check
shown to a host mid-call is worse than silence. That is a policy knob, not a text heuristic, and it
lives in ``config_live`` where it is visible and tunable.
"""

from __future__ import annotations

from src.live.config_live import live_settings
from src.live.schemas_live.common import FactCheckVerdict, new_id
from src.live.services_live.agent.schemas import ReasonerAction, ReasonerOutcome
from src.live.services_live.events import FactCheckEmitted
from src.live.services_live.projection import MeetingSession
from src.live.services_live.tools.base import Tool, ToolResult

_VERDICT_ALIASES: dict[str, FactCheckVerdict] = {
    "true": FactCheckVerdict.TRUE,
    "false": FactCheckVerdict.FALSE,
    "unclear": FactCheckVerdict.UNCLEAR,
    "unknown": FactCheckVerdict.UNCLEAR,
}


def parse_fact_verdict(raw: str | None) -> FactCheckVerdict | None:
    if raw is None:
        return None
    return _VERDICT_ALIASES.get(raw.strip().lower())


class FactCheckTool(Tool):
    action = ReasonerAction.FACT_CHECK

    def execute(self, session: MeetingSession, outcome: ReasonerOutcome) -> ToolResult:
        if outcome.action != self.action:
            return ToolResult(skipped=True, skip_reason="wrong action for fact_check tool")

        claim = (outcome.claim or "").strip()
        detail = (outcome.fact_detail or "").strip()
        verdict = parse_fact_verdict(outcome.fact_verdict)
        confidence = outcome.confidence

        if not claim:
            return ToolResult(skipped=True, skip_reason="missing claim")
        if verdict is None:
            return ToolResult(skipped=True, skip_reason="invalid or missing fact_verdict")
        if not detail:
            return ToolResult(skipped=True, skip_reason="missing fact_detail")
        if confidence is None:
            return ToolResult(skipped=True, skip_reason="missing confidence")

        if session.turn_by_seq(outcome.evidence_turn_seq) is None:
            return ToolResult(skipped=True, skip_reason="evidence_turn_seq not in transcript")

        floor = (
            live_settings.fact_check_min_unclear_confidence
            if verdict == FactCheckVerdict.UNCLEAR
            else live_settings.fact_check_min_assert_confidence
        )
        if confidence < floor:
            return ToolResult(
                skipped=True,
                skip_reason=f"abstained: confidence {confidence:.2f} below floor {floor:.2f}",
            )

        return ToolResult(
            events=(
                FactCheckEmitted(
                    fact_check_id=new_id("fc"),
                    claim=claim,
                    verdict=verdict,
                    confidence=confidence,
                    detail=detail,
                ),
            )
        )

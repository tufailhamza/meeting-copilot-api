"""``suggest_followup`` tool: address one suggestion to every coached user.

A suggestion is derived from the shared transcript, so every coached user gets one — but each as
their *own* addressed entity with its own id, so refining it from one user's chat cannot mutate
another user's panel. Any ``linked_question_id`` the reasoner supplies is canonical and is
translated into each user's own question id.

Novelty is the model's call: it already sees ``prior_suggestions`` and the prep questions and is
told not to duplicate them. The string-similarity check that used to sit here shrank the space of
allowed output with every emission, which is why the coach went quiet in the back half of a call.
"""

from __future__ import annotations

from src.live.schemas_live.common import new_id
from src.live.services_live.agent.schemas import ReasonerAction, ReasonerOutcome
from src.live.services_live.events import MeetingEvent, SuggestionEmitted
from src.live.services_live.projection import MeetingSession
from src.live.services_live.tools.base import Tool, ToolResult
from src.live.services_live.tools.mark_question import resolve_canonical_question


class SuggestFollowupTool(Tool):
    action = ReasonerAction.SUGGEST_FOLLOWUP

    def execute(self, session: MeetingSession, outcome: ReasonerOutcome) -> ToolResult:
        if outcome.action != self.action:
            return ToolResult(skipped=True, skip_reason="wrong action for suggest_followup tool")

        text = (outcome.suggestion_text or "").strip()
        rationale = (outcome.suggestion_rationale or outcome.rationale or "").strip()
        linked_question_id = (outcome.linked_question_id or "").strip() or None

        if not text:
            return ToolResult(skipped=True, skip_reason="missing suggestion_text")
        if not rationale:
            return ToolResult(skipped=True, skip_reason="missing suggestion_rationale")

        canonical = None
        if linked_question_id is not None:
            canonical = resolve_canonical_question(session, linked_question_id)
            if canonical is None:
                return ToolResult(skipped=True, skip_reason="unknown linked_question_id")

        recipients = session.user_ids
        if not recipients:
            return ToolResult(skipped=True, skip_reason="no coached users on session")

        events: list[MeetingEvent] = []
        for user_id in recipients:
            events.append(
                SuggestionEmitted(
                    user_id=user_id,
                    suggestion_id=new_id("sug"),
                    text=text,
                    rationale=rationale,
                    # Only link when this specific user actually owns the question.
                    linked_question_id=canonical.owners.get(user_id) if canonical else None,
                )
            )
        return ToolResult(events=tuple(events))

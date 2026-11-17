"""Chat tool: refine an existing coach suggestion and re-emit on the wire.

Scoped to the requesting user: a suggestion can only be refined by the user who owns it, so one
user's rewording never mutates another user's panel even when both hold the same advice.
"""

from __future__ import annotations

from src.live.services_live.events import SuggestionRefined
from src.live.services_live.projection import MeetingSession
from src.live.services_live.tools.base import ToolResult


class RefineSuggestionTool:
    def execute(
        self,
        session: MeetingSession,
        suggestion_id: str,
        new_text: str,
        *,
        user_id: str,
    ) -> ToolResult:
        sid = suggestion_id.strip()
        text = new_text.strip()
        if not user_id or not session.has_user(user_id):
            return ToolResult(skipped=True, skip_reason="unknown user_id")
        if not sid:
            return ToolResult(skipped=True, skip_reason="missing suggestion_id")
        if not text:
            return ToolResult(skipped=True, skip_reason="missing refined text")

        if not any(s.id == sid and s.user_id == user_id for s in session.suggestions):
            return ToolResult(skipped=True, skip_reason="unknown suggestion_id")

        return ToolResult(
            events=(
                SuggestionRefined(
                    user_id=user_id,
                    suggestion_id=sid,
                    text=text,
                ),
            )
        )

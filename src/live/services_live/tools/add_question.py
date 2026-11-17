"""Chat tool: add a prep question during the meeting.

Scoped to the user who asked for it — the question lands on their list only, and its id is the
lowest free ``qN`` *within that user*, so ids stay short and independent across users.

The host explicitly asked for this question, so there is nothing to second-guess: no similarity
check stands between the request and the result.
"""

from __future__ import annotations

from src.live.services_live.events import QuestionAdded
from src.live.services_live.projection import MeetingSession
from src.live.services_live.tools.base import ToolResult


def _next_question_id(session: MeetingSession, user_id: str) -> str:
    existing = session.questions_for(user_id)
    n = 1
    while f"q{n}" in existing:
        n += 1
    return f"q{n}"


class AddQuestionTool:
    def execute(
        self,
        session: MeetingSession,
        question_text: str,
        *,
        user_id: str,
    ) -> ToolResult:
        text = question_text.strip()
        if not user_id or not session.has_user(user_id):
            return ToolResult(skipped=True, skip_reason="unknown user_id")
        if not text:
            return ToolResult(skipped=True, skip_reason="missing question text")

        return ToolResult(
            events=(
                QuestionAdded(
                    user_id=user_id,
                    question_id=_next_question_id(session, user_id),
                    text=text,
                ),
            )
        )

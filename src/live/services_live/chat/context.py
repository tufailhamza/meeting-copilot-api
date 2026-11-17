"""Build chat context from the live meeting session, scoped to one user.

Chat is private. The requesting user's own questions, suggestions, prep context and thread history
go into the prompt; no other user's do. Only genuinely shared material — the transcript, detected
companies, broadcast risks — is common, because that is the meeting itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.live.services_live.projection import MeetingSession
from src.live.services_live.roster import roster_summary


@dataclass
class ChatContext:
    bot_id: str
    user_id: str = ""
    prep_context: dict[str, Any] = field(default_factory=dict)
    roster: dict[str, Any] = field(default_factory=dict)
    questions: list[dict[str, Any]] = field(default_factory=list)
    transcript_summary: str | None = None
    recent_transcript: list[dict[str, str]] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    chat_history: list[dict[str, str]] = field(default_factory=list)
    recent_companies: list[dict[str, Any]] = field(default_factory=list)
    # Populated on the second pass of a company question: what the company service returned for
    # the name the host asked about. ``None`` with a non-empty query means the service had no
    # record — say so, never fall back to guessing.
    company_lookup: dict[str, Any] | None = None

    def to_prompt_block(self) -> str:
        block = {
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "roster": self.roster,
            "prep_summary": {
                k: self.prep_context.get(k)
                for k in ("org_name", "meeting_type", "deal_context", "what_they_care_about")
                if self.prep_context.get(k) is not None
            },
            "prep_questions": self.questions,
            "transcript_summary": self.transcript_summary,
            "recent_transcript": self.recent_transcript,
            "suggestions": self.suggestions,
            "chat_history": self.chat_history[-20:],
            "recent_companies": self.recent_companies[-10:],
            "company_lookup": self.company_lookup,
        }
        return json.dumps(block, indent=2, ensure_ascii=False)


def build_chat_context(
    session: MeetingSession,
    *,
    user_id: str,
    transcript_window: int = 20,
    company_lookup: dict[str, Any] | None = None,
) -> ChatContext:
    summary, turns = session.prompt_transcript(transcript_window)
    companies = [
        {
            "company_name": c.company_name,
            "title": c.title,
            "text": c.text,
            "li_urn": c.li_urn,
            "li_name": c.li_name,
            "li_universalname": c.li_universalname,
            "li_industries": list(c.li_industries),
        }
        for c in session.companies
    ]
    return ChatContext(
        bot_id=session.bot_id,
        user_id=user_id,
        prep_context=dict(session.context_for(user_id)),
        roster=roster_summary(session),
        questions=[
            # ``answer`` is what the meeting said, so the host can ask "what did they say about
            # comp?" and get the recorded line back instead of a blank.
            {
                "id": q.id,
                "text": q.text,
                "status": q.status.value,
                "answer": q.evidence,
            }
            for q in session.ordered_questions_for(user_id)
        ],
        transcript_summary=summary,
        recent_transcript=[{"speaker": t.speaker, "text": t.text} for t in turns],
        suggestions=[
            {"id": s.id, "text": s.text, "rationale": s.rationale}
            for s in session.suggestions_for(user_id)
        ],
        chat_history=[{"role": c.role, "text": c.text} for c in session.chat_for(user_id)],
        recent_companies=companies[-10:],
        company_lookup=company_lookup,
    )

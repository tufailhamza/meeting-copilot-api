"""``answer_prep_question`` tool: record what the meeting answered, and keep it current.

A prep question is not a checkbox that flips once. It is a **question and its current answer**,
and a live meeting revises answers: a figure floated early gets corrected later, often without
anyone re-asking. So this tool emits on the first answer *and* on every genuine change, and the
projection replaces rather than appends — the coach reasons about what is true now.

Three resolutions happen here, all exact lookups rather than text comparisons:

- **Which question(s)** — the agent names *canonical* ids (``c1``), each mapping to every user who
  owns a copy. One stretch of conversation often answers several prep questions, so this is a list.
- **Which answer** — the agent names a transcript ``seq``, and the answer recorded is that turn's
  verbatim text. It must be the turn that *answers*, never the one that asks.
- **New answer or the same one** — the cited ``seq`` is compared to the ``seq`` already recorded.
  Same turn means nothing changed, so nothing is emitted; that is what stops one answer being
  re-sent on every evaluation cycle. It is an id comparison, not a similarity score, so a genuinely
  new answering turn always gets through.

Whether the substance actually changed is a semantic judgment and belongs to the model — someone
restating "around 200" as "two hundred thousand" has not changed their answer. Nothing here
second-guesses that with string inspection.
"""

from __future__ import annotations

from src.live.schemas_live.common import QuestionStatus
from src.live.services_live.agent.schemas import ReasonerAction, ReasonerOutcome
from src.live.services_live.events import PrepQuestionAnswered
from src.live.services_live.projection import CanonicalQuestion, MeetingSession
from src.live.services_live.tools.base import Tool, ToolResult


def resolve_canonical_question(
    session: MeetingSession,
    question_id: str,
) -> CanonicalQuestion | None:
    """Resolve a reasoner-supplied id to a canonical question.

    Accepts a canonical id (``c1``) directly. As a tolerance for model drift it also accepts a
    per-user question id (``q2``) when exactly one canonical question has it as an owner id —
    ambiguous ids resolve to nothing rather than to an arbitrary guess.
    """
    canonicals = session.canonical_questions()
    for canonical in canonicals:
        if canonical.id == question_id:
            return canonical

    matches = [c for c in canonicals if question_id in c.owners.values()]
    if len(matches) == 1:
        return matches[0]
    return None


class AnswerPrepQuestionTool(Tool):
    action = ReasonerAction.ANSWER_PREP_QUESTION

    def execute(self, session: MeetingSession, outcome: ReasonerOutcome) -> ToolResult:
        if outcome.action != self.action:
            return ToolResult(skipped=True, skip_reason="wrong action for answer_prep_question tool")

        if not outcome.question_marks:
            return ToolResult(skipped=True, skip_reason="no question_marks supplied")

        events: list[PrepQuestionAnswered] = []
        skips: list[str] = []
        for mark in outcome.question_marks:
            question_id = (mark.question_id or "").strip()
            canonical = resolve_canonical_question(session, question_id)
            if canonical is None:
                skips.append(f"unknown question_id {question_id!r}")
                continue

            # Existence check, not a similarity check: a fabricated seq simply is not there. The
            # citation is what grounds the answer; the wording below is what the host reads.
            turn = session.turn_by_seq(mark.evidence_turn_seq)
            if turn is None:
                skips.append(f"evidence_turn_seq not in transcript for {question_id!r}")
                continue
            end_seq = mark.evidence_turn_seq_end
            if end_seq is not None and session.turn_by_seq(end_seq) is None:
                skips.append(f"evidence_turn_seq_end not in transcript for {question_id!r}")
                continue
            answer = (mark.answer_text or "").strip()
            if not answer:
                skips.append(f"missing answer_text for {question_id!r}")
                continue

            # Fan out to every owner. Each owner keeps its own revision count, so a user whose
            # prep was reloaded mid-meeting is not forced to inherit someone else's history.
            emitted_any = False
            for user_id, owned_id in canonical.owners.items():
                state = session.questions_for(user_id).get(owned_id)
                if state is None:
                    continue
                if state.evidence_turn_seq == mark.evidence_turn_seq:
                    # Already recorded from this exact turn — re-emitting would duplicate a card.
                    continue
                events.append(
                    PrepQuestionAnswered(
                        user_id=user_id,
                        question_id=owned_id,
                        answer=answer,
                        evidence_turn_seq=mark.evidence_turn_seq,
                        evidence_turn_seq_end=end_seq,
                        revision=state.revision + 1,
                    )
                )
                emitted_any = True
            if not emitted_any:
                skips.append(f"answer already recorded from that turn for {question_id!r}")

        if not events:
            # A single reason keeps the log readable when the model named one bad id.
            return ToolResult(
                skipped=True,
                skip_reason=skips[0] if len(skips) == 1 else "; ".join(skips) or "nothing to record",
            )
        return ToolResult(events=tuple(events))

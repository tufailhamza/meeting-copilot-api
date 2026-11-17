"""Build the evaluation context snapshot passed to gate and reasoner prompts.

**Per-user context isolation.** Each coached user brings their own CRM/prep context, and those can
disagree — two users on the same call may hold different deal stages, sizes, or notes. Anything the
agent produces from the shared reasoning pass is broadcast to every user, so grounding it in one
user's private context would leak that context to the others.

The context is therefore split:

- ``shared_prep``   — only fields that are **identical across every user's** prep. Safe to ground
  broadcast output in, because every user already has it.
- ``per_user_prep`` — each user's remaining fields, labelled by user id. Present so the agent can
  understand who is who; the prompt forbids grounding broadcast output in it.

Questions arrive already deduped as *canonical* entries with an owner list, so the agent reasons
once regardless of how many users share a question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.live.services_live.projection import CanonicalQuestion, MeetingSession, TranscriptTurn
from src.live.services_live.roster import roster_summary

PREP_SUMMARY_KEYS = (
    "org_name",
    "org_details",
    "meeting_type",
    "meeting_type_detail",
    "deal_context",
    "what_they_care_about",
    "risks",
    "whos_on_the_call",
    "historical_context",
)


def _stable(value: Any) -> str:
    """Order-independent serialization used to compare context values across users."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def fold_utterances(turns: list[TranscriptTurn]) -> list[dict[str, Any]]:
    """Join consecutive fragments from one speaker back into whole utterances.

    Real-time transcription emits sub-sentence pieces — "I'm actually hoping that we will have",
    "like, a low latency", "transcription service." — and an evaluation firing between them sees a
    speaker who has started talking but said nothing yet. The agent then recorded "I'm actually
    hoping that we will have" as the answer to a prep question, seconds before the answer was
    spoken.

    This changes only what the model *reads*. It adds no threshold and rejects nothing, so it
    cannot cost a mark that would otherwise have happened — it just stops the model being asked to
    judge half a sentence.

    ``seq`` stays the **first** fragment's, so a citation still resolves against the real
    transcript, and ``seq_end`` gives the reasoner the range to cite when an answer spans several.
    """
    folded: list[dict[str, Any]] = []
    for turn in turns:
        prev = folded[-1] if folded else None
        if prev is not None and prev["speaker"] == turn.speaker:
            prev["text"] = f"{prev['text']} {turn.text}".strip()
            prev["seq_end"] = turn.seq
            continue
        folded.append(
            {
                "seq": turn.seq,
                "seq_end": turn.seq,
                "speaker": turn.speaker,
                "text": turn.text,
                "timestamp": turn.timestamp,
            }
        )
    return folded


def split_prep_context(
    contexts: dict[str, dict[str, Any]],
    *,
    keys: tuple[str, ...] = PREP_SUMMARY_KEYS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Split per-user prep into (shared, per_user).

    A key is shared only when **every** user has it and all values are equal. Everything else stays
    attributed to its owner. With a single user the result is simply (that user's prep, {}).
    """
    if not contexts:
        return {}, {}

    shared: dict[str, Any] = {}
    for key in keys:
        values = [ctx.get(key) for ctx in contexts.values()]
        if any(v is None for v in values):
            continue
        first = _stable(values[0])
        if all(_stable(v) == first for v in values[1:]):
            shared[key] = values[0]

    per_user: dict[str, dict[str, Any]] = {}
    for user_id, ctx in contexts.items():
        remainder = {
            key: ctx.get(key)
            for key in keys
            if ctx.get(key) is not None and key not in shared
        }
        if remainder:
            per_user[user_id] = remainder
    return shared, per_user


@dataclass
class EvaluationContext:
    bot_id: str
    external_event_id: str | None = None
    shared_prep: dict[str, Any] = field(default_factory=dict)
    per_user_prep: dict[str, dict[str, Any]] = field(default_factory=dict)
    roster: dict[str, Any] = field(default_factory=dict)
    questions: list[CanonicalQuestion] = field(default_factory=list)
    transcript_summary: str | None = None
    recent_transcript: list[TranscriptTurn] = field(default_factory=list)
    prior_suggestion_texts: list[str] = field(default_factory=list)
    prior_risk_texts: list[str] = field(default_factory=list)
    prior_fact_checks: list[dict[str, str | None]] = field(default_factory=list)
    memory_window_turns: int = 40
    # Canonical ids are positional, so they are only meaningful against the question set that
    # existed when this snapshot was taken. Compared on apply to catch a set that moved underneath.
    questions_version: int = 0

    def to_prompt_block(self) -> str:
        """Serialize context for the gate prompt (deterministic, no secrets).

        The gate's input is deliberately frozen — same fields, same formatting. Any token
        trimming happens on the reasoner side via :meth:`to_compact_block`.
        """
        return json.dumps(self._block(), indent=2, ensure_ascii=False)

    def to_compact_block(self) -> str:
        """Same information as :meth:`to_prompt_block`, without pretty-printing whitespace.

        Used for the reasoner, the second model call of every cycle — identical content, fewer
        tokens, so no context is lost to save latency.
        """
        return json.dumps(self._block(), separators=(",", ":"), ensure_ascii=False)

    def _block(self) -> dict[str, Any]:
        questions = [
            {
                "id": q.id,
                "text": q.text,
                "status": q.status.value,
                # The answer recorded so far, and which turn it came from. Both are here so the
                # model can recognise a *changed* answer without comparing strings — and so it
                # never re-cites a turn already on record.
                "evidence": q.evidence,
                "evidence_turn_seq": q.evidence_turn_seq,
                "answer_revision": q.revision,
                "owned_by_user_ids": q.user_ids,
            }
            for q in self.questions
        ]
        # ``seq`` is what the reasoner cites as evidence, so a mark points at a real transcript
        # line instead of the model retyping a quote we then have to string-match.
        transcript = fold_utterances(self.recent_transcript)
        return {
            "bot_id": self.bot_id,
            "external_event_id": self.external_event_id,
            "roster": self.roster,
            "shared_prep_summary": self.shared_prep,
            "per_user_prep_summary": self.per_user_prep,
            "prep_questions": questions,
            "transcript_summary": self.transcript_summary,
            "recent_transcript": transcript,
            "prior_suggestions": self.prior_suggestion_texts[-10:],
            "prior_risks": self.prior_risk_texts[-10:],
            "prior_fact_checks": self.prior_fact_checks[-10:],
        }


def build_evaluation_context(
    session: MeetingSession,
    *,
    memory_window_turns: int | None = None,
) -> EvaluationContext:
    window = memory_window_turns or 40
    summary, recent = session.prompt_transcript(window)
    shared_prep, per_user_prep = split_prep_context(session.context)
    return EvaluationContext(
        bot_id=session.bot_id,
        external_event_id=session.external_event_id,
        shared_prep=shared_prep,
        per_user_prep=per_user_prep,
        roster=roster_summary(session),
        questions=session.canonical_questions(),
        transcript_summary=summary,
        recent_transcript=recent,
        # Deduped: every user receives the same suggestion text, so show it to the agent once.
        prior_suggestion_texts=list(dict.fromkeys(s.text for s in session.suggestions)),
        prior_risk_texts=list(dict.fromkeys(r.text for r in session.risks)),
        prior_fact_checks=[
            {
                "claim": fc.claim,
                "verdict": fc.verdict.value,
                "detail": fc.detail,
            }
            for fc in session.fact_checks
        ],
        memory_window_turns=window,
        questions_version=session.questions_version,
    )

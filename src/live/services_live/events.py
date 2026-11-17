"""Domain events for the meeting session (the append-only, event-sourced source of truth).

The live meeting state is never mutated directly; instead, immutable ``MeetingEvent``s are
appended to an ``EventLog`` and the current state is a projection (fold) over them. This keeps
reconnection, rehydration, and replay-based evaluation as the same primitive rather than
bespoke code paths.

Events are discriminated on ``kind``. ``StoredEvent`` is the envelope the log wraps around each
event, adding a per-meeting monotonic ``offset``, a unique ``event_id``, and a wall-clock ``ts``.
Everything here is Pydantic so events serialize cleanly to JSON for durable storage later.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from src.live.schemas_live.common import FactCheckVerdict, QuestionStatus


def new_event_id() -> str:
    return uuid.uuid4().hex


class EventModel(BaseModel):
    """Base for domain events. ``extra="forbid"`` keeps the event schema strict — events are
    our own internal contract, so unexpected fields indicate a bug, not forward-compat."""

    model_config = ConfigDict(extra="forbid")


class PrepQuestion(EventModel):
    id: str
    text: str


class SessionUserRecord(EventModel):
    """One coached user, as recorded on ``MeetingStarted`` for the life of the session."""

    user_id: str
    name: str = ""
    soa_connected: bool = False
    participant_uuid: str | None = None


class MeetingStarted(EventModel):
    kind: Literal["meeting_started"] = "meeting_started"
    bot_id: str
    external_event_id: str | None = None
    users: list[SessionUserRecord] = Field(default_factory=list)
    meeting_type: str | None = None
    meeting_url: str | None = None
    started_at: str | None = None


class PrepQuestionsLoaded(EventModel):
    """Prep for exactly one user. A multi-user session emits one of these per user."""

    kind: Literal["prep_questions_loaded"] = "prep_questions_loaded"
    user_id: str
    questions: list[PrepQuestion] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class TranscriptSegmentAppended(EventModel):
    kind: Literal["transcript_segment_appended"] = "transcript_segment_appended"
    seq: int
    speaker: str
    speaker_is_host: bool = False
    participant_uuid: str | None = None
    text: str
    start_ms: int
    duration_ms: int
    timestamp: float
    turn_id: str | None = None


class QuestionGroupMember(EventModel):
    """One user's copy of a question inside a semantic group."""

    user_id: str
    question_id: str


class QuestionGroupsResolved(EventModel):
    """Which per-user questions are the *same* question, decided semantically.

    Two users' prep can word one question differently ("What is your ATS?" vs "Which ATS do you
    use?"). Character comparison calls those different; the model calls them the same. Resolved in
    the background after the session is already live, so it never delays the handshake.
    """

    kind: Literal["question_groups_resolved"] = "question_groups_resolved"
    groups: list[list[QuestionGroupMember]] = Field(default_factory=list)


class UserPrepReplaced(EventModel):
    """One user's prep re-fetched mid-session and swapped in wholesale.

    Distinct from ``PrepQuestionsLoaded`` on purpose: that one *loads* (and is the session-start
    path, unchanged), this one *replaces*. Everything the user previously had is dropped —
    questions, order, context — and every question comes back unmarked.
    """

    kind: Literal["user_prep_replaced"] = "user_prep_replaced"
    user_id: str
    questions: list[PrepQuestion] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class PrepQuestionAnswered(EventModel):
    """A prep question now has an answer from the meeting — or a different one than before.

    Addressed; ``question_id`` is unique within ``user_id``'s list, not session-wide. One logical
    question owned by several users produces one of these per owner, all citing the same turn.

    ``revision`` counts answers for this question: 1 is the first, 2+ means a participant said
    something that changed it. The distinction is derived from stored state rather than asked of
    the model, so it cannot be mis-reported.

    Deliberately **not** named ``QuestionAnswered`` — that is the answering lane, for factual
    questions asked out loud in the room. Sharing a name would make the two indistinguishable in
    logs and in the event union.
    """

    kind: Literal["prep_question_answered"] = "prep_question_answered"
    user_id: str
    question_id: str
    # Verbatim text of the transcript turn that answers the question — never a paraphrase, and
    # never the turn where it was asked.
    answer: str
    # Where the answer was said. ``_end`` is set when it spanned several transcript segments, so a
    # client can highlight the whole stretch rather than the first fragment of it.
    evidence_turn_seq: int
    evidence_turn_seq_end: int | None = None
    revision: int = 1


class SuggestionEmitted(EventModel):
    kind: Literal["suggestion_emitted"] = "suggestion_emitted"
    user_id: str
    suggestion_id: str
    text: str
    rationale: str | None = None
    linked_question_id: str | None = None


class SuggestionRefined(EventModel):
    kind: Literal["suggestion_refined"] = "suggestion_refined"
    user_id: str
    suggestion_id: str
    text: str
    rationale: str | None = None


class QuestionAdded(EventModel):
    kind: Literal["question_added"] = "question_added"
    user_id: str
    question_id: str
    text: str


class FactCheckEmitted(EventModel):
    """Broadcast: the claim was made in the shared transcript, so it concerns every user."""

    kind: Literal["fact_check_emitted"] = "fact_check_emitted"
    fact_check_id: str
    claim: str
    verdict: FactCheckVerdict
    confidence: float | None = None
    detail: str | None = None
    sources: list[str] = Field(default_factory=list)


class QuestionAnswered(EventModel):
    """A factual question asked out loud in the meeting, and the answer given.

    Broadcast: the room asked, so the room gets the answer. Distinct from ``FactCheckEmitted``,
    which passes a verdict on a claim someone *asserted* — there is no true/false to return when
    the question is "where is the Great Wall of China".
    """

    kind: Literal["question_answered"] = "question_answered"
    answer_id: str
    question: str
    answer: str
    source: Literal["knowledge", "web"] = "knowledge"


class AgendaRiskEmitted(EventModel):
    """``user_id`` None = broadcast (drift concerns the whole meeting); set = addressed."""

    kind: Literal["agenda_risk_emitted"] = "agenda_risk_emitted"
    risk_id: str
    text: str
    user_id: str | None = None


class CompanyDetected(EventModel):
    """Broadcast: the company was named in the shared transcript."""

    kind: Literal["company_detected"] = "company_detected"
    company_id: str
    company_name: str
    title: str
    text: str
    es_id: str | None = None
    li_urn: str | None = None
    li_name: str | None = None
    li_universalname: str | None = None
    li_industries: list[str] = Field(default_factory=list)


class ChatMessageAppended(EventModel):
    """One turn in a single user's private chat thread. Histories never cross users."""

    kind: Literal["chat_message_appended"] = "chat_message_appended"
    user_id: str
    role: Literal["user", "assistant"]
    message_id: str
    text: str


class AgentEvaluationTriggered(EventModel):
    """Internal signal that the orchestrator should run (gate -> reasoner). No LLM in P1.6."""

    kind: Literal["agent_evaluation_triggered"] = "agent_evaluation_triggered"
    reason: Literal["turn_threshold", "time_threshold"]
    window_turn_count: int
    total_turn_count: int


class TranscriptSummaryUpdated(EventModel):
    """Rolling summary of transcript turns compacted out of the live prompt window."""

    kind: Literal["transcript_summary_updated"] = "transcript_summary_updated"
    summary_text: str
    compacted_turn_count: int
    turns_compacted: int


MeetingEvent = Annotated[
    Union[
        MeetingStarted,
        PrepQuestionsLoaded,
        TranscriptSegmentAppended,
        AgentEvaluationTriggered,
        TranscriptSummaryUpdated,
        QuestionGroupsResolved,
        UserPrepReplaced,
        QuestionAnswered,
        PrepQuestionAnswered,
        SuggestionEmitted,
        SuggestionRefined,
        QuestionAdded,
        FactCheckEmitted,
        AgendaRiskEmitted,
        CompanyDetected,
        ChatMessageAppended,
    ],
    Field(discriminator="kind"),
]


class StoredEvent(BaseModel):
    """Envelope written to the log: a domain event plus its position and identity."""

    model_config = ConfigDict(extra="forbid")

    offset: int
    event_id: str
    ts: float
    meeting_id: str
    event: MeetingEvent

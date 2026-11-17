"""Outbound messages (server -> client): what the agent pushes back over the socket.

- ``session_ready``   : ack after session start; carries loaded call-prep questions per user.
- ``mark_question``   : a prep question got its answer, or a changed one (left panel).
- ``question_added``  : a question was added to one user's list mid-meeting.
- ``suggestion``      : a follow-up the agent suggests (right panel).
- ``fact_check``      : a claim the agent checked (right panel).
- ``answer``          : a factual question asked in the meeting, answered (right panel).
- ``risk``            : agenda/scope drift warning for the host (right panel).
- ``company_detected``: a company surfaced in conversation and resolved.
- ``chat_token``      : one streamed token of a 1-1 chat reply.
- ``chat_message``    : the final, complete 1-1 chat reply.
- ``error``           : a structured error (e.g. an invalid inbound frame).

**Addressing.** One socket carries events for every coached user on the meeting, and the client
filters by ``user_id``. That field is present on *every* outbound message:

- a **non-null** ``user_id`` means the frame belongs to that user alone;
- an explicit **null** means broadcast — every user on the session should see it.

The field is never omitted. A dropped field would silently turn an addressed frame into a
broadcast, which is a cross-user leak; making it required-but-nullable turns that class of bug
into a validation error instead.

All outbound messages are discriminated on ``type`` and share one serializer (``dump_outbound``).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union, Any

from pydantic import Field, TypeAdapter

from src.live.schemas_live.common import (
    FactCheckVerdict,
    LiveModel,
    QuestionStatus,
    QuestionView,
    UserQuestions,
    new_id,
)


class SessionReady(LiveModel):
    """Ack for ``transcript_session_start``; one entry per coached user.

    Every user from the session-start roster appears here, including those with no prep at all —
    they carry ``questions: []`` rather than being omitted, so the client can tell "in the session
    with no prep" apart from "not in the session".
    """

    type: Literal["session_ready"] = "session_ready"
    bot_id: str = Field(alias="botId")
    users: list[UserQuestions] = Field(default_factory=list)


class MarkQuestion(LiveModel):
    """Addressed. A prep question now has an answer from the meeting — or a different one.

    One frame per owning user: two users whose prep worded the same question differently each get
    their own, carrying the same ``evidence``. ``user_id`` is never null on this frame.

    ``status`` describes **what just happened**, which is why it is wider than
    :class:`QuestionStatus`:

    - ``answered``       — first answer recorded for this question
    - ``answer_updated`` — a participant later said something that changed it, with or without
      anyone re-asking

    Clients should upsert on ``(user_id, question_id)`` rather than append. Note that a
    ``session_ready`` snapshot reports the question's *resting* state, so a question answered three
    times appears there as ``answered`` — ``answer_updated`` only ever appears on a live frame.

    ``answer_text`` is the answer in plain words — cleaned of stutters, crosstalk and false starts.
    ``evidence_turn_seq`` / ``evidence_turn_seq_end`` say which transcript segments it came from,
    so a client can highlight the source.
    """

    type: Literal["mark_question"] = "mark_question"
    user_id: str
    question_id: str
    status: Literal["pending", "answered", "answer_updated"] = "answered"
    # The answer in plain words — what the host reads. Named for what it is: this is the answer,
    # not a quote from the transcript. The seq range below is what grounds it.
    answer_text: str | None = None
    # Which transcript segments it came from. ``evidence_turn_seq_end`` is null when the answer
    # fitted in one segment, and set when it ran across several, so the client can highlight the
    # whole stretch in the transcript view.
    evidence_turn_seq: int | None = None
    evidence_turn_seq_end: int | None = None


class QuestionAdded(LiveModel):
    """Addressed. ``id`` is unique within this user's list only."""

    type: Literal["question_added"] = "question_added"
    user_id: str
    id: str
    text: str
    status: QuestionStatus = QuestionStatus.PENDING


class Suggestion(LiveModel):
    """Addressed. ``linked_question_id`` indexes this user's question list.

    Re-emitted with the same ``id`` when refined from chat, so clients must upsert on ``id``
    rather than append.
    """

    type: Literal["suggestion"] = "suggestion"
    user_id: str
    id: str = Field(default_factory=lambda: new_id("sug"))
    text: str
    rationale: str | None = None
    linked_question_id: str | None = None


class FactCheck(LiveModel):
    """Broadcast: a claim made in the shared meeting concerns everyone. ``user_id`` is null."""

    type: Literal["fact_check"] = "fact_check"
    user_id: str | None = None
    id: str = Field(default_factory=lambda: new_id("fc"))
    claim: str
    verdict: FactCheckVerdict
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str | None = None
    sources: list[str] = Field(default_factory=list)


class Risk(LiveModel):
    """Either. Agenda drift about the meeting broadcasts (null); a user-specific risk addresses."""

    type: Literal["risk"] = "risk"
    user_id: str | None = None
    id: str = Field(default_factory=lambda: new_id("risk"))
    text: str


class Answer(LiveModel):
    """Broadcast: a factual question was asked out loud and this is the answer.

    Separate from ``fact_check`` on purpose — that carries a verdict on a claim someone made,
    which is meaningless for a question. ``source`` tells the host whether this is settled
    knowledge or something looked up just now.
    """

    type: Literal["answer"] = "answer"
    user_id: str | None = None
    id: str = Field(default_factory=lambda: new_id("ans"))
    question: str
    answer: str
    source: Literal["knowledge", "web"] = "knowledge"


class CompanyInfo(LiveModel):
    es_id: str | None = None
    li_urn: str | None = None
    li_name: str | None = None
    li_universalname: str | None = None
    li_industries: list[str] = Field(default_factory=list)


class CompanyDetectedOut(LiveModel):
    """Broadcast: the company was named in the shared transcript. ``user_id`` is null."""

    type: Literal["company_detected"] = "company_detected"
    user_id: str | None = None
    id: str
    company_name: str
    title: str
    text: str
    company_info: CompanyInfo


class ChatToken(LiveModel):
    """Addressed to the user whose ``chat_message`` started this turn."""

    type: Literal["chat_token"] = "chat_token"
    user_id: str
    message_id: str = Field(alias="messageId")
    delta: str


class ChatMessageOut(LiveModel):
    """Addressed. ``id`` equals the ``messageId`` of the preceding token stream."""

    type: Literal["chat_message"] = "chat_message"
    user_id: str
    id: str = Field(default_factory=lambda: new_id("msg"))
    text: str
    additional_data: dict[str, Any] | None = None


class ErrorMessage(LiveModel):
    """Addressed when the offending frame identified a user, broadcast (null) otherwise."""

    type: Literal["error"] = "error"
    user_id: str | None = None
    code: str
    message: str


OutboundMessage = Annotated[
    Union[
        SessionReady,
        MarkQuestion,
        QuestionAdded,
        Suggestion,
        Answer,
        FactCheck,
        Risk,
        CompanyDetectedOut,
        ChatToken,
        ChatMessageOut,
        ErrorMessage,
    ],
    Field(discriminator="type"),
]

_outbound_adapter: TypeAdapter[OutboundMessage] = TypeAdapter(OutboundMessage)


def dump_outbound(message: OutboundMessage) -> str:
    """Serialize any outbound message to JSON (camelCase aliases where defined)."""
    return _outbound_adapter.dump_json(message, by_alias=True).decode("utf-8")


def outbound_json_schema() -> dict:
    return _outbound_adapter.json_schema()

"""Shared primitives for the live message contract: id generation, enums, base model,
and the wire view of a call-prep question.

These types are part of the wire contract and are shared by both the inbound
(client -> server) and outbound (server -> client) message models.
"""

from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    """Stable, short, prefixed id (e.g. ``sug_9f3a1c...``) for outbound artifacts."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class QuestionStatus(str, Enum):
    """Resting state of a prep question. Two values, deliberately.

    ``answer_updated`` is **not** here: it describes a transition, not a state. A question answered
    three times is still, at rest, ``answered`` — which is what a ``session_ready`` snapshot must
    report on reconnect. The wire carries the transition separately (see ``MarkQuestionStatus``).
    """

    PENDING = "pending"
    ANSWERED = "answered"


class FactCheckVerdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNCLEAR = "unclear"


class LiveModel(BaseModel):
    """Base for every live message.

    ``extra="ignore"`` keeps the contract forward-compatible: if the Node backend adds a
    field we do not yet consume, validation still succeeds instead of hard-failing mid-meeting.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class QuestionView(LiveModel):
    """Wire representation of a single call-prep question and its live status.

    Ids are unique **within one user's** question list, not across the session — two users can
    both own a ``q1``. Always pair a question id with the ``user_id`` of the frame carrying it.
    """

    id: str
    text: str
    status: QuestionStatus = QuestionStatus.PENDING


class SessionUser(LiveModel):
    """One coached user attached to the session.

    A single meeting is coached on several screens at once: each entry here is a person who will
    receive addressed events (their own questions, suggestions, chat). Speakers in the transcript
    who are *not* in this list are meeting participants — context for the agent, never recipients.
    """

    user_id: str = Field(min_length=1, alias="userId")
    # Display/speaker name, used to match transcript speakers back to this user.
    name: str = Field(default="", alias="name")
    # Per-user, not per-session: some users have SOA-connected prep, others temporary prep.
    soa_connected: bool = Field(alias="soaConnected")
    # Optional and preferred over ``name`` for speaker matching when the pipeline supplies it.
    participant_uuid: str | None = Field(default=None, alias="participantUuid")


class UserQuestions(LiveModel):
    """One user's prep questions, as carried by ``session_ready``."""

    user_id: str
    questions: list[QuestionView] = Field(default_factory=list)

"""Inbound messages (client -> server): Node transcript pipeline wire format.

One WebSocket per live *meeting*, coached on one or more user screens. The client sends
``transcript_session_start`` first — carrying the full roster of coached users — then streams
``transcript_segment`` frames, and may send ``transcript_session_end``. ``chat_message`` is used
for 1-1 user chat with the coach on the same socket and must say which user is talking.

Field names match the Node backend (camelCase aliases). All messages are discriminated on
``type``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, TypeAdapter

from src.live.schemas_live.common import LiveModel, SessionUser


class TranscriptSegment(LiveModel):
    seq: int
    speaker: str = Field(min_length=1)
    speaker_is_host: bool = Field(alias="speakerIsHost")
    participant_uuid: str | None = Field(default=None, alias="participantUuid")
    text: str = Field(min_length=1)
    start_ms: int = Field(alias="startMs")
    duration_ms: int = Field(alias="durationMs")


class TranscriptSessionStartMessage(LiveModel):
    """First frame: binds this socket to a session and triggers call-prep load for every user.

    ``users`` replaces the old single ``userId`` / ``soaConnected`` pair. Prep is loaded per user
    and scoped by ``(eventId, userId)`` — one calendar event yields one ``meetings`` row per
    participant, so an unscoped lookup would resolve to an arbitrary user's prep.
    """

    type: Literal["transcript_session_start"] = "transcript_session_start"
    bot_id: str = Field(min_length=1, alias="botId")
    event_id: str | None = Field(default=None, alias="eventId")
    users: list[SessionUser] = Field(min_length=1)
    meeting_type: str | None = Field(default=None, alias="meetingType")
    meeting_url: str | None = Field(default=None, alias="meetingUrl")
    started_at: str | None = Field(default=None, alias="startedAt")


class TranscriptSegmentMessage(LiveModel):
    """One finalized transcript segment from the Node transcription pipeline.

    Shared by every coached user on this session — the transcript is the meeting, not a user.
    """

    type: Literal["transcript_segment"] = "transcript_segment"
    bot_id: str = Field(min_length=1, alias="botId")
    event_id: str | None = Field(default=None, alias="eventId")
    meeting_url: str | None = Field(default=None, alias="meetingUrl")
    segment: TranscriptSegment


class TranscriptSessionEndMessage(LiveModel):
    """Session teardown signal from Node."""

    type: Literal["transcript_session_end"] = "transcript_session_end"
    bot_id: str = Field(min_length=1, alias="botId")
    event_id: str | None = Field(default=None, alias="eventId")


class ChatMessageIn(LiveModel):
    """A user message in the 1-1 chat with the agent (same socket as the meeting).

    ``userId`` is required: several users share this socket, and the reply, its token stream, and
    any question/suggestion edits it produces are all addressed back to that user alone.
    """

    type: Literal["chat_message"] = "chat_message"
    user_id: str = Field(min_length=1, alias="userId")
    text: str = Field(min_length=1)
    id: str | None = None


class UserConnectedOnLiveScreen(LiveModel):
    """One coached user just opened their live-coach screen.

    Not a join: the roster is already complete from ``transcript_session_start``, and every user's
    prep was loaded there. This is a **refresh** — the user may have edited their prep right up
    until opening the screen, so we re-read it and swap in the latest.

    It may arrive repeatedly for the same user (reconnect, refresh, second tab), and it may never
    arrive at all for someone already sitting on the screen when the session started. Neither is a
    fault: the session-start load already covered them.

    Produces no outbound frame — the client is the side that already has the new prep.
    """

    type: Literal["user_connected_on_live_screen"] = "user_connected_on_live_screen"
    user_id: str = Field(min_length=1, alias="userId")
    # Per-user, and authoritative: it decides which prep table this reload reads.
    soa_connected: bool = Field(alias="soaConnected")
    # Checked against the session's own event id. A mismatch means the client has crossed wires
    # between two meetings, and acting on it would load prep from the wrong one.
    event_id: str = Field(min_length=1, alias="eventId")


InboundMessage = Annotated[
    Union[
        TranscriptSessionStartMessage,
        TranscriptSegmentMessage,
        TranscriptSessionEndMessage,
        ChatMessageIn,
        UserConnectedOnLiveScreen,
    ],
    Field(discriminator="type"),
]

_inbound_adapter: TypeAdapter[InboundMessage] = TypeAdapter(InboundMessage)


def parse_inbound(raw: str | bytes | dict) -> InboundMessage:
    """Validate a raw frame into a typed model.

    Raises ``pydantic.ValidationError`` on unknown/invalid messages.
    """
    if isinstance(raw, (str, bytes, bytearray)):
        return _inbound_adapter.validate_json(raw)
    return _inbound_adapter.validate_python(raw)


def inbound_json_schema() -> dict:
    return _inbound_adapter.json_schema()

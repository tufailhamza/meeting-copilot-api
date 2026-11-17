from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TranscriptText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(..., description="Spoken text for this turn.")


class TranscriptTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_name: str = Field(..., min_length=1)
    speaker_uuid: str = Field(..., min_length=1)
    speaker_user_uuid: str | None = None
    speaker_is_host: bool = False
    timestamp_ms: int
    duration_ms: int
    transcription: TranscriptText


class UserMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    user_id: int = Field(..., alias="userId", description="Stable participant identifier.")
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1)


class ExtractActionItemsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_timestamp_ms: int = Field(
        ...,
        description="Meeting start time (epoch ms). Anchor for transcript segment start/end in seconds.",
    )
    transcript: list[TranscriptTurn] = Field(..., min_length=1)
    user_metadata: list[UserMetadata] | None = Field(
        default=None,
        description="When provided, action items are extracted per user. When omitted, only transcriptions are returned.",
    )


class TranscriptionSegmentOut(BaseModel):
    """Cleaned diarized line: full speaker name and timeline in seconds from meeting start."""

    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    start: float = Field(..., description="Start time in seconds relative to start_timestamp_ms.")
    end: float = Field(..., description="End time in seconds relative to start_timestamp_ms.")


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    priority: str = Field(..., pattern="^(low|medium|high)$")


class ActionItemsByUser(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: int
    speaker_name: str
    email: str
    action_items: list[ActionItem]


class ExtractActionItemsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcriptions: list[TranscriptionSegmentOut]
    action_items_by_user: list[ActionItemsByUser] | None = Field(
        default=None,
        description="Present only when user_metadata was sent in the request.",
    )

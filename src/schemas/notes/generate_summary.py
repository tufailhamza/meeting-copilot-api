from pydantic import BaseModel, ConfigDict, Field

from src.schemas.transcription.transcribe import TranscriptionSegment


class GenerateSummaryRequest(BaseModel):
    """Body: diarized transcript segments to summarize (snake_case keys only)."""

    model_config = ConfigDict(extra="forbid")

    recording_id: str = Field(
        ...,
        min_length=1,
        description="Recording identifier. Echoed in the response.",
    )
    transcriptions: list[TranscriptionSegment] = Field(
        ...,
        min_length=1,
        description="Ordered diarized lines with speaker, text, and time bounds in seconds.",
    )


class GenerateSummaryResponse(BaseModel):
    """Meeting summary as a Markdown string (unordered list items only)."""

    model_config = ConfigDict(extra="forbid")

    recording_id: str = Field(
        ...,
        description="Same id as in the request.",
    )
    summary: str = Field(
        ...,
        description="Markdown for **react-markdown**: only `- ` list lines, one bullet per line, real newline separators (optional single blank line between groups); no headings/numbered lists/raw HTML; grounded in the transcript.",
    )

from pydantic import BaseModel, ConfigDict, Field


class TranscribeResponse(BaseModel):
    message: str = Field(examples=["this function transcribes :)"])
    provider: str = Field(
        default="assembly",
        description='Engine used: "openai" or "assembly"',
    )


class TranscriptionSegment(BaseModel):
    """One diarized line; JSON keys must be snake_case (``extra`` forbids aliases / extras)."""

    model_config = ConfigDict(extra="forbid")

    speaker: str
    text: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")


class RecordingTranscribeRequest(BaseModel):
    """Body: recording id, signed URL, MIME type. Keys may be snake_case or camelCase (mutually exclusive names per field)."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    recording_id: str = Field(
        ...,
        min_length=1,
        alias="recordingId",
        description="Your recording id (e.g. UUID). Echoed unchanged in the response.",
    )
    signed_url: str = Field(
        ...,
        min_length=1,
        alias="signedUrl",
        description="HTTPS URL to the recording (e.g. GCS signed URL).",
    )
    content_type: str = Field(
        ...,
        min_length=1,
        alias="contentType",
        description='MIME type: must start with audio/ or video/, e.g. audio/mpeg, audio/flac, audio/m4a, video/mp4, video/webm, video/x-matroska.',
        examples=["audio/mpeg", "audio/flac", "video/mp4", "video/x-matroska"],
    )


class TranscribeAcceptedResponse(BaseModel):
    """Returned immediately after audio is prepared; full result is POSTed to TRANSCRIBER_URL."""

    model_config = ConfigDict(extra="forbid")

    recording_id: str = Field(
        ...,
        description="Same id as the request; processing continues in the background.",
    )
    status: str = Field(
        default="accepted",
        description="Recording is queued for transcription, transliteration, and host labeling.",
    )


class NonDiarizeTranscribeResponse(BaseModel):
    """Synchronous plain-text transcription (no speakers, no callback)."""

    model_config = ConfigDict(extra="forbid")

    recording_id: str = Field(
        ...,
        description="Same id as the request.",
    )
    transcript_text: str = Field(
        ...,
        description="Full audio-to-text transcript with no speaker diarization.",
    )


class RecordingTranscribeResponse(BaseModel):
    """Callback payload to TRANSCRIBER_URL (snake_case JSON)."""

    model_config = ConfigDict(extra="forbid")

    recording_id: str = Field(
        ...,
        description="Same value you sent in the request; not AssemblyAI’s transcript id.",
    )
    transcriptions: list[TranscriptionSegment] = Field(
        ...,
        description="Diarized lines; text transliterated to Latin where applicable; detected host speaker relabeled as \"Host\".",
    )

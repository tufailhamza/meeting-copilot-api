from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schemas.notes.generate_summary import GenerateSummaryRequest


class NotesTemplateSection(BaseModel):
    """One slot in an optional notes template (title + content guide)."""

    model_config = ConfigDict(extra="forbid")

    section_title: str = Field(
        ...,
        min_length=1,
        description="Exact heading to use for this notes section.",
    )
    general_pattern: str = Field(
        ...,
        min_length=1,
        description="Guide for what content belongs in this section.",
    )


class NotesTemplate(BaseModel):
    """Optional structure that pins section titles and content patterns."""

    model_config = ConfigDict(extra="forbid")

    template_title: str = Field(
        ...,
        min_length=1,
        description="Human label for the template (hint for meeting title).",
    )
    template_sections: list[NotesTemplateSection] = Field(
        ...,
        min_length=1,
        description="Ordered section slots; notes must follow this order and titles.",
    )


class GenerateNotesRequest(GenerateSummaryRequest):
    """Recording id + diarized segments; optional template pins section structure."""

    template: NotesTemplate | None = Field(
        default=None,
        description="When set, fill these sections in order; when omitted, auto-structure.",
    )


class NoteDuration(BaseModel):
    """Single contiguous time range for one section (float seconds, same as transcript)."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(..., description="Section start time in seconds (transcript timeline).")
    end: float = Field(..., description="Section end time in seconds (exclusive upper bound semantics as in transcript).")


class NoteSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_title: str = Field(..., min_length=1, description="Minimal, specific title for this block.")
    section_description: str = Field(
        ...,
        min_length=1,
        description="Markdown body only (no # headings); grounded in this section’s time range.",
    )
    duration: NoteDuration = Field(..., description="Exactly one contiguous range for this section.")

    @model_validator(mode="after")
    def duration_order(self) -> NoteSection:
        if self.duration.start >= self.duration.end:
            raise ValueError("duration.start must be strictly less than duration.end")
        return self


class GenerateNotesResponse(BaseModel):
    """Structured meeting notes: whole-meeting title & blurb, themes, and timed sections."""

    model_config = ConfigDict(extra="forbid")

    recording_id: str = Field(
        ...,
        description="Same id as in the request.",
    )
    meeting_title: str = Field(
        ...,
        min_length=1,
        description="AI-generated title for the entire meeting.",
    )
    meeting_short_description: str = Field(
        ...,
        min_length=1,
        max_length=150,
        description="Single-sentence overview of the whole meeting; at most 150 characters.",
    )
    themes: list[str] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="Main themes, keywords, or short topic phrases (3–8 items).",
    )
    sections: list[NoteSection] = Field(
        ...,
        description="Chronological sections; each has one duration (float seconds).",
    )

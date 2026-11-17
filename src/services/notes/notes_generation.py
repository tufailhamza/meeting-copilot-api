"""Partition transcript into themed sections with Markdown descriptions (OpenAI gpt-4.1 by default)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import APIError, OpenAI
from pydantic import ValidationError

from src.config import settings
from src.schemas.notes.generate_notes import (
    GenerateNotesResponse,
    NoteDuration,
    NotesTemplate,
    NoteSection,
)
from src.schemas.transcription.transcribe import TranscriptionSegment

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPT_JSON_CHARS = 240_000
_MEETING_SHORT_DESC_MAX_CHARS = 150
_THEME_ITEM_MAX_CHARS = 120

_SYSTEM_PROMPT = """You are NOTES_GENERATION_AGENT, a production-grade meeting-notes engine.

## What you receive
You will be given:
1) A `recording_id` string (for traceability only; do not treat as secret).
2) A chronological diarized transcript as a JSON array of turns in time order.
Each turn includes:
- `speaker`: label or name as provided (do not rename unless the transcript itself contains a clearer proper name used consistently).
- `text`: what was said.
- `start` and `end`: floating-point timestamps in **seconds** on one shared timeline for the recording.

The transcript is the **only** source of truth.

## Primary objective
Produce structured meeting notes by:
- Writing a **whole-meeting** `meeting_title`, a **single-sentence** `meeting_short_description` (≤150 characters), and **3–8** `themes` keywords/phrases grounded in the full transcript.
- Identifying major topics and how the conversation progresses over time.
- Partitioning the transcript into **sections**. Each section is **one contiguous** discussion block on the timeline.
- Writing a **Markdown** `section_description` that summarizes **only** what occurs within that section’s assigned time range.

## Hard requirements (all mandatory)

### A) Grounding / hallucination (strict)
- Use ONLY information supported by the transcript.
- Do NOT invent facts, people, organizations, numbers, dates, commitments, decisions, action items, or timelines not clearly present.
- If uncertain, use careful phrasing (“The speakers discuss…”, “It is not fully specified…”).
- Do NOT let a section description rely on transcript material **outside** that section’s `[duration.start, duration.end]` window.

### B) Sectioning (strict)
- Use **multiple sections** when the meeting clearly shifts topics (typical: several sections; very short audio may be **1** section).
- Order sections **chronologically** by `duration.start` (ascending).
- Each section has **exactly one** `duration` object: `start` and `end` as JSON **numbers** (float seconds), **not** strings, **not** `HH:MM:SS`.
- **No** disjoint ranges: one continuous interval per section. If a theme reappears later after unrelated talk, use a **separate** section with its own `duration`.

### C) Time alignment (strict)
- `duration.start` = the **minimum** `start` among transcript turns that belong to this section.
- `duration.end` = the **maximum** `end` among those same turns.
- Ensure `duration.start < duration.end`.
- Do not invent timestamps.

### D) Section titles (strict)
- `section_title`: **minimal word count** while staying **specific** (avoid vague titles like “Discussion” or “Updates”).
- No timestamps in titles.

### E) Section descriptions — Markdown (strict)
- `section_description` MUST be real **Markdown** suitable for rendering: paragraphs, bullet lists when helpful, **bold** for key terms, short quoted phrases when reflecting transcript wording.
- **Forbidden:** Markdown headings. Do **not** use lines starting with `#`, `##`, `###`, etc. Do not fake headings with ALL-CAPS title lines.
- Structure content with paragraphs and bullets instead of headings.

### F) Content priorities (when present in that window)
Emphasize, without inventing: decisions, agreements, action items, key topics/constraints/risks/metrics, outcomes, and notable contributions (attribute using given speaker labels).

### G) Whole-meeting metadata (strict)
- **`meeting_title`**: One concise, specific title for the **entire** conversation (not per section). No timestamps. Avoid generic titles (“Meeting”, “Discussion”). Ground only in the transcript.
- **`meeting_short_description`**: **Exactly one sentence** describing the **whole** meeting for someone who was not there. **Plain text only**: no line breaks, bullets, or markdown. **Maximum 150 characters** (count characters, not words). Must still be faithful; do not invent.
- **`themes`**: **Between 3 and 8** strings. Each is a **short** keyword or topic phrase clearly supported by the transcript (entities, regions, policies, etc.). **Distinct** entries; minimal overlap; no empty strings; no vague fillers like “important” or “updates” unless the transcript literally centers on that.

## Output format (strict machine-readable JSON)
Return **ONLY** valid JSON (no markdown fences around the JSON, no commentary). Top-level object:

{
  "meeting_title": string,
  "meeting_short_description": string,
  "themes": [ string, ... ],
  "sections": [
    {
      "section_title": string,
      "section_description": string,
      "duration": { "start": number, "end": number }
    }
  ]
}

Rules:
- `themes` MUST contain **at least 3** and **at most 8** items.
- For a **non-empty** transcript, `sections` MUST have **at least one** entry.
- Every `section_description` must be non-empty for non-empty transcripts.
- Use JSON numbers for `start` and `end` (not strings).

## Self-check (apply before returning)
- `meeting_title`, `meeting_short_description`, and every `themes[]` entry are grounded in the transcript as a whole.
- `meeting_short_description` is a single sentence and **≤ 150 characters**.
- `themes` length is 3–8 with distinct, substantive phrases.
- Each section description uses only material from turns overlapping `[duration.start, duration.end]`.
- Sections are chronological; one contiguous `duration` per section.
- No `#` heading lines in any `section_description`.
- If the transcript is extremely short, one section spanning from the first turn’s `start` to the last turn’s `end` is acceptable."""


_TEMPLATE_SYSTEM_PROMPT = """You are NOTES_TEMPLATE_FILL_AGENT, a production-grade meeting-notes engine that fills a caller-supplied template from a diarized transcript.

## What you receive
You will be given:
1) A `recording_id` string (for traceability only; do not treat as secret).
2) A chronological diarized transcript as a JSON array of turns in time order.
   Each turn includes `speaker`, `text`, and `start`/`end` as floating-point **seconds** on one shared timeline.
3) A `TEMPLATE` JSON object with:
   - `template_title`: a human label for this note structure (optional hint for the whole-meeting title).
   - `template_sections`: an ordered array of slots. Each slot has:
     - `section_title`: the exact heading you MUST use for that notes section.
     - `general_pattern`: the content guide for what belongs in that section’s Markdown body.

The transcript is the **only** source of factual truth. The template defines **structure and intent only** — it does not invent facts.

## Primary objective
Produce structured meeting notes whose `sections` array **exactly matches** the template slots:
- Same **count**, same **order**, and each `section_title` **exactly equal** (character-for-character after trimming) to the corresponding template `section_title`.
- For each slot, write `section_description` Markdown that follows that slot’s `general_pattern`, using only transcript-supported material.
- Also produce whole-meeting `meeting_title`, `meeting_short_description` (≤150 characters), and **3–8** `themes` grounded in the full transcript.

Do **not** invent your own section list. Do **not** rename, reorder, merge, split, or drop template slots. Do **not** add extra sections.

## Hard requirements (all mandatory)

### A) Grounding / hallucination (strict)
- Use ONLY information supported by the transcript.
- Do NOT invent facts, people, organizations, numbers, dates, commitments, decisions, action items, owners, or deadlines not clearly present.
- If a template slot’s pattern asks for something the transcript does not support, write a brief honest note (e.g. that the topic was not discussed or not specified) — still non-empty Markdown — and do not fabricate filler.
- Prefer the speakers’ own wording for challenges, quotes, and commitments when the pattern asks for it.

### B) Template sectioning (strict)
- `sections.length` MUST equal `template_sections.length`.
- `sections[i].section_title` MUST equal `template_sections[i].section_title` exactly (trim whitespace only; do not paraphrase).
- Order MUST match the template order (index 0 → first, etc.).
- Use each slot’s `general_pattern` as the content brief for that section only.

### C) Time alignment (strict)
- Each section has **exactly one** `duration` object with JSON **numbers** (float seconds), not strings, not `HH:MM:SS`.
- `duration.start` / `duration.end` must come from real transcript timestamps (`duration.start < duration.end`).
- Prefer the min `start` / max `end` among turns that materially support that section’s content.
- If almost nothing in the transcript supports a slot, use the full transcript span (first turn `start` → last turn `end`) rather than inventing times.
- Durations **may overlap** across sections; chronological ordering of `duration.start` is **not** required in template mode.

### D) Section descriptions — Markdown (strict)
- `section_description` MUST be real **Markdown**: paragraphs and/or bullets; **bold** for key terms; short quotes when reflecting transcript wording.
- **Forbidden:** Markdown headings. Do **not** use lines starting with `#`, `##`, `###`, etc. Do not fake headings with ALL-CAPS title lines.
- Every `section_description` must be non-empty.

### E) Whole-meeting metadata (strict)
- **`meeting_title`**: Concise, specific title for the **entire** conversation. You MAY use `template_title` as a hint when it fits the transcript; still ground the title in what was actually discussed. No timestamps. Avoid bare generics (“Meeting”, “Discussion”).
- **`meeting_short_description`**: **Exactly one sentence**, plain text only (no line breaks, bullets, or markdown), **≤ 150 characters**, faithful to the whole meeting.
- **`themes`**: **3–8** distinct short keywords/phrases supported by the transcript (not vague fillers).

## Output format (strict machine-readable JSON)
Return **ONLY** valid JSON (no markdown fences, no commentary). Top-level object:

{
  "meeting_title": string,
  "meeting_short_description": string,
  "themes": [ string, ... ],
  "sections": [
    {
      "section_title": string,
      "section_description": string,
      "duration": { "start": number, "end": number }
    }
  ]
}

Rules:
- `themes` MUST contain **at least 3** and **at most 8** items.
- `sections` MUST align 1:1 with `template_sections` (count, order, exact titles).
- Use JSON numbers for `start` and `end`.

## Self-check (apply before returning)
- Section count, order, and titles match the template exactly.
- Each description follows its `general_pattern` without inventing unsupported facts.
- No `#` heading lines in any `section_description`.
- `meeting_short_description` is one sentence and ≤ 150 characters.
- `themes` length is 3–8.
- Every duration uses real transcript timestamps with `start < end`."""


_NOTES_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "meeting_title": {"type": "string"},
        "meeting_short_description": {"type": "string", "maxLength": 150},
        "themes": {
            "type": "array",
            "items": {"type": "string", "maxLength": 120},
            "minItems": 3,
            "maxItems": 8,
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_title": {"type": "string"},
                    "section_description": {"type": "string"},
                    "duration": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                        },
                        "required": ["start", "end"],
                        "additionalProperties": False,
                    },
                },
                "required": ["section_title", "section_description", "duration"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "meeting_title",
        "meeting_short_description",
        "themes",
        "sections",
    ],
    "additionalProperties": False,
}

_HEADING_LINE = re.compile(r"^\s*#+\s")


def _turns_json(segments: list[TranscriptionSegment]) -> str:
    rows: list[dict[str, str | float]] = []
    for s in segments:
        rows.append(
            {
                "speaker": (s.speaker or "").strip() or "Unknown",
                "text": (s.text or "").strip(),
                "start": float(s.start),
                "end": float(s.end),
            }
        )
    while rows:
        raw = json.dumps(rows, ensure_ascii=False)
        if len(raw) <= _MAX_TRANSCRIPT_JSON_CHARS:
            if len(rows) < len(segments):
                logger.warning(
                    "Notes: truncated transcript to %s turns to fit char budget",
                    len(rows),
                )
            return raw
        rows.pop()
    return "[]"


def _template_json(template: NotesTemplate) -> str:
    return json.dumps(
        {
            "template_title": template.template_title,
            "template_sections": [
                {
                    "section_title": sec.section_title,
                    "general_pattern": sec.general_pattern,
                }
                for sec in template.template_sections
            ],
        },
        ensure_ascii=False,
    )


def _has_markdown_heading_line(text: str) -> bool:
    return any(_HEADING_LINE.match(line) for line in text.splitlines())


def _normalize_meeting_short_description(raw: str) -> str:
    """Single line, collapsed whitespace; hard cap at 150 characters."""
    s = " ".join((raw or "").split())
    return s[:_MEETING_SHORT_DESC_MAX_CHARS]


def _normalize_themes(raw: object) -> list[str]:
    if not isinstance(raw, list):
        raise RuntimeError('"themes" must be a JSON array of strings')
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        if not isinstance(x, str):
            continue
        t = x.strip()[:_THEME_ITEM_MAX_CHARS]
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    if len(out) < 3:
        raise RuntimeError(
            f'"themes" must have at least 3 non-empty distinct items after normalization, got {len(out)}'
        )
    return out[:8]


def _validate_sections_against_transcript(
    segments: list[TranscriptionSegment],
    sections: list[NoteSection],
    *,
    require_chronological: bool = True,
) -> None:
    if not segments:
        if sections:
            raise RuntimeError("Empty transcript must produce no sections")
        return
    t_min = min(s.start for s in segments)
    t_max = max(s.end for s in segments)
    if not sections:
        raise RuntimeError("Non-empty transcript requires at least one section")
    prev_start: float | None = None
    for i, sec in enumerate(sections):
        d = sec.duration
        if d.start < t_min - 0.01 or d.end > t_max + 0.01:
            raise RuntimeError(
                f"Section {i} duration [{d.start}, {d.end}] is outside transcript bounds [{t_min}, {t_max}]"
            )
        if _has_markdown_heading_line(sec.section_description):
            raise RuntimeError(
                f"Section {i} description must not contain Markdown heading lines (#)"
            )
        if require_chronological:
            if prev_start is not None and d.start + 1e-6 < prev_start:
                raise RuntimeError("Sections must be ordered chronologically by duration.start")
            prev_start = d.start


def _validate_sections_match_template(
    template: NotesTemplate,
    sections: list[NoteSection],
) -> None:
    expected = [s.section_title.strip() for s in template.template_sections]
    got = [s.section_title.strip() for s in sections]
    if len(got) != len(expected):
        raise RuntimeError(
            f"Template requires {len(expected)} sections, model returned {len(got)}"
        )
    for i, (exp, actual) in enumerate(zip(expected, got)):
        if actual != exp:
            raise RuntimeError(
                f"sections[{i}].section_title must be {exp!r}, got {actual!r}"
            )


def _parse_sections_list(raw_sections: object) -> list[NoteSection]:
    if not isinstance(raw_sections, list):
        raise RuntimeError('Notes model output must contain key "sections" with a JSON array')
    out: list[NoteSection] = []
    for i, item in enumerate(raw_sections):
        if not isinstance(item, dict):
            raise RuntimeError(f"sections[{i}] must be an object")
        try:
            dur = item.get("duration") or {}
            out.append(
                NoteSection(
                    section_title=str(item.get("section_title", "")).strip(),
                    section_description=str(item.get("section_description", "")).strip(),
                    duration=NoteDuration(
                        start=float(dur["start"]),
                        end=float(dur["end"]),
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise RuntimeError(f"Invalid section at index {i}: {exc}") from exc
    return out


def _parse_model_output(content: str) -> tuple[str, str, list[str], list[NoteSection]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from notes model: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Notes model output must be a JSON object")

    meeting_title = str(data.get("meeting_title", "")).strip()
    if not meeting_title:
        raise RuntimeError("meeting_title must be non-empty")

    desc_raw = str(data.get("meeting_short_description", "")).strip()
    meeting_short_description = _normalize_meeting_short_description(desc_raw)
    if not meeting_short_description:
        raise RuntimeError("meeting_short_description must be non-empty")

    themes = _normalize_themes(data.get("themes"))
    sections = _parse_sections_list(data.get("sections"))
    return meeting_title, meeting_short_description, themes, sections


def _build_user_message(
    recording_id: str,
    transcript_json: str,
    template: NotesTemplate | None,
) -> str:
    if template is None:
        return f"""recording_id: {recording_id}

TRANSCRIPT_JSON (chronological array of objects with speaker, text, start, end in seconds):
{transcript_json}

Produce JSON exactly as specified in the system message. Do not echo the full transcript back."""

    return f"""recording_id: {recording_id}

TRANSCRIPT_JSON (chronological array of objects with speaker, text, start, end in seconds):
{transcript_json}

TEMPLATE (fill sections to match this structure exactly — titles and order are authoritative):
{_template_json(template)}

Produce JSON exactly as specified in the system message. Do not echo the full transcript back."""


def generate_notes_sync(
    recording_id: str,
    segments: list[TranscriptionSegment],
    template: NotesTemplate | None = None,
) -> GenerateNotesResponse:
    if not settings.openai_api_key.strip():
        raise ValueError("OPENAI_API_KEY is not set")

    model = (settings.openai_notes_model or "gpt-4.1").strip()
    transcript_json = _turns_json(segments)
    system_prompt = _TEMPLATE_SYSTEM_PROMPT if template is not None else _SYSTEM_PROMPT
    user_message = _build_user_message(recording_id, transcript_json, template)

    client = OpenAI(api_key=settings.openai_api_key.strip())
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.25,
            max_tokens=16384,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "meeting_notes",
                    "strict": True,
                    "schema": _NOTES_JSON_SCHEMA,
                },
            },
        )
    except APIError as exc:
        raise RuntimeError(str(exc)) from exc

    choice = resp.choices[0].message
    raw = (choice.content or "").strip()
    if not raw:
        raise RuntimeError("Notes model returned empty content")

    meeting_title, meeting_short_description, themes, sections = _parse_model_output(raw)
    if template is not None:
        _validate_sections_match_template(template, sections)
    _validate_sections_against_transcript(
        segments,
        sections,
        require_chronological=template is None,
    )
    return GenerateNotesResponse(
        recording_id=recording_id,
        meeting_title=meeting_title,
        meeting_short_description=meeting_short_description,
        themes=themes,
        sections=sections,
    )

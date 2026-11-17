"""Generate a rich transcript summary via OpenAI (gpt-4.1 by default).

The ``summary`` string is returned in JSON for the **frontend**, which renders it with
**react-markdown**. Use real newlines and standard ``- `` unordered lists (CommonMark-friendly);
avoid raw HTML in model output so the renderer stays predictable.
"""

from __future__ import annotations

import logging

from openai import APIError, OpenAI

from src.config import settings
from src.schemas.transcription.transcribe import TranscriptionSegment

logger = logging.getLogger(__name__)

# Rough guardrail; model context is large but we avoid pathological payloads.
_MAX_TRANSCRIPT_CHARS = 240_000

_SYSTEM_PROMPT = """You are a senior editorial assistant. You read a diarized transcript (lines are in chronological order; each line is who spoke and what they said).

## Your task
Write a **rich, readable summary** of the conversation as **Markdown**, using **unordered bullet points only**. The summary must be **faithful to the transcript only** — do not invent facts, speakers, commitments, or events that are not clearly supported by the text.

## What the summary should cover (each point is one `-` list item; logical order top-to-bottom)
- What the conversation is about and how it develops from beginning to end.
- Who is involved and their roles or stance when that is clear from dialogue (use names or speaker labels as given).
- Main themes, disagreements, conclusions, and any explicit decisions, agreements, or action items.
- Open questions or follow-ups **only** if they clearly appear in the transcript.

## Output shape (strict) — Markdown, unordered bullets ONLY
- Your entire answer must be **only** an **unordered Markdown list**: every substantive line **must** start with **`- `** (hyphen + space). Use a **flat** list (no nested sub-bullets, no indented continuation blocks).
- **First character of your entire response must be `-`** (the start of the first bullet). Do **not** prefix the answer with quotation marks, braces, or commentary. Do **not** wrap the full summary in outer `"` `'` or ` ``` ` fences.
- **Line breaks (required):** Each bullet is **exactly one line** starting with `- `. Separate bullets with **real line breaks** (press Enter between bullets in the model output so the string contains newline characters). Do **not** emit the two-character sequence backslash + `n` as text; do **not** output one giant paragraph with no breaks. You **may** insert **a single blank line** between **groups** of related bullets for readability (still only `-` lines; no prose between groups).
- You may use **inline Markdown** inside a bullet where helpful: **bold** for key terms, *italic* sparingly. Do **not** wrap the whole answer in a code fence. Do **not** use HTML line tags (`<br>`) or other raw HTML; the client renders this with **react-markdown** (standard CommonMark-style lists and inline emphasis work best).
- **Forbidden:** paragraph text **without** a leading `- `; **numbered** lists (`1.`); **any** heading lines (`#`, `##`, …); tables; horizontal rules (`---`); blockquotes as the main format.
- **No timestamps or time references of any kind** in list text: no clock times (`1:23`), no “at minute …”, no “first half of the recording”, no second-based ranges tied to the media timeline. You may describe sequence in words (“Later, they turn to …”) without timecodes.
- **No** preamble line like “Here is a summary:” unless it too is a normal `- ` bullet (prefer to start directly with content bullets). Keep tone neutral and professional.

## Language
Use the dominant language of the transcript; if it is mixed, follow the mix naturally.

## Constraints
- Do **not** output JSON or XML. Output **only** the Markdown list string.
- Do **not** paste or replay the transcript; synthesize into bullets.
- If the transcript is fragmented or ambiguous, use careful wording inside bullets (“It appears …”, “One speaker suggests …”) instead of inventing detail."""


def _normalize_summary_output(text: str) -> str:
    """Strip accidental outer quotes; turn literal \\n in a single-line blob into real newlines."""
    t = (text or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
        t = t[1:-1].strip()
    if "\n" not in t and "\\n" in t:
        t = t.replace("\\n", "\n")
    return t.strip()


def _build_transcript_block(segments: list[TranscriptionSegment]) -> tuple[str, bool]:
    lines: list[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        sp = (seg.speaker or "").strip() or "Unknown"
        lines.append(f"{sp}: {text}")
    raw = "\n".join(lines)
    truncated = len(raw) > _MAX_TRANSCRIPT_CHARS
    if truncated:
        raw = raw[:_MAX_TRANSCRIPT_CHARS].rstrip() + "\n\n[… transcript truncated for processing length …]"
    return raw, truncated


def generate_summary_sync(
    recording_id: str,
    segments: list[TranscriptionSegment],
) -> str:
    if not settings.openai_api_key.strip():
        raise ValueError("OPENAI_API_KEY is not set")

    model = (settings.openai_summary_model or "gpt-4.1").strip()
    transcript_block, was_truncated = _build_transcript_block(segments)
    if was_truncated:
        logger.warning(
            "Summary input truncated to %s chars for recording_id=%s",
            _MAX_TRANSCRIPT_CHARS,
            recording_id,
        )

    user_message = f"""Recording id (for context only; do not repeat in the summary unless useful): `{recording_id}`

Transcript (chronological; each line is speaker then what they said — do not copy times into your answer; the transcript has no timestamps below):

{transcript_block}"""

    client = OpenAI(api_key=settings.openai_api_key.strip())
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.35,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except APIError as exc:
        raise RuntimeError(str(exc)) from exc

    choice = resp.choices[0].message
    text = _normalize_summary_output(choice.content or "")
    if not text:
        raise RuntimeError("Summary model returned empty content")
    return text

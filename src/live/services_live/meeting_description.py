"""Enrich a meeting title/description into clean prose via OpenAI (gpt-4.1)."""

from __future__ import annotations

import logging
import re

from openai import APIError, OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

_MODEL = "gpt-4.1"
_MAX_INPUT_CHARS = 8_000

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|"
    r"(?:meet\.google\.com|zoom\.us|teams\.microsoft\.com|"
    r"calendar\.google\.com|outlook\.(?:office|live)\.com)/\S+",
    re.IGNORECASE,
)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


_SYSTEM_PROMPT = """You write concise meeting descriptions for a call-prep UI.

Rules (strict):
- Use ONLY facts supported by the meeting title and/or raw description provided. Do not invent attendees, topics, goals, companies, or outcomes that are not clearly implied by the input.
- Remove all URLs, links, dial-in codes, calendar boilerplate, and participant lists from your output — even if they appear in the input.
- Write plain prose paragraphs only (no bullets, no markdown, no headings).
- If context is sparse (title only, or title plus very little meaningful detail): output exactly ONE short paragraph (2–4 sentences).
- If title and description together provide rich, specific context: output at most TWO paragraphs.
- Keep a neutral, professional tone. Do not mention that you cleaned or enriched the text."""


def _normalize_whitespace(text: str) -> str:
    t = _URL_RE.sub(" ", text or "")
    t = _MULTI_SPACE_RE.sub(" ", t)
    t = _MULTI_NEWLINE_RE.sub("\n\n", t)
    return t.strip()


def _build_user_message(meeting_title: str, meeting_description: str) -> str:
    title = (meeting_title or "").strip()
    desc = _normalize_whitespace(meeting_description or "")
    parts = [f"Meeting title:\n{title or '(empty)'}"]
    if desc:
        parts.append(f"Raw meeting description (may contain links/noise):\n{desc}")
    else:
        parts.append("Raw meeting description: (empty)")
    return "\n\n".join(parts)


def _normalize_output(text: str) -> str:
    t = (text or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
        t = t[1:-1].strip()
    t = _URL_RE.sub("", t)
    t = _MULTI_SPACE_RE.sub(" ", t)
    t = _MULTI_NEWLINE_RE.sub("\n\n", t)
    return t.strip()


def enrich_meeting_description_sync(
    meeting_title: str,
    meeting_description: str = "",
) -> str:
    if not settings.openai_api_key.strip():
        raise ValueError("OPENAI_API_KEY is not set")

    title = (meeting_title or "").strip()
    if not title and not (meeting_description or "").strip():
        raise ValueError("At least meeting_title or meeting_description must be non-empty")

    user_message = _build_user_message(title, meeting_description)
    if len(user_message) > _MAX_INPUT_CHARS:
        user_message = user_message[:_MAX_INPUT_CHARS].rstrip() + "\n\n[… truncated …]"
        logger.warning("Meeting description input truncated to %s chars", _MAX_INPUT_CHARS)

    client = OpenAI(api_key=settings.openai_api_key.strip())
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0.3,
            max_tokens=512,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except APIError as exc:
        raise RuntimeError(str(exc)) from exc

    text = _normalize_output(resp.choices[0].message.content or "")
    if not text:
        raise RuntimeError("Meeting description model returned empty content")
    return text

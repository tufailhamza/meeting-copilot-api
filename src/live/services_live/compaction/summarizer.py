"""SummarizerClient: fold transcript turns into a rolling meeting summary."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from openai import APIError, AsyncOpenAI

from src.config import settings as app_settings
from src.live.config_live import live_settings
from src.live.services_live.projection import TranscriptTurn

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You maintain a rolling summary of a live business meeting transcript.

You receive:
- An optional existing summary of earlier conversation
- New transcript turns to fold in

Write a concise executive summary that preserves:
- Decisions, commitments, timelines, and numbers
- Named people, companies, products, and roles
- Open questions, risks, and objections
- Topics already covered (so the coach does not repeat itself)

Rules:
- Do NOT invent facts not present in the turns.
- Merge with the existing summary; do not repeat verbatim.
- Keep the result under 1200 words.
- Use short paragraphs or bullet points — optimized for a coach reading quickly.
"""


class SummarizerClient(ABC):
    @abstractmethod
    async def summarize(
        self,
        existing_summary: str | None,
        turns: list[TranscriptTurn],
    ) -> str: ...


class OpenAISummarizer(SummarizerClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_chars: int,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is not set")
        self._model = model.strip()
        self._max_chars = max_chars
        self._client = client or AsyncOpenAI(api_key=api_key.strip())

    async def summarize(
        self,
        existing_summary: str | None,
        turns: list[TranscriptTurn],
    ) -> str:
        if not turns:
            return existing_summary or ""

        lines = [f"{t.speaker}: {t.text}" for t in turns]
        user = "EXISTING_SUMMARY:\n"
        user += (existing_summary or "(none)") + "\n\n"
        user += "NEW_TURNS:\n" + "\n".join(lines)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=900,
            )
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned empty summary")
        return text[: self._max_chars]


class FakeSummarizer(SummarizerClient):
    """Deterministic summarizer for tests — bounded output regardless of input size."""

    def __init__(self, *, max_chars: int = 4000) -> None:
        self._max_chars = max_chars
        self.calls = 0

    async def summarize(
        self,
        existing_summary: str | None,
        turns: list[TranscriptTurn],
    ) -> str:
        self.calls += 1
        speakers = sorted({t.speaker for t in turns})
        chunk = f"Summary of {len(turns)} turns ({', '.join(speakers)})."
        if existing_summary:
            merged = f"{existing_summary}\n{chunk}"
        else:
            merged = chunk
        return merged[: self._max_chars]


def build_summarizer_client() -> SummarizerClient | None:
    if not app_settings.openai_api_key.strip():
        return None
    return OpenAISummarizer(
        api_key=app_settings.openai_api_key,
        model=live_settings.openai_summary_model,
        max_chars=live_settings.summary_max_chars,
    )

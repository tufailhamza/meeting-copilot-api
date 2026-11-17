"""Rolling transcript memory helpers.

The agent does not read the entire meeting transcript on every evaluation — it reads a bounded
recent window plus an optional rolling summary of earlier turns. The full transcript remains
in the event log / projection; ``prompt_transcript`` selects what enters LLM prompts.
"""

from __future__ import annotations

from src.live.services_live.projection import MeetingSession, TranscriptTurn


def recent_transcript(transcript: list[TranscriptTurn], window: int) -> list[TranscriptTurn]:
    """Return the last ``window`` transcript turns (or the full list if shorter)."""
    if window <= 0:
        return []
    if len(transcript) <= window:
        return list(transcript)
    return transcript[-window:]


def prompt_transcript(session: MeetingSession, window: int) -> tuple[str | None, list[TranscriptTurn]]:
    """Return (earlier_summary, recent_turns) for agent/chat prompts."""
    if session.compacted_turn_count > 0:
        live = session.transcript[session.compacted_turn_count :]
    else:
        live = session.transcript
    recent = recent_transcript(live, window)
    return session.transcript_summary, recent

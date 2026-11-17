"""Detect which diarized speaker is the host (OpenAI), then relabel that speaker to ``Host``."""

import asyncio
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import APIError, OpenAI

from src.schemas.transcription.transcribe import TranscriptionSegment

_ROOT = Path(__file__).resolve().parents[3]
_MODEL = "gpt-4.1"

logger = logging.getLogger(__name__)

# Host prompt only: cap JSON size; use head + tail + per-speaker backfill so late-only speakers stay visible.
_HOST_PROMPT_MAX_JSON_CHARS = 25_000
_HEAD_JSON_FRAC = 0.38
_TAIL_JSON_FRAC = 0.38
_BACKFILL_SPEAKER_TEXT_MAX = 280
_SHRINK_LONGEST_TEXT_FLOOR = 80

_SYSTEM_PROMPT = """You decide which speaker label is the **host** of the conversation (e.g. moderator, main interviewer, or the person clearly leading / owning the session).

INPUT FORMAT:
- A JSON **array** in **time order**. Each element is a single JSON object with **exactly one key**: the speaker label (e.g. "A", "B"). The value is what that speaker said in that turn (already transliterated to Latin script where applicable).
- Long recordings may be **excerpted** (beginning, end, and short samples so every speaker appears at least once). Infer the host only from what is shown; every label you see was a real diarized speaker in the session.

YOUR TASK:
- Pick **exactly one** speaker label that is the host. It **must** be one of the object keys that actually appears in the array (copy the key string exactly, same casing).

OUTPUT — JSON ONLY:
- A single JSON object: {"host_speaker": "<KEY>"}
- Example: {"host_speaker": "A"}
- Do not add other keys. Do not explain."""

_USER_TEMPLATE = """Ordered turns (each object has one speaker key and their line):

{payload_json}
"""


def _speaker_key(block: dict[str, str]) -> str:
    return next(iter(block.keys()))


def _payload_json_chars(blocks: list[dict[str, str]]) -> int:
    if not blocks:
        return 0
    return len(json.dumps(blocks, ensure_ascii=False))


def _merge_consecutive_speaker_blocks(segments: list[TranscriptionSegment]) -> list[dict[str, str]]:
    """Merge consecutive same speaker with a single space between pieces."""
    if not segments:
        return []
    out: list[dict[str, str]] = []
    cur_speaker = segments[0].speaker
    parts: list[str] = []
    for seg in segments:
        piece = (seg.text or "").strip()
        if seg.speaker == cur_speaker:
            if piece:
                parts.append(piece)
        else:
            out.append({cur_speaker: " ".join(parts)})
            cur_speaker = seg.speaker
            parts = [piece] if piece else []
    out.append({cur_speaker: " ".join(parts)})
    return out


def _budget_merged_host_payload(merged: list[dict[str, str]]) -> list[dict[str, str]]:
    """Head + tail windows under separate char budgets, then backfill any speaker missing from both."""
    if not merged:
        return []
    max_c = _HOST_PROMPT_MAX_JSON_CHARS
    if _payload_json_chars(merged) <= max_c:
        return [dict(b) for b in merged]

    head_budget = int(max_c * _HEAD_JSON_FRAC)
    tail_budget = int(max_c * _TAIL_JSON_FRAC)
    n = len(merged)

    head_indices: list[int] = []
    head_blocks: list[dict[str, str]] = []
    for i in range(n):
        trial = head_blocks + [merged[i]]
        if _payload_json_chars(trial) > head_budget:
            break
        head_blocks.append(merged[i])
        head_indices.append(i)
    head_set = set(head_indices)

    tail_indices: list[int] = []
    tail_blocks: list[dict[str, str]] = []
    for j in range(n - 1, -1, -1):
        if j in head_set:
            break
        trial_idx = sorted(tail_indices + [j])
        trial = [merged[k] for k in trial_idx]
        if _payload_json_chars(trial) > tail_budget:
            break
        tail_indices.append(j)
    tail_set = set(tail_indices)

    shown_speakers: set[str] = set()
    for i in head_indices:
        shown_speakers.add(_speaker_key(merged[i]))
    for j in tail_indices:
        shown_speakers.add(_speaker_key(merged[j]))

    all_speakers = {_speaker_key(m) for m in merged}
    missing = all_speakers - shown_speakers

    index_to_block: dict[int, dict[str, str]] = {}
    for i in head_indices:
        index_to_block[i] = dict(merged[i])
    for j in tail_indices:
        index_to_block[j] = dict(merged[j])

    for s in missing:
        for i, m in enumerate(merged):
            if _speaker_key(m) != s:
                continue
            if i in index_to_block:
                break
            k, v = next(iter(m.items()))
            excerpt = v[:_BACKFILL_SPEAKER_TEXT_MAX] if v else v
            index_to_block[i] = {k: excerpt}
            break

    ordered = sorted(index_to_block)
    out = [index_to_block[i] for i in ordered]

    out = [dict(b) for b in out]
    while out and _payload_json_chars(out) > max_c:
        lengths = [len(next(iter(b.values()))) for b in out]
        if not lengths or max(lengths) <= _SHRINK_LONGEST_TEXT_FLOOR:
            break
        li = max(range(len(out)), key=lambda i: lengths[i])
        k, v = next(iter(out[li].items()))
        new_len = max(_SHRINK_LONGEST_TEXT_FLOOR, int(len(v) * 0.82))
        out[li] = {k: v[:new_len]}
    return out


def build_host_detection_payload(segments: list[TranscriptionSegment]) -> list[dict[str, str]]:
    """Build JSON array content for the host model only (does not mutate segments).

    Consecutive rows with the same speaker are merged with a single space. If the merged
    JSON would exceed a budget, only a head window, tail window, and minimal first-turn
    excerpts for speakers absent from head/tail are kept—callback segments stay unchanged.
    """
    merged = _merge_consecutive_speaker_blocks(segments)
    return _budget_merged_host_payload(merged)


def _parse_host_json(content: str) -> str:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from host model: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Host model output must be a JSON object")
    host = data.get("host_speaker")
    if host is None or not isinstance(host, str) or not host.strip():
        raise RuntimeError('Host model output must include non-empty string "host_speaker"')
    return host.strip()


def detect_host_speaker_sync(
    segments: list[TranscriptionSegment],
    recording_id: str | None = None,
) -> str:
    log_prefix = f"[transcribe][{recording_id}]" if recording_id else "[host_detection]"
    if not segments:
        return ""

    logger.info("%s host detection OpenAI call starting", log_prefix)

    load_dotenv(_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    allowed_labels = {seg.speaker for seg in segments}
    payload = build_host_detection_payload(segments)
    user_msg = _USER_TEMPLATE.format(
        payload_json=json.dumps(payload, ensure_ascii=False),
    )

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
    except APIError as exc:
        raise RuntimeError(str(exc)) from exc

    raw = resp.choices[0].message.content or ""
    host = _parse_host_json(raw)
    if host not in allowed_labels:
        raise RuntimeError(
            f'host_speaker {host!r} is not among labels in this transcript: {sorted(allowed_labels)}'
        )
    logger.info("%s host detection resolved host_speaker=%r", log_prefix, host)
    return host


def relabel_host_speaker_sync(
    segments: list[TranscriptionSegment],
    host_speaker: str,
) -> list[TranscriptionSegment]:
    if not host_speaker or not segments:
        return list(segments)
    out: list[TranscriptionSegment] = []
    for seg in segments:
        speaker = "Host" if seg.speaker == host_speaker else seg.speaker
        out.append(
            TranscriptionSegment(
                speaker=speaker,
                text=seg.text,
                start=seg.start,
                end=seg.end,
            )
        )
    return out


async def detect_and_relabel_host(
    segments: list[TranscriptionSegment],
    *,
    recording_id: str | None = None,
) -> list[TranscriptionSegment]:
    if not segments:
        return segments
    host_key = await asyncio.to_thread(detect_host_speaker_sync, segments, recording_id)
    if not host_key:
        return segments
    return await asyncio.to_thread(relabel_host_speaker_sync, segments, host_key)

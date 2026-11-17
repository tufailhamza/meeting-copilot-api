"""Chunked Hindi→Latin transliteration via OpenAI (not translation)."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIError, OpenAI

from src.schemas.transcription.transcribe import TranscriptionSegment

_ROOT = Path(__file__).resolve().parents[3]
_CHUNK_SIZE = 15
_MODEL = "gpt-4.1"

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a strict transliteration engine.

HARD RULE — LATIN / ASCII ONLY OUTPUT (NON-NEGOTIABLE):
- In **every** output `"transcript"` string, you must **not** output **any** character from a **non-Latin script**. Forbidden examples include **any** Devanagari, **any** Arabic / Perso-Arabic / Urdu Nastaliq, **any** Cyrillic, Hebrew, Thai, CJK, Tamil, Bengali, etc.—**zero** such letters or syllabic characters anywhere, not even one glyph mixed inside English.
- **Allowed** in `"transcript"` only: (1) **Latin letters** `A–Z` and `a–z` for all alphabetic writing (Roman Hindi / Roman Urdu / English); (2) **ASCII digits** `0–9`; (3) **ASCII whitespace** (space, newline if needed); (4) **ASCII punctuation** commonly used in speech transcripts: `.,?!'"()-:;/` and `%` if needed. **Do not** use accented Latin (é, ñ, ü) or curly/smart quotes—use plain ASCII equivalents (e, n, u, straight quotes).
- If the input contains a non-Latin character, you **must** rewrite it into **allowed ASCII/Latin** (transliterate or substitute spelling). **Never** copy through, omit, or half-convert a Devanagari or Arabic letter. If unsure, pick the closest Roman spelling using **only** `A–Za–z`; never leave a foreign-script glyph in the output.
- Before you finish: **mentally scan** each output `"transcript"`—if any character is outside the allowed sets above, **fix it** until compliant.

DEFINITION — TRANSLITERATION (what you MUST do):
- Rewrite text from non-Latin scripts (especially Devanagari for Hindi, and Perso-Arabic script for Urdu when present) into **Roman form using only** `A–Za–z` as described in the HARD RULE.
- Preserve **pronunciation and wording**: same things are still being said—only the **script** changes. Do **not** translate into different English words or paraphrase.
- Use **natural, conventional Roman Hindi** or **Roman Urdu** spellings (as in everyday South Asian typing), not academic-only forms unless the source clearly warrants them:
  - **Hindi (Devanagari → Roman)**: reflect aspirates and clusters clearly (e.g. kh, gh, ch, th, dh, ph, bh, sh, ri patterns where heard); use vowel length in a sensible way (e.g. aa, ee, oo where people usually write them); keep common particles and verb endings recognizable (hai/hain, hoon, tha, raha, kar, etc.) in widely used spellings.
  - **Urdu (Arabic script → Roman Urdu)**: use spellings people actually type (e.g. kya, kyun, main/mein, tum/aap, nahi/nahin, achha/theek, zindagi—pick common variants and stay consistent within the same line).
- **English-only or already-Latin lines**: if a "transcript" field is only English (or already fully in Latin letters / Roman code-mix), **copy it verbatim** for that turn—**do not** change wording, add words, remove words, or "fix" grammar. You may only normalize obvious duplicate spaces.
- **Using other turns as context (transliteration only, when needed)**: you still output **one row per input row** with the same "turn" and "speaker". You must **not** merge lines or move wording across turns. For **non-Latin** text only, if a word or phrase is **genuinely ambiguous** in isolation (script choice, homograph, boundary, register), you **may** use **other "transcript" fields in the same INPUT JSON array** (same chunk) to pick a consistent Roman spelling—**only** to disambiguate transliteration, **not** to change facts, add content from elsewhere, or translate.
- Do **not** add explanations, notes, or markdown. Do **not** translate meaning into different English words.

OUTPUT — JSON ONLY:
- Respond with a single JSON **object** (not an array at the top level) with exactly one key: `"segments"`.
- `"segments"` must be an **array** of objects. Each object must have exactly these keys: `"turn"` (integer), `"speaker"` (string), `"transcript"` (string).
- You MUST return **one object per input row**, same count, same `turn` values, same `speaker` values as provided. Only change `"transcript"` to the transliterated text. Each `"transcript"` must **fully** satisfy the **HARD RULE — LATIN / ASCII ONLY OUTPUT** (no exceptions).
- `turn` is for alignment only: copy the integer you were given for each row; do not invent or skip turns."""

_USER_CHUNK_TEMPLATE = """Below is a JSON array of dialogue turns. Transliterate non-Latin parts of each "transcript" to Latin letters as specified in the system message.

INPUT (do not echo it back except as the structure requires — output ONLY the JSON object with "segments"):
{input_json}
"""


def _build_chunk_payload(
    global_turn_start: int,
    segments: list[TranscriptionSegment],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        out.append(
            {
                "turn": global_turn_start + i,
                "speaker": seg.speaker,
                "transcript": seg.text,
            }
        )
    return out


def _parse_openai_json(content: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Model output must be a JSON object")
    segs = data.get("segments")
    if not isinstance(segs, list):
        raise RuntimeError('Model output must contain key "segments" with a JSON array')
    return segs


def _validate_and_extract(
    chunk_payload: list[dict[str, Any]],
    parsed_segments: list[dict[str, Any]],
) -> dict[int, str]:
    expected_turns = {int(x["turn"]) for x in chunk_payload}
    by_turn: dict[int, str] = {}
    for item in parsed_segments:
        if not isinstance(item, dict):
            continue
        turn = item.get("turn")
        tr = item.get("transcript")
        if turn is None or tr is None:
            continue
        by_turn[int(turn)] = str(tr).strip() if isinstance(tr, str) else str(tr)
    missing = expected_turns - set(by_turn.keys())
    if missing:
        raise RuntimeError(f"Model omitted turns: {sorted(missing)}")
    extra = set(by_turn.keys()) - expected_turns
    if extra:
        raise RuntimeError(f"Model returned unexpected turns: {sorted(extra)}")
    return by_turn


def _run_chunk(client: OpenAI, chunk_payload: list[dict[str, Any]]) -> dict[int, str]:
    user_msg = _USER_CHUNK_TEMPLATE.format(
        input_json=json.dumps(chunk_payload, ensure_ascii=False),
    )
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
    choice = resp.choices[0].message
    raw = choice.content or ""
    parsed = _parse_openai_json(raw)
    return _validate_and_extract(chunk_payload, parsed)


def transliterate_segments_sync(
    segments: list[TranscriptionSegment],
    recording_id: str | None = None,
) -> list[TranscriptionSegment]:
    log_prefix = f"[transcribe][{recording_id}]" if recording_id else "[transliterate]"
    if not segments:
        logger.info("%s transliteration skipped: 0 segments", log_prefix)
        return []

    load_dotenv(_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    total_chunks = (len(segments) + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    logger.info(
        "%s transliteration: %s segment(s) in %s chunk(s) of up to %s",
        log_prefix,
        len(segments),
        total_chunks,
        _CHUNK_SIZE,
    )

    client = OpenAI(api_key=api_key)
    turn_to_text: dict[int, str] = {}

    for chunk_index, offset in enumerate(range(0, len(segments), _CHUNK_SIZE), start=1):
        batch = segments[offset : offset + _CHUNK_SIZE]
        global_start = offset + 1
        chunk_payload = _build_chunk_payload(global_start, batch)
        logger.info(
            "%s transliteration chunk %s/%s starting (turns %s-%s)",
            log_prefix,
            chunk_index,
            total_chunks,
            global_start,
            global_start + len(batch) - 1,
        )
        t_chunk0 = time.perf_counter()
        try:
            merged = _run_chunk(client, chunk_payload)
        except Exception as exc:
            logger.exception(
                "%s transliteration chunk %s/%s failed: %s",
                log_prefix,
                chunk_index,
                total_chunks,
                exc,
            )
            raise
        turn_to_text.update(merged)
        logger.info(
            "%s transliteration chunk %s/%s complete in %.2f seconds",
            log_prefix,
            chunk_index,
            total_chunks,
            time.perf_counter() - t_chunk0,
        )

    out: list[TranscriptionSegment] = []
    for i, seg in enumerate(segments):
        turn = i + 1
        new_text = turn_to_text.get(turn, seg.text)
        out.append(
            TranscriptionSegment(
                speaker=seg.speaker,
                text=new_text,
                start=seg.start,
                end=seg.end,
            )
        )
    return out


async def transliterate_transcriptions(
    segments: list[TranscriptionSegment],
    *,
    recording_id: str | None = None,
) -> list[TranscriptionSegment]:
    return await asyncio.to_thread(transliterate_segments_sync, segments, recording_id)

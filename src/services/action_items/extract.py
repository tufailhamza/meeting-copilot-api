"""Extract per-participant action items from a meeting transcript via OpenAI (gpt-4.1)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import APIError, OpenAI

from src.config import settings
from src.schemas.action_items.extract_action_items import (
    ActionItem,
    ActionItemsByUser,
    ExtractActionItemsRequest,
    ExtractActionItemsResponse,
    TranscriptionSegmentOut,
    TranscriptTurn,
    UserMetadata,
)

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPT_JSON_CHARS = 240_000
_DESCRIPTION_MAX_WORDS = 15

_SYSTEM_PROMPT = """You are ACTION_ITEMS_EXTRACTION_AGENT — a production-grade meeting action-item extraction engine used in enterprise workflows. Your sole job is to read a chronological meeting transcript and a fixed participant roster, then output machine-readable action items assigned to the correct participants.

You are not a summarizer, note-taker, or general assistant. You extract actionable commitments only.

═══════════════════════════════════════════════════════════════════════════════
§1  INPUTS YOU RECEIVE
═══════════════════════════════════════════════════════════════════════════════

You will receive exactly two JSON payloads in the user message:

1) PARTICIPANTS_JSON — array of { participant_id (integer), speaker_name (string) }.
   This roster is authoritative. You may ONLY assign action items to participant_ids present here.

2) TRANSCRIPT_JSON — chronological array of { speaker_name, text }.
   The transcript is the ONLY source of truth.

═══════════════════════════════════════════════════════════════════════════════
§2  DEFINITIONS (STRICT)
═══════════════════════════════════════════════════════════════════════════════

ACTION ITEM — Include ONLY when the transcript supports at least one of:
  (A) EXPLICIT COMMITMENT — "I will…", "I'll…", "I'm going to…", "I need to…", "Let me…", "I am testing…"
  (B) DIRECT ASSIGNMENT — work assigned to a named participant ("Obaid, can you…", "you should…" with clear addressee)
  (C) GROUP OBLIGATION — "We all need to…", "Both of us should…", "Everyone must…"
  (D) DECISION REQUIRING FOLLOW-UP — concrete open decision with identifiable owner in context

NOT AN ACTION ITEM — Never extract for:
  - Greetings, small talk, acknowledgments ("Hello", "Nice", "Oh I see", "Yes", "I think so")
  - Past/completed work unless clearly ongoing ("I checked… and it works" with no future obligation)
  - Opinions, praise, observations with no future obligation
  - Vague discussion with no identifiable owner
  - Hypotheticals with no commitment
  - Filler fragments with no recoverable task
  - Inferring tasks for participants who never received or accepted work

DEFAULT WHEN UNCERTAIN: Do NOT create an action item.

═══════════════════════════════════════════════════════════════════════════════
§3  INTERNAL REASONING (DO NOT OUTPUT)
═══════════════════════════════════════════════════════════════════════════════

Silently: SCAN → FILTER → DEDUPE → ASSIGN → PRIORITIZE → WRITE → SELF-CHECK (§8).
Output ONLY final JSON.

═══════════════════════════════════════════════════════════════════════════════
§4  ASSIGNMENT RULES
═══════════════════════════════════════════════════════════════════════════════

4.1 SINGLE OWNER — "I will test OpenAI" by Obaid → Obaid only.
4.2 MULTIPLE PEOPLE — same item duplicated under each named participant_id.
4.3 ALL PARTICIPANTS — attach same item to EVERY participant_id in PARTICIPANTS_JSON.
4.4 NAME MATCHING — tolerate minor ASR errors ("Anik" ≈ "Aneeq Ahmed") only if unambiguous.
4.5 NO HALLUCINATED OWNERS — never assign work because someone merely attended or spoke.
4.6 Participants with zero obligations must NOT appear in assignments (server adds empty arrays).

═══════════════════════════════════════════════════════════════════════════════
§5  ACTION ITEM FIELDS
═══════════════════════════════════════════════════════════════════════════════

title — short, specific, actionable (verb-led when possible). No markdown.
description — ONE plain sentence, 10–15 words inclusive, HARD MAX 15 words. No markdown.
priority — exactly "low" | "medium" | "high"

HIGH: blocks progress, explicit urgency, critical decisions with delivery risk.
MEDIUM: clear follow-up, ongoing work, identifiable next step, no hard deadline stated.
LOW: optional, exploratory, soft suggestions. When ambiguous, choose lower priority.

═══════════════════════════════════════════════════════════════════════════════
§6  OUTPUT FORMAT (JSON ONLY)
═══════════════════════════════════════════════════════════════════════════════

{
  "assignments": [
    {
      "participant_id": <integer>,
      "action_items": [
        { "title": "<string>", "description": "<string>", "priority": "low"|"medium"|"high" }
      ]
    }
  ]
}

- Include assignments ONLY for participants with ≥1 action item.
- participant_id MUST exist in PARTICIPANTS_JSON.
- No extra keys, no markdown fences, no commentary.

═══════════════════════════════════════════════════════════════════════════════
§7  POSITIVE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

Obaid: "I'm currently testing OpenAI. I've added the API key in the hosted service."
→ participant_id Obaid: { title: "Complete OpenAI transcription testing", description: "Finish testing OpenAI transcription on hosted service with API key configured.", priority: "medium" }

Obaid: "Now we have to decide — closed caption per participant or OpenAI."
→ participant_id Obaid: { title: "Decide transcription approach", description: "Choose between per-participant closed captions or OpenAI meeting transcription.", priority: "high" }

Host: "We all need to review the security checklist before Friday's release."
→ same item on EVERY participant_id in roster.

Obaid: "Aneeq, please document the bot migration steps by next standup."
→ participant_id Aneeq only.

═══════════════════════════════════════════════════════════════════════════════
§8  NEGATIVE EXAMPLES (DO NOT EXTRACT)
═══════════════════════════════════════════════════════════════════════════════

Aneeq: "I checked transcriptions and everything works fine" → NO item (past observation).
"Hello." / "Nice." / "Yes." → NO items.
Obaid: "I'll handle the test." + Aneeq only says "Oh, I see" → NO item for Aneeq.
"Something like that could work maybe." → NO item.
18-word description → INVALID; rewrite to ≤15 words.

═══════════════════════════════════════════════════════════════════════════════
§9  SELF-CHECK (MANDATORY)
═══════════════════════════════════════════════════════════════════════════════

For each item: grounded in transcript? correct owner? not filler? description ≤15 words?
no invented tools/dates? priority not inflated? no duplicate same task per participant?

If any check fails, delete or fix. Prefer deletion over hallucination.
Optimize for PRECISION over RECALL."""

_ACTION_ITEMS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "participant_id": {"type": "integer"},
                    "action_items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string", "maxLength": 200},
                                "priority": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                },
                            },
                            "required": ["title", "description", "priority"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["participant_id", "action_items"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["assignments"],
    "additionalProperties": False,
}

_WORD_SPLIT = re.compile(r"\s+")


def _word_count(text: str) -> int:
    return len(_WORD_SPLIT.split(text.strip())) if text.strip() else 0


def _truncate_to_max_words(text: str, max_words: int) -> str:
    words = _WORD_SPLIT.split(text.strip())
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])


def build_transcriptions(
    turns: list[TranscriptTurn],
    start_timestamp_ms: int,
) -> list[TranscriptionSegmentOut]:
    """Map bot turns to speaker/text/start/end (seconds from meeting start_timestamp_ms)."""
    ordered = sorted(turns, key=lambda t: t.timestamp_ms)
    non_empty = [
        t
        for t in ordered
        if (t.transcription.transcript or "").strip()
    ]
    if not non_empty:
        return []

    out: list[TranscriptionSegmentOut] = []
    for turn in non_empty:
        text = (turn.transcription.transcript or "").strip()
        speaker = (turn.speaker_name or "").strip() or "Unknown"
        start = (turn.timestamp_ms - start_timestamp_ms) / 1000.0
        end = start + (max(turn.duration_ms, 0) / 1000.0)
        if end <= start:
            end = start + 0.001
        out.append(
            TranscriptionSegmentOut(
                speaker=speaker,
                text=text,
                start=start,
                end=end,
            )
        )
    return out


def _clean_transcript(
    turns: list[TranscriptTurn],
    start_timestamp_ms: int,
) -> list[dict[str, str]]:
    return [
        {"speaker_name": seg.speaker, "text": seg.text}
        for seg in build_transcriptions(turns, start_timestamp_ms)
    ]


def _participants_for_llm(users: list[UserMetadata]) -> list[dict[str, int | str]]:
    return [
        {"participant_id": u.user_id, "speaker_name": (u.name or "").strip()}
        for u in users
    ]


def _json_with_budget(rows: list[dict[str, Any]], max_chars: int) -> str:
    payload = list(rows)
    while payload:
        raw = json.dumps(payload, ensure_ascii=False)
        if len(raw) <= max_chars:
            if len(payload) < len(rows):
                logger.warning(
                    "Action items: truncated transcript to %s turns to fit char budget",
                    len(payload),
                )
            return raw
        payload.pop()
    return "[]"


def _normalize_action_item(raw: dict[str, Any]) -> ActionItem | None:
    title = str(raw.get("title", "")).strip()
    description = str(raw.get("description", "")).strip()
    priority = str(raw.get("priority", "")).strip().lower()
    if not title or not description or priority not in {"low", "medium", "high"}:
        return None
    if _word_count(description) > _DESCRIPTION_MAX_WORDS:
        description = _truncate_to_max_words(description, _DESCRIPTION_MAX_WORDS)
        logger.warning("Action item description truncated to %s words", _DESCRIPTION_MAX_WORDS)
    return ActionItem(title=title, description=description, priority=priority)


def _parse_model_assignments(
    content: str,
    valid_participant_ids: set[int],
) -> dict[int, list[ActionItem]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from action items model: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Action items model output must be a JSON object")

    assignments_raw = data.get("assignments")
    if not isinstance(assignments_raw, list):
        raise RuntimeError('Action items model output must contain key "assignments" with a JSON array')

    by_id: dict[int, list[ActionItem]] = {}
    for entry in assignments_raw:
        if not isinstance(entry, dict):
            continue
        pid_raw = entry.get("participant_id")
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            continue
        if pid not in valid_participant_ids:
            logger.warning("Dropping action items for unknown participant_id=%s", pid)
            continue
        items_raw = entry.get("action_items")
        if not isinstance(items_raw, list):
            continue
        parsed: list[ActionItem] = []
        seen: set[tuple[str, str, str]] = set()
        for item in items_raw:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_action_item(item)
            if normalized is None:
                continue
            key = (
                normalized.title.casefold(),
                normalized.description.casefold(),
                normalized.priority,
            )
            if key in seen:
                continue
            seen.add(key)
            parsed.append(normalized)
        if parsed:
            by_id[pid] = parsed
    return by_id


def _merge_with_roster(
    users: list[UserMetadata],
    by_id: dict[int, list[ActionItem]],
) -> list[ActionItemsByUser]:
    out: list[ActionItemsByUser] = []
    for user in users:
        out.append(
            ActionItemsByUser(
                participant_id=user.user_id,
                speaker_name=user.name.strip(),
                email=user.email.strip(),
                action_items=by_id.get(user.user_id, []),
            )
        )
    return out


def _wants_action_items(body: ExtractActionItemsRequest) -> bool:
    return bool(body.user_metadata)


def extract_action_items_sync(body: ExtractActionItemsRequest) -> ExtractActionItemsResponse:
    transcriptions = build_transcriptions(body.transcript, body.start_timestamp_ms)

    if not _wants_action_items(body):
        return ExtractActionItemsResponse(transcriptions=transcriptions)

    assert body.user_metadata is not None

    if not settings.openai_api_key.strip():
        raise ValueError("OPENAI_API_KEY is not set")

    model = (settings.openai_action_items_model or "gpt-4.1").strip()
    valid_ids = {u.user_id for u in body.user_metadata}

    transcript_rows = _clean_transcript(body.transcript, body.start_timestamp_ms)
    if not transcript_rows:
        return ExtractActionItemsResponse(
            transcriptions=transcriptions,
            action_items_by_user=_merge_with_roster(body.user_metadata, {}),
        )

    participants_json = json.dumps(_participants_for_llm(body.user_metadata), ensure_ascii=False)
    transcript_json = _json_with_budget(transcript_rows, _MAX_TRANSCRIPT_JSON_CHARS)

    user_message = f"""PARTICIPANTS_JSON:
{participants_json}

TRANSCRIPT_JSON:
{transcript_json}

Extract action items per the system instructions. Return JSON only."""

    client = OpenAI(api_key=settings.openai_api_key.strip())
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "action_items_assignments",
                    "strict": True,
                    "schema": _ACTION_ITEMS_JSON_SCHEMA,
                },
            },
        )
    except APIError as exc:
        raise RuntimeError(str(exc)) from exc

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("Action items model returned empty content")

    by_id = _parse_model_assignments(raw, valid_ids)
    return ExtractActionItemsResponse(
        transcriptions=transcriptions,
        action_items_by_user=_merge_with_roster(body.user_metadata, by_id),
    )

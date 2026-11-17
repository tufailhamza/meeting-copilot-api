from __future__ import annotations

"""Call-prep chat endpoint helpers: DB access + OpenAI + enterprise SOA calls."""

import hashlib
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, List, Literal

import asyncpg
import httpx
from openai import APIError, AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel, Field

from src.config import settings as app_settings
from src.live.config_live import enterprise_search_settings, live_db_settings, live_settings

logger = logging.getLogger(__name__)

_POOL: asyncpg.Pool | None = None

DISCOVERY_SECTION_ID = "sec_discovery_questions"
DISCOVERY_SECTION_KEY = "discovery_questions"
PEOPLE_SECTION_KEY = "people"
SEVERITIES = frozenset({"high", "medium", "low", "info"})
# Block shapes the chat agent may write. Deliberately a subset of what the renderer understands:
# ``person``, ``company`` and ``timeline`` belong to sections it is not allowed to touch anyway,
# and ``entity_cards``/``table`` need nested rows that bloat the structured-output schema for
# content nobody has asked the assistant to produce.
WRITABLE_BLOCK_TYPES = ("numbered_list", "flagged_list", "key_values", "chips", "prose", "links")
# Everything the payload may legitimately contain, including shapes only the load path writes.
KNOWN_BLOCK_TYPES = frozenset(WRITABLE_BLOCK_TYPES) | {"timeline", "person", "company"}
# What a section becomes when nothing semantic fits. Stated in the prompt too, so an unsure turn
# has somewhere to land rather than inventing a type.
DEFAULT_BLOCK_TYPE = "numbered_list"
# Sections whose shape is decided by their key, never by the model. Mirrors the sibling service's
# canonical set, restricted to keys that actually appear in our payloads.
CANONICAL_BLOCK_TYPES = {
    "discovery_questions": "numbered_list",
    "watch_points": "flagged_list",
    "key_discussion_points": "chips",
    "about": "prose",
    "deal_context": "key_values",
    "meeting_type": "key_values",
    "historical_context": "timeline",
    "people": "person",
    "company": "company",
}
# Strings that mean "the model had nothing" — dropped at every level rather than rendered.
JUNK_VALUES = frozenset({"unknown", "null", "n/a", "unclear"})
# The heading a canonical section always carries. The model retypes a section's name on every turn
# and its wording drifts; for these keys the stored name is not its to choose.
CANONICAL_SECTION_NAMES = {
    "discovery_questions": "Questions to Ask",
    "watch_points": "Watch Points",
    "key_discussion_points": "Key Discussion Points",
    "about": "About",
    "deal_context": "Deal Context",
    "meeting_type": "Meeting Type",
    "historical_context": "Historical Context",
    "people": "Who's on the Call",
    "company": "Company",
}
# Names that mean a canonical section, slugified. Identity cannot come from slugging the display
# name alone: "Questions to Ask" slugs to ``questions_to_ask``, which would create a second,
# unprotected questions section beside the real one the first time the real one is missing.
CANONICAL_NAME_ALIASES = {
    "questions_to_ask": "discovery_questions",
    "questions": "discovery_questions",
    "discovery_questions": "discovery_questions",
    "discovery": "discovery_questions",
    "watch_points": "watch_points",
    "watchpoints": "watch_points",
    "risks": "watch_points",
    "risks_to_watch": "watch_points",
    "red_flags": "watch_points",
    "key_discussion_points": "key_discussion_points",
    "discussion_points": "key_discussion_points",
    "about": "about",
    "purpose": "about",
    # The overview IS the about section. Without these a plan naming "Overview" slugs to a key of
    # its own, creating a second framing section beside the stale one it meant to replace.
    "overview": "about",
    "meeting_overview": "about",
    "meeting_purpose": "about",
    "deal_context": "deal_context",
    "meeting_type": "meeting_type",
    "historical_context": "historical_context",
    "history": "historical_context",
    "people": "people",
    "whos_on_the_call": "people",
    "who_s_on_the_call": "people",
    "company": "company",
}
# Filler words that carry no identity. Dropped before matching, so "Risks to Watch" and
# "Watch Risks" are the same request.
_ALIAS_STOPWORDS = frozenset(
    {"to", "for", "the", "of", "on", "in", "at", "a", "an", "and", "my", "our", "this", "with"}
)
# Canonical sections keyed by their distinctive words rather than an exact slug, so plurals and
# filler stop mattering. Matched as a WHOLE SET, never as a subset: "Questions for Legal" is
# {question, legal}, which matches nothing and stays a section of its own. That conservatism is
# deliberate — resolving it to ``discovery_questions`` would overwrite the host's real question
# list, which is the most damaging write available here.
CANONICAL_TOKEN_ALIASES = {
    frozenset({"question"}): "discovery_questions",
    frozenset({"question", "ask"}): "discovery_questions",
    frozenset({"discovery", "question"}): "discovery_questions",
    frozenset({"watch", "point"}): "watch_points",
    frozenset({"risk"}): "watch_points",
    frozenset({"risk", "watch"}): "watch_points",
    frozenset({"risk", "point"}): "watch_points",
    frozenset({"red", "flag"}): "watch_points",
    frozenset({"red", "flag", "watch"}): "watch_points",
    frozenset({"thing", "watch"}): "watch_points",
    frozenset({"thing", "watch", "out"}): "watch_points",
    frozenset({"discussion", "point"}): "key_discussion_points",
    frozenset({"key", "discussion", "point"}): "key_discussion_points",
    frozenset({"talking", "point"}): "key_discussion_points",
    frozenset({"about"}): "about",
    frozenset({"purpose"}): "about",
    frozenset({"overview"}): "about",
    frozenset({"meeting", "overview"}): "about",
    frozenset({"meeting", "purpose"}): "about",
    frozenset({"deal", "context"}): "deal_context",
    frozenset({"meeting", "type"}): "meeting_type",
    frozenset({"historical", "context"}): "historical_context",
    frozenset({"history"}): "historical_context",
    frozenset({"meeting", "history"}): "historical_context",
    frozenset({"people"}): "people",
    frozenset({"who", "s", "call"}): "people",
    frozenset({"attendee"}): "people",
    frozenset({"participant"}): "people",
    frozenset({"company"}): "company",
}
# A custom section whose name says plainly what it holds gets that shape, whatever the model put
# in ``type``. Ordered: the first group with a word in common wins, so "Risk Points" is a risk
# list rather than a set of chips. Only consulted for sections that are NOT canonical, so this can
# never change an existing section or merge two together — it decides rendering, never identity.
SHAPE_HINT_WORDS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"risk", "watch", "flag", "concern", "caution", "pitfall", "blocker"}),
     "flagged_list"),
    (frozenset({"question", "ask"}), "numbered_list"),
    (frozenset({"link", "source", "reference", "url", "reading"}), "links"),
    (frozenset({"topic", "theme", "tag", "language", "keyword", "skill", "point"}), "chips"),
    (frozenset({"summary", "background", "overview", "context", "note", "brief"}), "prose"),
)


def _name_tokens(key: str) -> frozenset[str]:
    """The distinctive words of a slug: filler dropped, plurals folded onto the singular.

    Both sides of every comparison run through this, so the folding only has to be consistent —
    it does not have to be correct English.
    """
    out: set[str] = set()
    for word in key.split("_"):
        if not word or word in _ALIAS_STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return frozenset(out)


def _shape_hint(key: str) -> str:
    """The block type a custom section's own name implies, or empty when it implies nothing."""
    tokens = _name_tokens(key)
    for words, block_type in SHAPE_HINT_WORDS:
        if tokens & words:
            return block_type
    return ""
# Generated elsewhere and never the chat agent's to touch: no create, no rewrite, no content
# change, no delete, whatever verb the host uses.
IMMUTABLE_SECTION_KEYS = frozenset({"people", "company", "historical_context"})
# Sections that may be edited but never removed. Empty by deliberate choice: ``discovery_questions``
# used to sit here, and is now an ordinary section — creatable when absent, deletable when present.
# Its *shape* is still not negotiable; that is pinned by ``CANONICAL_BLOCK_TYPES``, not by this set.
# The machinery stays so re-protecting a key later is a one-word change.
PERMANENT_SECTION_KEYS: frozenset[str] = frozenset()
# A section whose shape the agent cannot express can still be deleted whole, but rewriting its
# content would silently reshape it, so content edits on those are refused instead. With the typed
# blocks below this is now only the shapes the load path or the sibling service writes.
EDITABLE_BLOCK_TYPES = frozenset(WRITABLE_BLOCK_TYPES)
# A confused turn should not be able to write a wall of JSON into someone's prep.
MAX_SECTION_ITEMS = 10
# No cap on how many sections one turn may write. A prep-wide change — the agenda moved, the
# meeting became a different meeting — legitimately touches every framing section at once, and a
# cap of 2 or 3 did not stop that turn from happening: it silently truncated it to whichever
# sections the model happened to emit first, which is the questions list. A turn that writes too
# much is visible and correctable; one that quietly wrote half of what it announced is not.
# Per-section rows are still bounded by ``MAX_SECTION_ITEMS``.
MAX_NEW_SECTIONS: int | None = None
MAX_DELETE_SECTIONS = 2
# The one trigger a regeneration offer raises. Named for the whole prep rather than a section,
# because that is what the host is being asked about — the per-section triggers stay as they are.
REGENERATION_TRIGGER_ACTION = "regenerate_call_prep"
# Answers that approve a pending regeneration, matched against the WHOLE message once punctuation
# is stripped. Deliberately closed: anything else — a follow-up question, a narrowed "yes but not
# the questions", silence on the subject — drops the plan and runs an ordinary turn. An unclear
# answer must never be able to rewrite someone's prep, and the host can always ask again.
REGENERATION_APPROVALS = frozenset(
    {
        "y", "yes", "yep", "yeah", "yup", "ya", "sure", "ok", "okay", "k",
        "do it", "go ahead", "go for it", "please do", "yes please", "sounds good",
        "yes do it", "yeah do it", "sure do it", "ok do it", "okay do it",
        "do it please", "yes go ahead", "yeah go ahead", "confirm", "confirmed",
        "approve", "approved", "regenerate", "yes regenerate", "update it", "proceed",
    }
)
# A turn is at most 3 sections of 10 rows plus a short reply; this is several times that. It is a
# runaway guard, not a budget. Without it a degenerate generation runs until the context is gone —
# observed at four and a half minutes before the SDK gave up — and because ``assistant_reply`` is
# written last, a truncated turn loses the reply as well as the sections.
MAX_TURN_TOKENS = 4000
# Enterprise emits more event types than the client needs (``status``, ``done``, …), so this is an
# allowlist: anything not named here is dropped silently. Adding a type here is the whole cost of
# forwarding a new one — events pass through verbatim, keys and all.
#
# ``trigger`` carries a prompt for the user rather than content: the assistant drafted something
# without being asked to save it, and the client should offer a choice (e.g. ``{"type": "trigger",
# "action": "add_to_call_prep", "values": ["yes", "no"]}``). The answer comes back as an ordinary
# ``user_message`` on the next request, so nothing here needs to interpret it.
#
# ``reasoning`` is the assistant's working shown to the host before the reply itself — one short
# observation per event (e.g. ``{"type": "reasoning", "text": "The prep does not list a CHRO."}``).
FORWARD_EVENT_TYPES = frozenset({"message", "call_prep", "trigger", "reasoning"})


async def _get_pool() -> asyncpg.Pool:
    global _POOL
    if _POOL is None:
        if not live_db_settings.configured:
            raise RuntimeError("Live DB settings are not configured")
        _POOL = await asyncpg.create_pool(
            host=live_db_settings.db_host,
            port=live_db_settings.db_port,
            user=live_db_settings.db_user,
            password=live_db_settings.db_password,
            database=live_db_settings.db_name,
            min_size=1,
            max_size=5,
        )
    return _POOL


@dataclass
class TemporaryPrepRow:
    id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class SoaMeetingContext:
    meeting_id: int
    user_id: int
    user_email: str
    user_name: str
    meeting_title: str | None = None


@dataclass(frozen=True)
class ChatSessionSyncResult:
    """Result of syncing transcriber_call_prep_chat_sessions for a request."""

    conversation_id: str
    messages_newest_first: tuple[dict[str, str], ...]
    should_inject: bool
    conversation_changed: bool
    created: bool
    # A regeneration the assistant offered on the previous turn and the host has not answered yet.
    # ``None`` on every turn where no offer is outstanding, which is almost all of them.
    pending_regeneration: dict[str, Any] | None = None


EMPTY_CONVERSATION_PAYLOAD: dict[str, Any] = {"messages": []}
# Turns of history the prompt gets back. Three was short enough that a fact the host stated — a
# changed agenda, a correction — had aged out of ``recent_conversation`` two turns later, and the
# assistant then contradicted them from the calendar text. Pairs are short; seven is still small.
MAX_STORED_CHAT_PAIRS = 7
# Where a pending regeneration lives inside ``conversation_payload``. A sibling of ``messages``
# rather than a column of its own: the jsonb is already scoped to (user_id, external_event_id),
# already reset when the conversation changes, and this needs no migration to a live table.
PENDING_REGENERATION_KEY = "pending_regeneration"


def _as_payload_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(raw, dict):
        return raw
    return {}


def normalize_sections_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize payload to ``{"sections": [...]}``; migrate legacy flat discovery_questions."""
    data = dict(payload or {})
    sections = data.get("sections")
    if isinstance(sections, list):
        return {"sections": sections}

    # Legacy flat shape: {"discovery_questions": ["..."]}
    legacy = data.get("discovery_questions")
    if isinstance(legacy, list):
        questions = [str(q).strip() for q in legacy if str(q).strip()]
        if questions:
            return {"sections": [build_discovery_section(questions)]}
    return {"sections": []}


def build_discovery_section(
    questions: list[str],
    *,
    origin: str = "user_added",
    prompt: str | None = None,
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "id": DISCOVERY_SECTION_ID,
        "key": DISCOVERY_SECTION_KEY,
        "origin": origin,
        "block": {
            "name": "Questions to Ask",
            "type": "numbered_list",
            "data": {"items": [{"text": q} for q in questions]},
        },
    }
    if prompt:
        section["prompt"] = prompt
    return section


def slugify_section_key(name: str) -> str:
    """``"Watch Points"`` -> ``watch_points``. Empty when the name has nothing usable in it."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


def _clean_value(raw: Any) -> str:
    """Trimmed text, or empty when the model filled the field with a placeholder.

    ``"unknown"`` rendered in a prep reads as a fact about the meeting rather than an absence, so
    the junk set is stripped here once and every block shape below inherits it.
    """
    text = str(raw or "").strip()
    return "" if text.lower() in JUNK_VALUES else text


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _main_text(item: "NewSectionItem") -> str:
    """The item's primary string, whichever field the model reached for.

    ``text`` and ``label`` mean the same thing to the model and it picks between them by feel; the
    per-type output below is what makes them different. Accepting either here is what stops a
    chips section arriving under ``text`` from being dropped.
    """
    return _clean_value(item.text) or _clean_value(item.label)


def normalize_block_data(
    block_type: str,
    items: list["NewSectionItem"],
    *,
    known_urls: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """Project superset items into exactly the ``data`` shape ``block_type`` renders.

    Coercive by design: it drops what does not fit rather than raising, because a turn that half
    fits should still land the half that does. ``None`` means nothing survived — the caller
    refuses the write rather than storing an empty section.

    One flat item model covers every type, so this function is the only place that knows which
    fields a given shape actually uses. Everything else is dropped, which is what keeps a stray
    ``severity`` on a chips row from reaching the renderer.
    """
    rows: list[dict[str, Any]] = []

    if block_type in ("numbered_list", "list"):
        for item in items:
            text = _main_text(item)
            if text:
                rows.append({"text": text})

    elif block_type == "flagged_list":
        for item in items:
            text = _main_text(item)
            if not text:
                continue
            severity = _clean_value(item.severity).lower()
            # Always present on output: the renderer draws a dot for every row, and a missing
            # severity would leave it with nothing to draw.
            row: dict[str, Any] = {
                "text": text,
                "severity": severity if severity in SEVERITIES else "info",
            }
            detail = _clean_value(item.detail)
            if detail:
                row["detail"] = detail
            rows.append(row)

    elif block_type == "key_values":
        for item in items:
            label = _clean_value(item.label) or _clean_value(item.text)
            value = _clean_value(item.value) or _clean_value(item.detail)
            # "Stage: Negotiation" in one field is the common near-miss. Splitting it recovers a
            # correct row where dropping it would lose the content entirely.
            if label and not value and ":" in label:
                head, _, tail = label.partition(":")
                label, value = _clean_value(head), _clean_value(tail)
            if label and value:
                rows.append({"label": label, "value": value})

    elif block_type == "chips":
        for item in items:
            label = _main_text(item)
            if label:
                rows.append({"label": label})

    elif block_type == "links":
        for item in items:
            url = _clean_value(item.url)
            # A fabricated URL looks authoritative and goes nowhere, so only addresses that came
            # in with the meeting are allowed through. No sources in, no sources out.
            if not _is_http_url(url) or url not in known_urls:
                continue
            rows.append({"label": _main_text(item) or url, "url": url})

    elif block_type == "prose":
        # One paragraph, however many items the model split it across.
        text = " ".join(t for t in (_main_text(i) for i in items) if t).strip()
        return {"text": text} if text else None

    else:
        return None

    return {"items": rows} if rows else None


def comparable_rows(section: dict[str, Any] | None) -> list[str]:
    """One string per row, for deciding whether a turn added, reworded or removed content.

    Types keep their rows under different keys and different field names, so a single ``text``
    lookup would read every ``key_values`` or ``chips`` section as empty — and an empty "before"
    makes every edit look like an addition.
    """
    block = (section or {}).get("block") or {}
    data = block.get("data") or {}
    if block.get("type") == "prose":
        text = _clean_value(data.get("text"))
        return [text] if text else []
    out: list[str] = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            text = _clean_value(item)
            if text:
                out.append(text)
            continue
        label = _clean_value(item.get("label"))
        value = _clean_value(item.get("value"))
        if label and value:
            out.append(f"{label}: {value}")
            continue
        row = (
            _clean_value(item.get("text"))
            or label
            or _clean_value(item.get("title"))
            or _clean_value(item.get("url"))
        )
        if row:
            out.append(row)
    return out


def resolve_block_type(key: str, requested: str, existing: dict[str, Any] | None) -> str:
    """The shape a section will actually have, which is rarely the one the model named.

    Three rules, in order. A section that already exists keeps its shape — otherwise "add one more
    watch point" would quietly turn a flagged list into a plain one. A canonical key has a fixed
    shape regardless of the turn. Only a genuinely new, unrecognised section takes the model's
    word for it, and an unusable answer lands on the default rather than failing.
    """
    if existing is not None:
        current = str((existing.get("block") or {}).get("type") or "")
        if current in KNOWN_BLOCK_TYPES:
            return current
    canonical = CANONICAL_BLOCK_TYPES.get(key)
    if canonical:
        return canonical
    # The name is deterministic and the model's ``type`` is a judgement call; where the name says
    # something definite, it wins. "Risk Points" renders with severity dots even when the turn
    # labelled it a plain list.
    hint = _shape_hint(key)
    if hint:
        return hint
    return requested if requested in WRITABLE_BLOCK_TYPES else DEFAULT_BLOCK_TYPE


def build_typed_section(
    key: str,
    name: str,
    block_type: str,
    items: list["NewSectionItem"],
    *,
    prompt: str | None = None,
    known_urls: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """A section in wire shape, or ``None`` when nothing in it survived normalization."""
    data = normalize_block_data(block_type, items, known_urls=known_urls)
    if data is None:
        return None
    section: dict[str, Any] = {
        "id": f"sec_{key}",
        "key": key,
        "block": {"name": name.strip() or key, "type": block_type, "data": data},
        "origin": "user_added",
    }
    if prompt:
        section["prompt"] = prompt
    return section


def collect_known_urls(
    existing_sections: list[dict[str, Any]],
    *extra_text: str,
) -> frozenset[str]:
    """Every http(s) address already attached to this meeting.

    The allowlist a ``links`` section is checked against: the assistant may surface a source the
    host already has, never one it recalled.
    """
    blob = json.dumps(existing_sections or [], ensure_ascii=False)
    for text in extra_text:
        blob += "\n" + (text or "")
    found = re.findall(r"https?://[^\s\"'<>)\]]+", blob)
    return frozenset(url.rstrip(".,;") for url in found)

def upsert_section(payload: dict[str, Any] | None, section: dict[str, Any]) -> dict[str, Any]:
    """Replace the section with the same key, or append it. Every other section is preserved."""
    normalized = normalize_sections_payload(payload)
    sections = [s for s in (normalized.get("sections") or []) if isinstance(s, dict)]
    out: list[dict[str, Any]] = []
    replaced = False
    for existing in sections:
        if existing.get("key") == section["key"] or existing.get("id") == section["id"]:
            out.append(section)
            replaced = True
        else:
            out.append(existing)
    if not replaced:
        out.append(section)
    return {"sections": out}


def remove_section(payload: dict[str, Any] | None, key: str) -> tuple[dict[str, Any], str | None]:
    """Drop the section with ``key``. Returns the new payload and the id removed, if any."""
    normalized = normalize_sections_payload(payload)
    out: list[dict[str, Any]] = []
    removed_id: str | None = None
    for section in normalized.get("sections") or []:
        if not isinstance(section, dict):
            continue
        if section.get("key") == key:
            removed_id = str(section.get("id") or f"sec_{key}")
            continue
        out.append(section)
    return {"sections": out}, removed_id


def _section_index(
    existing_sections: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """``{key: section}`` plus ``{slugified display name: key}``, so "Watch Points" resolves too."""
    by_key: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for section in existing_sections or []:
        if not isinstance(section, dict):
            continue
        key = str(section.get("key") or "").strip()
        if not key:
            continue
        by_key[key] = section
        alias = slugify_section_key(str((section.get("block") or {}).get("name") or ""))
        if alias and alias not in aliases:
            aliases[alias] = key
    return by_key, aliases


def _resolve_section_key(name: str, aliases: dict[str, str]) -> str:
    """The key a requested section name refers to.

    A section already in the prep wins: its display name maps to whatever key it actually has, so
    an existing custom section is never retargeted at a canonical one. Only when nothing in the
    prep matches does the canonical alias table apply — which is what stops "Questions to Ask"
    from creating ``sec_questions_to_ask`` on a prep that has no questions section yet.
    """
    key = slugify_section_key(name)
    if not key:
        return ""
    if key in aliases:
        return aliases[key]
    canonical = CANONICAL_NAME_ALIASES.get(key)
    if canonical:
        return canonical
    return CANONICAL_TOKEN_ALIASES.get(_name_tokens(key), key)


# One trigger per pending change, named for what it would do. ``section`` vs ``content`` is the
# axis that matters to the host: whether the section itself appears/disappears, or only its rows.
_CONTENT_TRIGGERS = {
    "add_content": "add_content_to_{key}",
    "update_content": "update_content_in_{key}",
    "delete_content": "delete_content_from_{key}",
}


def _content_operation(before: list[str], after: list[str]) -> str:
    """Name what happened to a list by comparing it with what was there.

    The model labels its own turns, and it labels a rewrite ``add_content`` often enough that the
    label cannot name the trigger on its own — and the trigger is what the host reads before
    saying yes. The lists themselves cannot be wrong about it.
    """
    kept, proposed = set(before), set(after)
    if not before:
        return "add_content"
    if kept <= proposed and len(proposed) > len(kept):
        return "add_content"
    if proposed < kept:
        return "delete_content"
    return "update_content"


@dataclass(frozen=True)
class SectionWrite:
    """A section to upsert, with the trigger that would confirm it."""

    key: str
    section: dict[str, Any]
    trigger: str


@dataclass(frozen=True)
class SectionRemoval:
    """A section to drop whole, with the trigger that would confirm it."""

    key: str
    section_id: str
    trigger: str


@dataclass(frozen=True)
class SectionChangePlan:
    """What a turn would do to the prep, resolved against what is actually there."""

    writes: list[SectionWrite]
    removals: list[SectionRemoval]
    refusals: list[tuple[str, str]]

    @property
    def writes_anything(self) -> bool:
        return bool(self.writes or self.removals)

    def triggers(self) -> list[str]:
        """Trigger actions for what this turn is waiting on — one per pending change."""
        actions: list[str] = [w.trigger for w in self.writes]
        actions.extend(r.trigger for r in self.removals)
        # No fallback: a turn that resolved to nothing — a refusal, or advice to change nothing —
        # has nothing to confirm, and offering a yes/no for a change that cannot happen is worse
        # than offering none.
        return actions


def annotate_people_with_host(
    existing_sections: list[dict[str, Any]],
    host: "HostPersona | None",
) -> list[dict[str, Any]]:
    """Copy of the sections with the host's own row flagged ``is_you``.

    Naming the host in a separate field is not enough on its own: the model still has to work out
    which entry in a list of colleagues that name refers to. Flagging the row removes the step.
    """
    if host is None:
        return existing_sections
    target = (host.email or "").strip().lower()
    if not target:
        return existing_sections
    out: list[dict[str, Any]] = []
    for section in existing_sections or []:
        if not isinstance(section, dict) or section.get("key") != PEOPLE_SECTION_KEY:
            out.append(section)
            continue
        block_data = section.get("block") or {}
        people = (block_data.get("data") or {}).get("people") or []
        marked = [
            {**person, "is_you": True}
            if isinstance(person, dict) and target in _emails_of(person)
            else person
            for person in people
        ]
        out.append(
            {
                **section,
                "block": {
                    **block_data,
                    "data": {**(block_data.get("data") or {}), "people": marked},
                },
            }
        )
    return out


def build_section_inventory(existing_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per section actually in the prep, with what may be done to it.

    The model is otherwise guessing: it sees section contents but nothing about which are
    off-limits, and nothing that distinguishes "not there" from "there and empty". Both guesses go
    wrong the same way — it announces a change the planner then refuses, and the host is told
    something happened that did not.
    """
    inventory: list[dict[str, Any]] = []
    for section in existing_sections or []:
        if not isinstance(section, dict):
            continue
        key = str(section.get("key") or "").strip()
        if not key:
            continue
        block = section.get("block") or {}
        data = block.get("data") or {}
        rows = data.get("items")
        if not isinstance(rows, list):
            if isinstance(data.get("people"), list):
                rows = data["people"]
            else:
                # ``prose`` and ``company`` are flat objects: one row if they carry anything.
                rows = [data] if _clean_value(data.get("text") or data.get("name")) else []
        if key in IMMUTABLE_SECTION_KEYS:
            status = "immutable"
        elif key in PERMANENT_SECTION_KEYS:
            status = "permanent"
        elif str(block.get("type") or "") not in EDITABLE_BLOCK_TYPES:
            status = "deletable_only"
        else:
            status = "editable"
        inventory.append(
            {
                "key": key,
                "name": str(block.get("name") or ""),
                "type": str(block.get("type") or ""),
                "items": len(rows),
                "status": status,
            }
        )
    return inventory


def plan_section_changes(
    turn: "CallPrepChatTurn",
    existing_sections: list[dict[str, Any]],
    *,
    prompt: str | None = None,
    known_urls: frozenset[str] = frozenset(),
    # ``None`` means no cap, which is the default. Kept as a parameter so a caller that needs to
    # bound a turn still can; nothing does today.
    max_new_sections: int | None = MAX_NEW_SECTIONS,
    max_delete_sections: int = MAX_DELETE_SECTIONS,
) -> SectionChangePlan:
    """Resolve a turn's requested changes against the prep as it stands.

    The operation the model declared is a statement of intent, not the last word: creating a
    section that already exists is a content change, and editing one that does not exist is
    refused. Deciding here — rather than trusting the label — is what keeps the trigger the host
    sees matched to what would actually happen.
    """
    by_key, aliases = _section_index(existing_sections)
    refusals: list[tuple[str, str]] = []

    writes: list[SectionWrite] = []
    seen_keys: set[str] = set()
    for requested in (turn.new_sections or [])[:max_new_sections]:
        key = _resolve_section_key(requested.name, aliases)
        if not key:
            refusals.append((requested.name, "unusable name"))
            continue
        if key in IMMUTABLE_SECTION_KEYS:
            refusals.append((key, "immutable"))
            continue
        if key in seen_keys:
            refusals.append((key, "duplicate in turn"))
            continue

        existing = by_key.get(key)
        operation = requested.operation
        if existing is None:
            if operation in ("update_content", "delete_content"):
                refusals.append((key, "no such section"))
                continue
            operation = "add_section"
        else:
            block_type = str((existing.get("block") or {}).get("type") or "")
            if block_type and block_type not in EDITABLE_BLOCK_TYPES:
                refusals.append((key, f"block type {block_type} is not editable as a list"))
                continue
            if operation == "add_section":
                operation = "add_content"

        items = list(requested.items or [])[:MAX_SECTION_ITEMS]
        if not items and operation in ("add_section", "add_content"):
            refusals.append((key, "no items"))
            continue

        block_type = resolve_block_type(key, requested.type, existing)
        # A canonical section's heading is fixed; anything else keeps the heading it already has,
        # because the model retypes the name every turn and its wording drifts.
        name = CANONICAL_SECTION_NAMES.get(key) or (
            str((existing.get("block") or {}).get("name") or "") if existing else ""
        )
        section = build_typed_section(
            key,
            name or requested.name,
            block_type,
            items,
            prompt=prompt,
            known_urls=known_urls,
        )
        if section is None:
            # Everything in it was junk, unusable for this shape, or an unsourced link.
            if items:
                refusals.append((key, f"nothing usable for a {block_type} section"))
                continue
            # Deliberately emptied: keep the section, drop its rows.
            section = {
                "id": f"sec_{key}",
                "key": key,
                "block": {
                    "name": name or requested.name.strip() or key,
                    "type": block_type,
                    "data": {"text": ""} if block_type == "prose" else {"items": []},
                },
                "origin": "user_added",
            }
            if prompt:
                section["prompt"] = prompt

        if operation == "add_section" and existing is None:
            trigger = f"add_section_{key}"
        else:
            before, after = comparable_rows(existing), comparable_rows(section)
            if before == after:
                # Returning a section unchanged is not an edit. Writing it would bump the row and
                # offer the host a yes/no prompt for a change that does not exist.
                refusals.append((key, "unchanged"))
                continue
            trigger = _CONTENT_TRIGGERS[_content_operation(before, after)].format(key=key)
        seen_keys.add(key)
        writes.append(SectionWrite(key=key, section=section, trigger=trigger))

    removals: list[SectionRemoval] = []
    for name in (turn.delete_sections or [])[:max_delete_sections]:
        key = _resolve_section_key(str(name or ""), aliases)
        if not key:
            refusals.append((str(name), "unusable name"))
            continue
        if key in IMMUTABLE_SECTION_KEYS:
            refusals.append((key, "immutable"))
            continue
        if key in PERMANENT_SECTION_KEYS:
            refusals.append((key, "permanent"))
            continue
        existing = by_key.get(key)
        if existing is None:
            refusals.append((key, "no such section"))
            continue
        if key in seen_keys:
            # Written and deleted in the same turn is contradictory; the write already stands.
            refusals.append((key, "also written this turn"))
            continue
        seen_keys.add(key)
        removals.append(
            SectionRemoval(
                key=key,
                section_id=str(existing.get("id") or f"sec_{key}"),
                trigger=f"delete_section_{key}",
            )
        )

    for key, reason in refusals:
        logger.info("[live] chat: refused section op key=%s reason=%s", key, reason)

    return SectionChangePlan(writes=writes, removals=removals, refusals=refusals)


def resolve_regeneration_plan(
    plan: "RegenerationPlan | None",
    existing_sections: list[dict[str, Any]],
    *,
    max_new_sections: int | None = None,
    max_changes: int | None = None,
) -> list[dict[str, Any]]:
    """What a proposed regeneration may actually carry out, resolved against the prep.

    The same job ``plan_section_changes`` does for a write, moved one turn earlier. The host
    approves on the strength of a sentence, so anything that would be refused at write time has to
    be gone before the offer is made — otherwise they say yes to a change that never happens.

    Returns ``[]`` when nothing survives, and the caller then makes no offer at all.
    """
    if plan is None:
        return []
    max_new = (
        live_settings.regeneration_max_new_sections
        if max_new_sections is None
        else max_new_sections
    )
    max_total = live_settings.regeneration_max_changes if max_changes is None else max_changes

    by_key, aliases = _section_index(existing_sections)
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    created = 0
    for change in plan.changes or []:
        key = _resolve_section_key(str(change.section or ""), aliases)
        if not key or key in seen:
            continue
        # Immutable sections are refused at write time whatever the turn says; dropping them here
        # is what stops the reply from promising one.
        if key in IMMUTABLE_SECTION_KEYS:
            logger.info("[live] chat: regeneration dropped immutable section key=%s", key)
            continue
        existing = by_key.get(key)
        if existing is None:
            action = "add_section"
        else:
            block_type = str((existing.get("block") or {}).get("type") or "")
            if block_type and block_type not in EDITABLE_BLOCK_TYPES:
                # Not a list, so its content is not the assistant's to rewrite.
                logger.info(
                    "[live] chat: regeneration dropped key=%s reason=block type %s not editable",
                    key,
                    block_type,
                )
                continue
            # The model labels its own plan and mislabels it the same way it mislabels operations:
            # what exists decides whether this is a create or a change, not what it was called.
            action = "extend" if change.action == "extend" else "rewrite"
        if action == "add_section":
            if created >= max_new:
                continue
            created += 1
        seen.add(key)
        resolved.append(
            {
                "key": key,
                # What the host will see named in the reply, and what the applying turn is told to
                # target. Canonical headings are fixed; an existing section keeps the one it has.
                "name": CANONICAL_SECTION_NAMES.get(key)
                or (
                    str((existing.get("block") or {}).get("name") or "")
                    if existing
                    else str(change.section or "").strip()
                )
                or key,
                "action": action,
                # Settled now so the applying turn is told the exact shape to fill rather than
                # picking again from a name it is reading a second time.
                "type": resolve_block_type(key, str(change.type or ""), existing),
                "reason": _clean_value(change.reason),
            }
        )
        if len(resolved) >= max_total:
            break
    return resolved


def approved_regeneration_payload(pending: dict[str, Any] | None) -> dict[str, Any] | None:
    """The stored offer as the prompt reads it, or ``None`` when there is nothing to carry out."""
    if not pending:
        return None
    changes = [c for c in (pending.get("changes") or []) if isinstance(c, dict) and c.get("key")]
    if not changes:
        return None
    payload: dict[str, Any] = {"changes": changes}
    # The message that prompted the offer. ``recent_conversation`` usually still holds it, but it
    # is capped at three pairs — storing it means the turn that does the work always has the
    # context the work is based on.
    context = str(pending.get("context") or "").strip()
    if context:
        payload["context"] = context
    return payload


def restrict_turn_to_regeneration(
    turn: "CallPrepChatTurn",
    existing_sections: list[dict[str, Any]],
    changes: list[dict[str, Any]],
) -> None:
    """Cut a carrying-out turn back to the sections the host actually approved.

    The prompt already says to touch nothing outside the plan. This is what makes that true: the
    host agreed to a named list of sections, and a turn that reaches past it — however good the
    addition — is a change they were never shown. Narrowing is allowed, widening is not.
    """
    approved = {str(c.get("key") or "") for c in changes}
    _, aliases = _section_index(existing_sections)
    kept: list[NewSection] = []
    for section in turn.new_sections or []:
        key = _resolve_section_key(section.name, aliases)
        if key in approved:
            kept.append(section)
        else:
            logger.info(
                "[live] chat: regeneration dropped unapproved section name=%s key=%s",
                section.name,
                key,
            )
    turn.new_sections = kept
    # A regeneration rewrites, extends and adds; removing a section whole stays behind the
    # explicit-instruction path where the host named it themselves.
    turn.delete_sections = []


def extract_discovery_questions(payload: dict[str, Any] | None) -> list[str]:
    normalized = normalize_sections_payload(payload)
    for section in normalized.get("sections") or []:
        if not isinstance(section, dict):
            continue
        if section.get("key") != DISCOVERY_SECTION_KEY:
            continue
        block = section.get("block") or {}
        data = block.get("data") or {}
        items = data.get("items") or []
        out: list[str] = []
        for item in items:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
            else:
                text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _parse_conversation_messages(raw_payload: Any) -> list[dict[str, str]]:
    data = _as_payload_dict(raw_payload)
    messages = data.get("messages")
    if not isinstance(messages, list):
        return []
    out: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        text = str(item.get("text") or "").strip()
        if role in ("user", "assistant") and text:
            out.append({"role": role, "text": text})
    return out


def messages_for_prompt_chronological(
    messages_newest_first: list[dict[str, str]] | tuple[dict[str, str], ...],
) -> list[dict[str, str]]:
    """Convert newest-first storage order to chronological for the model prompt."""
    return list(reversed(list(messages_newest_first)))


async def sync_chat_session(
    *,
    user_id: int,
    external_event_id: str,
    conversation_id: str,
) -> ChatSessionSyncResult:
    """Load/create/update transcriber_call_prep_chat_sessions for (user_id, external_event_id).

    - Missing row → create with this conversation_id and empty messages.
    - Same conversation_id → keep history; re-inject only if a prior inject never succeeded.
    - Different conversation_id → update id, reset messages.

    ``should_inject`` is driven by ``injected_at``, which only ``mark_session_injected`` writes
    after a confirmed-successful inject. A failed inject leaves it NULL so the next request
    retries, instead of skipping inject forever on the strength of a matching conversation_id.
    """
    pool = await _get_pool()
    conv_id = conversation_id.strip()
    event_id = external_event_id.strip()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT conversation_id, conversation_payload, injected_at
            FROM transcriber_call_prep_chat_sessions
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            event_id,
        )
        if row is None:
            await conn.execute(
                """
                INSERT INTO transcriber_call_prep_chat_sessions (
                  user_id, external_event_id, conversation_id, conversation_payload, injected_at
                )
                VALUES ($1, $2, $3::uuid, $4::jsonb, NULL)
                """,
                user_id,
                event_id,
                conv_id,
                json.dumps(EMPTY_CONVERSATION_PAYLOAD),
            )
            return ChatSessionSyncResult(
                conversation_id=conv_id,
                messages_newest_first=(),
                should_inject=True,
                conversation_changed=True,
                created=True,
            )

        existing_conv = str(row["conversation_id"])
        if existing_conv == conv_id:
            messages = _parse_conversation_messages(row["conversation_payload"])
            # A NULL injected_at means the previous inject for this conversation never
            # succeeded — retry it rather than chatting without call prep.
            never_injected = row["injected_at"] is None
            if never_injected:
                logger.info(
                    "[live] soa chat re-injecting: injected_at is NULL "
                    "conversation_id=%s external_event_id=%s user_id=%s",
                    conv_id,
                    event_id,
                    user_id,
                )
            return ChatSessionSyncResult(
                conversation_id=conv_id,
                messages_newest_first=tuple(messages),
                should_inject=never_injected,
                conversation_changed=False,
                created=False,
                # Only this branch can carry one: the other two either created the row or reset it
                # for a new conversation, and an offer does not survive a new thread.
                pending_regeneration=parse_pending_regeneration(row["conversation_payload"]),
            )

        # New conversation: reset history and clear injected_at so the inject below is what
        # stamps it. mark_session_injected is the only writer of injected_at.
        await conn.execute(
            """
            UPDATE transcriber_call_prep_chat_sessions
            SET conversation_id = $3::uuid,
                conversation_payload = $4::jsonb,
                injected_at = NULL,
                updated_at = NOW()
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            event_id,
            conv_id,
            json.dumps(EMPTY_CONVERSATION_PAYLOAD),
        )
        return ChatSessionSyncResult(
            conversation_id=conv_id,
            messages_newest_first=(),
            should_inject=True,
            conversation_changed=True,
            created=False,
        )


async def mark_session_injected(
    *,
    user_id: int,
    external_event_id: str,
) -> None:
    """Set injected_at after a successful enterprise inject call."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE transcriber_call_prep_chat_sessions
            SET injected_at = NOW(),
                updated_at = NOW()
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            external_event_id.strip(),
        )


async def append_chat_session_turn(
    *,
    user_id: int,
    external_event_id: str,
    user_message: str,
    assistant_reply: str,
    max_pairs: int = MAX_STORED_CHAT_PAIRS,
) -> None:
    """Prepend latest user+assistant pair (newest first); keep at most max_pairs pairs."""
    pool = await _get_pool()
    event_id = external_event_id.strip()
    user_text = (user_message or "").strip()
    assistant_text = (assistant_reply or "").strip()
    if not user_text and not assistant_text:
        return

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT conversation_payload
            FROM transcriber_call_prep_chat_sessions
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            event_id,
        )
        if row is None:
            return
        existing = _parse_conversation_messages(row["conversation_payload"])
        new_pair: list[dict[str, str]] = []
        if assistant_text:
            new_pair.append({"role": "assistant", "text": assistant_text})
        if user_text:
            new_pair.append({"role": "user", "text": user_text})
        combined = new_pair + existing
        # Keep at most max_pairs * 2 messages (each pair = user + assistant).
        trimmed = combined[: max_pairs * 2]
        # Merge rather than replace. ``messages`` is the only key this function owns; a pending
        # regeneration lives beside it, and writing a bare {"messages": ...} here would drop the
        # offer the host is in the middle of answering.
        merged = _as_payload_dict(row["conversation_payload"])
        merged["messages"] = trimmed
        await conn.execute(
            """
            UPDATE transcriber_call_prep_chat_sessions
            SET conversation_payload = $3::jsonb,
                updated_at = NOW()
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            event_id,
            json.dumps(merged),
        )


def parse_pending_regeneration(raw_payload: Any) -> dict[str, Any] | None:
    """The outstanding regeneration offer in a session payload, or ``None``.

    Tolerant on purpose: a payload written before this feature existed simply has no such key, and
    a malformed one is treated as no offer rather than as a reason to fail the turn.
    """
    pending = _as_payload_dict(raw_payload).get(PENDING_REGENERATION_KEY)
    if not isinstance(pending, dict):
        return None
    changes = [c for c in (pending.get("changes") or []) if isinstance(c, dict) and c.get("key")]
    if not changes:
        return None
    return {
        "changes": changes,
        "context": str(pending.get("context") or ""),
        "fingerprint": str(pending.get("fingerprint") or ""),
    }


def sections_fingerprint(sections: list[dict[str, Any]] | None) -> str:
    """A stable digest of what the prep held when a plan was made.

    The offer names sections by what was in them at the time. If something else moved the prep in
    between — another device, the load path finishing — carrying out the old plan would apply an
    intent the host approved against a prep that no longer exists, so the plan is dropped instead.
    """
    shape = [
        [str(s.get("key") or ""), comparable_rows(s)]
        for s in (sections or [])
        if isinstance(s, dict)
    ]
    blob = json.dumps(shape, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def regeneration_is_approved(user_message: str, additional_context: str | None) -> bool:
    """Did the host just say yes to a pending regeneration?

    Decided here rather than by the model. The model already interprets "yes" for its own drafts,
    but this answer authorises a write across several sections at once, so it is settled by a
    closed set of affirmatives — and anything outside it counts as "not approved".
    """
    context = (additional_context or "").lower()
    if REGENERATION_TRIGGER_ACTION in context:
        # The shape the client sends back:
        # `User selected "yes" to the action: regenerate_call_prep`.
        head = context.split(REGENERATION_TRIGGER_ACTION)[0]
        if '"no"' in head or "'no'" in head or " no " in head:
            return False
        if '"yes"' in head or "'yes'" in head or " yes " in head:
            return True
    answer = re.sub(r"[^a-z0-9\s]", "", (user_message or "").strip().lower()).strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer in REGENERATION_APPROVALS


async def store_pending_regeneration(
    *,
    user_id: int,
    external_event_id: str,
    changes: list[dict[str, Any]],
    context: str,
    fingerprint: str,
) -> None:
    """Record the offer so the next turn can carry out exactly what was announced."""
    pool = await _get_pool()
    event_id = external_event_id.strip()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT conversation_payload
            FROM transcriber_call_prep_chat_sessions
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            event_id,
        )
        if row is None:
            return
        payload = _as_payload_dict(row["conversation_payload"])
        payload[PENDING_REGENERATION_KEY] = {
            "changes": changes,
            "context": context,
            "fingerprint": fingerprint,
        }
        await conn.execute(
            """
            UPDATE transcriber_call_prep_chat_sessions
            SET conversation_payload = $3::jsonb,
                updated_at = NOW()
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            event_id,
            json.dumps(payload),
        )


async def clear_pending_regeneration(*, user_id: int, external_event_id: str) -> None:
    """Drop the outstanding offer. Called once the host answers it, either way."""
    pool = await _get_pool()
    event_id = external_event_id.strip()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT conversation_payload
            FROM transcriber_call_prep_chat_sessions
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            event_id,
        )
        if row is None:
            return
        payload = _as_payload_dict(row["conversation_payload"])
        if PENDING_REGENERATION_KEY not in payload:
            return
        payload.pop(PENDING_REGENERATION_KEY, None)
        await conn.execute(
            """
            UPDATE transcriber_call_prep_chat_sessions
            SET conversation_payload = $3::jsonb,
                updated_at = NOW()
            WHERE user_id = $1 AND external_event_id = $2
            """,
            user_id,
            event_id,
            json.dumps(payload),
        )


async def load_soa_meeting_context(
    external_event_id: str,
    *,
    user_id: int | None = None,
) -> SoaMeetingContext | None:
    """Resolve meeting + user name/email from external_event_id (optionally scoped by user_id)."""
    pool = await _get_pool()
    meetings_table = live_settings.meetings_table
    async with pool.acquire() as conn:
        if user_id is None:
            row = await conn.fetchrow(
                f"""
                SELECT
                  m.id AS meeting_id,
                  m.user_id AS user_id,
                  m.title AS meeting_title,
                  m.organizer_email AS organizer_email,
                  u.name AS user_name,
                  u.email AS user_email
                FROM {meetings_table} AS m
                LEFT JOIN users AS u ON u.id = m.user_id
                WHERE m.external_event_id = $1
                ORDER BY m.updated_at DESC NULLS LAST, m.id DESC
                LIMIT 1
                """,
                external_event_id,
            )
        else:
            row = await conn.fetchrow(
                f"""
                SELECT
                  m.id AS meeting_id,
                  m.user_id AS user_id,
                  m.title AS meeting_title,
                  m.organizer_email AS organizer_email,
                  u.name AS user_name,
                  u.email AS user_email
                FROM {meetings_table} AS m
                LEFT JOIN users AS u ON u.id = m.user_id
                WHERE m.external_event_id = $1
                  AND m.user_id = $2
                ORDER BY m.updated_at DESC NULLS LAST, m.id DESC
                LIMIT 1
                """,
                external_event_id,
                user_id,
            )
    if row is None:
        return None
    if row["user_id"] is None:
        logger.warning(
            "[live] soa meeting missing user_id external_event_id=%s meeting_id=%s",
            external_event_id,
            row["meeting_id"],
        )
        return None
    user_email = (row["user_email"] or row["organizer_email"] or "").strip()
    user_name = (row["user_name"] or "").strip() or user_email or "User"
    return SoaMeetingContext(
        meeting_id=int(row["meeting_id"]),
        user_id=int(row["user_id"]),
        user_email=user_email,
        user_name=user_name,
        meeting_title=(row["meeting_title"] or None),
    )


def _enterprise_base_url() -> str:
    if not enterprise_search_settings.configured:
        raise RuntimeError("ENTERPRISE_SEARCH_BASE_URL is not set")
    return enterprise_search_settings.enterprise_search_base_url.rstrip("/")


async def _post_enterprise(url: str, request_body: dict[str, Any], *, as_text: bool = False) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, json=request_body)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[live] enterprise request failed url=%s", url)
        return {
            "ok": False,
            "url": url,
            "request": request_body,
            "error": str(exc),
        }

    body: Any
    if as_text:
        body = resp.text
    else:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = resp.text

    return {
        "ok": resp.is_success,
        "url": url,
        "status_code": resp.status_code,
        "request": request_body,
        "response": body,
    }


async def call_enterprise_inject_call_prep(
    *,
    user_id: int,
    meeting_id: int,
    user_email: str,
    conversation_id: str,
) -> dict[str, Any]:
    """POST /inject-call-prep on enterprise search; returns status + body for debug printing."""
    base = _enterprise_base_url()
    url = f"{base}/inject-call-prep"
    request_body = {
        "user_id": user_id,
        "conv_id": conversation_id,
        "meeting_id": meeting_id,
        "user_email": user_email,
    }
    return await _post_enterprise(url, request_body)


async def call_enterprise_chat(
    *,
    user_id: int,
    name: str,
    email: str,
    user_message: str,
    conversation_id: str,
) -> dict[str, Any]:
    """POST /chat on enterprise search; response is NDJSON text."""
    base = _enterprise_base_url()
    url = f"{base}/chat"
    request_body = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "message_id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "user_message": user_message,
    }
    return await _post_enterprise(url, request_body, as_text=True)


def compose_user_message(user_message: str, additional_context: str | None) -> str:
    """Fold ``additional_context`` into the message body sent to the enterprise chat service.

    That service takes exactly one free-text field, so background can only reach its model by
    riding along inside the message. Which makes the framing load-bearing: concatenated raw, the
    context reads as part of what the host said, and the assistant answers the context instead of
    the question. The label and the "only if it helps" instruction keep the host's message as the
    thing being answered.

    Empty, whitespace-only or absent context returns the message **unchanged**, so callers that do
    not use this field see byte-identical behaviour.
    """
    context = (additional_context or "").strip()
    if not context:
        return user_message
    return (
        f"{user_message}\n\n"
        "---\n"
        "Additional context for this request (background only, not part of the message above). "
        "Use it only where it helps you answer that message; ignore it otherwise.\n"
        f"{context}"
    )


def parse_forwardable_line(line: str) -> dict[str, Any] | None:
    """One NDJSON line from enterprise -> the event to forward, or ``None`` to drop it."""
    stripped = (line or "").strip()
    if not stripped:
        return None
    try:
        event = json.loads(stripped)
    except Exception:  # noqa: BLE001
        logger.warning("[live] soa chat skipped non-json line=%r", stripped[:200])
        return None
    if not isinstance(event, dict):
        return None
    return event if event.get("type") in FORWARD_EVENT_TYPES else None


async def stream_enterprise_chat(
    *,
    user_id: int,
    name: str,
    email: str,
    user_message: str,
    conversation_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield forwardable events from enterprise ``/chat`` **as they arrive**.

    Enterprise serves this as SSE and generates over several seconds. Reading the body with a
    plain ``post`` waits for the last byte and throws that away, so the host sees nothing until
    the model has finished. Streaming keeps their timing intact all the way to the client.
    """
    url = f"{_enterprise_base_url()}/chat"
    request_body = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "message_id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "user_message": user_message,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=request_body) as resp:
            if not resp.is_success:
                detail = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"Enterprise chat failed: {resp.status_code} {detail}")
            async for line in resp.aiter_lines():
                event = parse_forwardable_line(line)
                if event is not None:
                    yield event


def iter_forwardable_chat_events(raw_body: str) -> Iterator[dict[str, Any]]:
    """Parse enterprise NDJSON and yield the events the client is meant to see.

    Each surviving event is yielded **whole** — every key, unmodified — so the wire shape is
    owned by the enterprise service and this layer never has to track it.
    """
    for line in (raw_body or "").splitlines():
        event = parse_forwardable_line(line)
        if event is not None:
            yield event


@dataclass(frozen=True)
class HostPersona:
    """Who is actually chatting — the one thing the prep payload cannot say on its own.

    Every participant's row carries the same ``people`` section, so "which of these is me" is not
    answerable from the prep. ``non_soa_meetings.participant_user_metadata`` maps ``user_id`` to an
    email, and the email matches a row in that section, which is where the title and background
    come from.
    """

    user_id: int
    email: str
    name: str = ""
    title: str = ""
    role: str = ""
    background: str = ""

    def as_prompt_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name or self.email or "the host", "email": self.email}
        for field in ("title", "role", "background"):
            value = getattr(self, field)
            if value:
                out[field] = value
        return out


def _emails_of(person: dict[str, Any]) -> list[str]:
    """Every address a person row might be keyed by, newest shape first.

    Older prep rows carried the address in ``detail`` where newer ones use ``email``, and rows
    written between the two carry a prose background in ``detail`` instead. Reading both costs
    nothing and keeps preps generated months apart matchable.
    """
    out = []
    for field in ("email", "detail"):
        value = str(person.get(field) or "").strip().lower()
        if value and "@" in value and " " not in value:
            out.append(value)
    return out


async def load_host_persona(
    external_event_id: str,
    user_id: int,
    existing_sections: list[dict[str, Any]] | None = None,
) -> HostPersona | None:
    """Resolve the chatting user to a person on the call, or ``None`` if they cannot be placed."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            """
            SELECT participant_user_metadata
            FROM non_soa_meetings
            WHERE external_event_id = $1
            LIMIT 1
            """,
            external_event_id,
        )
    entries = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(entries, list):
        logger.info("[live] chat: no participant metadata external_event_id=%s", external_event_id)
        return None

    email = ""
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # Stored as a string in jsonb; the request carries an int.
        if str(entry.get("user_id") or "").strip() == str(user_id):
            email = str(entry.get("email") or "").strip()
            break
    if not email:
        logger.info(
            "[live] chat: user_id=%s not in participant metadata external_event_id=%s",
            user_id,
            external_event_id,
        )
        return None

    for section in existing_sections or []:
        if not isinstance(section, dict) or section.get("key") != PEOPLE_SECTION_KEY:
            continue
        for person in ((section.get("block") or {}).get("data") or {}).get("people") or []:
            if not isinstance(person, dict) or email.lower() not in _emails_of(person):
                continue
            return HostPersona(
                user_id=user_id,
                email=email,
                name=str(person.get("name") or "").strip(),
                title=str(person.get("title") or person.get("es_title") or "").strip(),
                role=str(person.get("role") or "").strip(),
                background=str(person.get("detail") or "").strip() if "@" not in str(
                    person.get("detail") or ""
                ) else "",
            )

    # On the call per the metadata, but not in the people section — still worth naming.
    return HostPersona(user_id=user_id, email=email)


async def load_or_create_temporary_prep(
    external_event_id: str,
    user_id: int,
) -> TemporaryPrepRow:
    """Load the latest temporary_meeting_call_prep row, or create a new pending one."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, payload
            FROM temporary_meeting_call_prep
            WHERE external_event_id = $1 AND user_id = $2
            ORDER BY created_at DESC
            LIMIT 1
            """,
            external_event_id,
            user_id,
        )
        if row is not None:
            payload = normalize_sections_payload(_as_payload_dict(row["payload"]))
            return TemporaryPrepRow(id=str(row["id"]), payload=payload)

        # No row: create with empty sections payload.
        created = await conn.fetchrow(
            """
            INSERT INTO temporary_meeting_call_prep (external_event_id, user_id, payload)
            VALUES ($1, $2, $3::jsonb)
            RETURNING id, payload
            """,
            external_event_id,
            user_id,
            json.dumps({"sections": []}),
        )
        created_payload = normalize_sections_payload(_as_payload_dict(created["payload"]))
        return TemporaryPrepRow(id=str(created["id"]), payload=created_payload)


async def update_temporary_prep_sections(
    row_id: str,
    payload: dict[str, Any],
) -> None:
    """Overwrite payload with a sections-shaped object (caller merges first)."""
    pool = await _get_pool()
    normalized = normalize_sections_payload(payload)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE temporary_meeting_call_prep
            SET payload = $1::jsonb,
                ai_generated_at = NOW(),
                error_message = NULL,
                updated_at = NOW()
            WHERE id = $2
            """,
            json.dumps(normalized),
            row_id,
        )


class NewSectionItem(BaseModel):
    """One row of a section, as a superset of what every block type needs.

    One flat model rather than a shape per type: structured output handles unions badly, and the
    schema is generated once for all of them, so a union would put every variant in front of the
    model on every turn. Which fields a type actually uses is decided in ``normalize_block_data``
    — anything not used there is dropped before the row is stored.
    """

    text: str = ""
    label: str = ""
    value: str = ""
    detail: str = ""
    severity: str = ""
    url: str = ""


class NewSection(BaseModel):
    """A section to write. ``items`` is always the FULL list that section should end up with.

    ``operation`` states what the host asked for, and only names the trigger they are shown — the
    payload is a full replace either way. It is reconciled against the prep in
    ``plan_section_changes``, so a mislabelled turn is corrected rather than trusted.
    """

    name: str
    # Only the shapes the assistant may write. A section that already exists keeps its stored type
    # and a canonical key forces its own, so this is consulted only when creating something new —
    # see ``resolve_block_type``.
    type: Literal[
        "numbered_list", "flagged_list", "key_values", "chips", "prose", "links"
    ] = "numbered_list"
    operation: Literal["add_section", "add_content", "update_content", "delete_content"] = (
        "add_content"
    )
    items: List[NewSectionItem] = Field(default_factory=list)


class RegenerationChange(BaseModel):
    """One section a proposed regeneration would touch.

    Intent only — no content. The host approves on the strength of the reply that describes these,
    and the rows themselves are written on the turn that carries the plan out. That keeps a
    declined offer cheap and keeps the promise at the level the host was actually asked about.
    """

    section: str
    action: Literal["rewrite", "extend", "add_section"] = "rewrite"
    # Only consulted for ``add_section``. An existing section keeps the shape it already has,
    # decided by ``resolve_block_type`` rather than here.
    type: str = ""
    reason: str = ""


class RegenerationPlan(BaseModel):
    """A prep-wide refresh the assistant is offering, resolved against what is in the prep."""

    # What the host's message revealed that the prep does not reflect. Stated before the changes,
    # so the changes are derived from the gap rather than the gap justified after the fact.
    new_facts: List[str] = Field(default_factory=list)
    changes: List[RegenerationChange] = Field(default_factory=list)


class CallPrepChatTurn(BaseModel):
    """Structured output from the non-SOA call-prep assistant.

    Field order is load-bearing. Structured output is generated in declaration order, so a field
    declared before another is written before it exists to be read.

    ``reasoning`` comes first so the working streams out ahead of everything else.

    ``needs_confirmation`` comes next, before the sections, so the consent decision is made on the
    host's message alone. Declared after them, it was being answered about a change already written
    out — and a change already written wants to be saved.

    ``assistant_reply`` comes LAST, after every field describing what the turn does, so the model
    has already committed to the change before it writes a sentence about it. With the reply declared first it was
    composing "I've updated the About section" before deciding whether to return one, and nothing
    made the two agree.
    """

    reasoning: List[str] = Field(default_factory=list)
    # Decided BEFORE the sections it governs. Whether a turn may write depends only on what the
    # host asked for, so it is answerable without knowing the content — and composing the change
    # first was enough to make writing feel like the decision already taken.
    needs_confirmation: bool = False
    # A prep-wide refresh offered because the host's message carried context the prep does not
    # reflect. ``None`` on almost every turn.
    #
    # Sits here, with the other decision, for the same reason: whether this is a whole-prep
    # situation is answerable from the message and the inventory alone. Declared after the
    # sections it would replace, the model would already have drafted the one small edit before
    # deciding to offer the larger fix, and a change already written wants to be kept.
    regeneration: RegenerationPlan | None = None
    # Every section the turn writes, questions included. There is no separate path for questions:
    # one mechanism means one set of rules, and the model cannot pick the wrong door.
    new_sections: List[NewSection] = Field(default_factory=list)
    # Sections to remove whole. Names, resolved against what is actually in the prep; immutable
    # sections are refused there rather than here.
    delete_sections: List[str] = Field(default_factory=list)
    assistant_reply: str = Field(min_length=1)


# The prompt with the regeneration material still marked off. Everything between the sentinels is
# kept when the feature is on and cut when it is off, so a deployment with it disabled runs an
# assistant that was never told the capability exists rather than one told not to use it.
_SYSTEM_PROMPT_TEMPLATE = """You are the CALL-PREP ASSISTANT for a live meeting coach used in high-stakes sales and recruiting calls.

## Your role

You help the host prepare for an upcoming meeting. You can:
- Answer questions about the meeting using the provided title, description, and current prep sections.
- Generate, refine, add, remove, or reorder DISCOVERY QUESTIONS — questions the HOST asks the
  OTHER participants, never questions put to the host.
- Create, edit and delete other prep sections — watch points, objections, notes, anything they name.

You do NOT:
- Talk about implementation details (databases, indices, arrays, “updating the backend”).
- Role-play the meeting participant or answer the discovery questions yourself.
- Invent CRM history, attendees, or deal facts that are not in the inputs.

## Inputs you receive (JSON)

- `meeting_title`
- `meeting_description`
- `host`: WHO YOU ARE WRITING FOR — their `name`, `email`, and where known `title`, `role` and
  `background`. Their own entry in the `people` section is flagged `is_you`. May be absent; see
  below.
- `existing_sections`: the FULL current content of every section in the prep (may be empty) — every
  row of every section, not just their names. This is what you read to judge whether a section
  still holds.
- `section_inventory`: one row per section that EXISTS right now — `key`, `name`, `type`, `items`,
  `status`. Read this before any section operation: it is the authority on WHICH sections exist and
  what may be done to each, and `type` tells you which fields to fill when editing it. It carries
  no content — `items` is a row count, not a summary — so never judge what a section SAYS from it.
  For that, read the section itself in `existing_sections`.
- `existing_discovery_questions`: convenience list extracted from the discovery section (may be empty).
- `user_message`: the host’s latest message you must respond to.
- `recent_conversation`: up to the last few prior chat turns in chronological order (`role`, `text`), loaded from the session store.
- `additional_context` (optional): background the client attached to this turn. It is NOT something the host typed and is never the thing you answer — use it only where it helps you answer `user_message`, and ignore it otherwise. It may carry the host's answer to a yes/no prompt you raised, e.g. `User selected "yes" to the action: add_to_call_prep`. Treat such an answer as an explicit instruction: commit the list you drafted last turn.
<<<REGEN>>>- `approved_regeneration` (optional): a prep-wide refresh you offered on an earlier turn that the host has now approved. When it is present, this turn CARRIES IT OUT — see “Regenerating the prep”.
<<</REGEN>>>
## Who you are writing for

There are two sides on every call, and confusing them ruins the output.

- **The host** — the person chatting with you, identified by `host` and flagged `is_you` in the
  `people` section. They will ASK. Everything you write is for them to say out loud.
- **Everyone else on the call** — they will BE ASKED. Never write a question for them, and never
  write a question that is really aimed AT the host.

Before writing any question, check the direction: could the host answer this from their own job?
Then it is the wrong way round. A QA engineer asks about design intent, rollout risk and what
changed; she does not ask "what edge cases should we watch during QA" — that is a question someone
else would put to HER. Turn it around, or drop it.

Their `title`, `role` and `background` shape the angle: write questions that person would credibly
ask, about what they specifically need to know. Do not write questions a colleague on their own
side would ask — a designer, an engineer and a recruiter walk out of the same meeting needing
different things.

Write in the host's voice, addressed to the other participants. Prefer "you" and "your" for the
other side. Avoid "we" and "our", which blur who is asking whom.

**When `host` is absent** — you could not be placed on the call. Write for the meeting's own side
in the neutral case, keep the direction outward, and do not guess which named person you are.
Never claim to be one of the participants.

## Discovery questions

Questions the host will ask are a section like any other, written through `new_sections` under the
name **Questions to Ask**. There is no separate field for them.

- Keep the list focused, usually 3–7.
- Specific, open-ended, professional.
- Return the FULL list the section should end up with, exactly as for any other section.
- Its shape is always `numbered_list`, whatever you put in `type`.
- It is an ordinary section otherwise: create it when it is not listed, add to it, reword, reorder,
  remove individual questions, or delete the whole section when the host asks.

**Only return it when the turn changes it.** If the host asked something informational — what the
meeting is about, which question matters most, who is attending — answer them and return NOTHING in
`new_sections`. Do not re-send the current questions to show you read them; an unchanged list is not
an edit, and repeating it back is not an answer.

**When asked to draft questions, write NEW ones.** "Draft me some questions", "what should I ask?",
"give me a few more" are requests for questions that are not already there. Read
`existing_discovery_questions` to see what has been asked, then write something else. Returning the
list they already have is the one answer that is always wrong.

## Reasoning (usually empty)

`reasoning` is your working, shown to the host **before** your reply. It exists for the turns where
you genuinely had to work something out — not to narrate every turn.

Emit reasoning ONLY when you actually reasoned:
- You reconciled something conflicting or absent ("The prep lists no CHRO among the participants").
- You chose what to keep, cut, or reorder, and the choice was not obvious.
- You inferred something from the description or history that the host did not state.
- You judged which of several angles the question was really asking about.

Emit NOTHING — an empty list — when the turn was easy, which is most turns:
- Acknowledging or confirming ("yes", "no", "thanks").
- A direct edit you were told to make.
- Answering a question whose answer sits plainly in the inputs.
- Any turn where the honest reasoning would be "the host asked, so I did it".

Each entry is one short sentence stating something you established — a finding, not a plan. Write
"The description does not mention budget authority", never "I will now look at the description" or
"Let me think about this". At most 3 entries; one is common; zero is the default.

If you are unsure whether a turn deserves reasoning, it does not.

## Other sections

Beyond discovery questions you can create, edit and delete other prep sections through
`new_sections` and `delete_sections`.

**Writing a section** — one entry in `new_sections`, carrying the section **name**, the
**operation**, and the **full** list of items that section should end up with. Never send a delta:
to add one watch point to three existing ones, return all four.

`operation` says what the host asked for. Because you always return the full list, work it out by
comparing the list you are about to send against the one that is there now:

- `add_section` — the section does not exist yet and you are creating it.
- `add_content` — every current row survives unchanged, and you added rows on top.
- `update_content` — the same subjects, reworded, reordered or rewritten. Nothing is being dropped
  and nothing is being introduced; the rows just read differently.
- `delete_content` — rows are gone and none were added. Return the rows that SURVIVE, not the ones
  being cut. An empty list empties the section but keeps it.

If a turn both drops and adds rows, label it `update_content`. Never label a rewrite `add_content`:
the host is shown the trigger by name, and "add" on a turn that replaces their list reads as a lie.

**Choosing the section's `type`.** Every section has a shape, and the shape decides how it is
rendered. Pick it from what the content IS:

| `type` | Use it for | Fill these item fields | Leave everything else `""` |
|---|---|---|---|
| `numbered_list` | questions, steps, plain ordered points | `text` | |
| `flagged_list` | watch points, risks, red flags, signals | `text`, `severity`, `detail` | |
| `key_values` | deal context, mandate, relationship owner, attributes — anything that is a label and a value | `label`, `value` | |
| `chips` | key discussion points, languages, topics, short tags | `label` | |
| `prose` | about, purpose, a paragraph of background | `text` (ONE item) | |
| `links` | sources, references | `label`, `url` | |

Rules that matter:

- **The list is closed.** Those six are the only values. Never invent a type.
- **`numbered_list` is the fallback.** When nothing above genuinely fits the content, use it. Do
  not force content into a shape it does not have.
- **Fill only that type's fields.** Every field you are not told to fill stays `""`. A `severity`
  on a chips row, or a `url` on a watch point, is noise that gets dropped.
- `severity` is exactly `high`, `medium`, `low` or `info`. `info` is a neutral signal worth
  noticing, not a risk — use it rather than inflating something to `low`.
- `detail` on a watch point is one sentence of supporting context. Nothing else uses it.
- **`links` may only carry a URL that already appears in the inputs.** Never write a URL from
  memory. If you do not have one in front of you, do not create a links section.
- **A section's shape is fixed once it exists.** When you edit a section already in
  `section_inventory`, keep returning items in ITS type's fields — the `type` you send is ignored
  for it. Its shape is in the inventory row.
- Some names have a fixed shape whatever you pick: `discovery_questions` is always
  `numbered_list`, `watch_points` always `flagged_list`, `key_discussion_points` always `chips`,
  `about` always `prose`, `deal_context` always `key_values`.

At most 10 items each. Write as many sections as the turn genuinely changes — no more.

**Deleting a section** — put its name in `delete_sections` to remove the whole section, heading and
all. Use this only when the host wants the section gone, not merely emptied. At most 2 per turn.

**Choosing where content goes.** Decide by what the content *is*, not by which field is easier.

Apply this test to every line you are about to write: **would the host say this sentence out loud
to the other side, as a question?** Only then is it a discovery question. Anything the host merely
*notices*, *tracks*, *remembers* or *reacts to* is not — it belongs in a section.

- "What is your rollout timeline?" — they say it. Question.
- "Watch whether the budget owner is absent" — they notice it. Not a question.

Read the ask the same way. "What should I ask?" wants questions. "What should I keep an eye on?",
"what should I watch for?", "anything I should be careful about?" want watch points — the answer is
a list of things to notice, so it goes in `new_sections`, never in `discovery_questions`, even
though the host phrased the request as a question to you.

Putting non-question content in `discovery_questions` overwrites the host's question list with
something that does not belong there. It is the single most damaging mistake available to you: the
list is replaced outright and what was there is gone.

The same reading picks the type. "What should I watch for?" is a `flagged_list` section — not
questions, and not a plain list. "What are the key discussion points?" is `chips`. "What's the deal
context?" is `key_values`. "Tell me about them" written into the prep is `prose`.

## Read `section_inventory` before you touch a section

It is the ground truth for WHAT EXISTS, and it overrules anything in `recent_conversation` — a
section you drafted or announced earlier is NOT in the prep unless it appears here. What a section
says is a different question, answered from its content in `existing_sections`.

- **Not listed** → the section does not exist. Creating it is `add_section`. Editing or deleting it
  is impossible: say there is no such section rather than pretending to act on it.
- `status: "editable"` → full control: add content, update content, delete content, delete the
  section.
- `status: "deletable_only"` → the section is not a list, so its content is not yours to rewrite.
  You may delete the whole section; refuse content edits.
- `status: "immutable"` → nothing at all. See below.

## What you may not change

**`people`, `company` and `historical_context` are immutable.** They are generated elsewhere and
nothing you return can touch them. You cannot create, add to, rewrite, reorder, empty or delete
them, and you cannot move their content somewhere else — whatever verb the host uses. Never put
them in `new_sections` or `delete_sections`.

When the host asks for one of these, say plainly that it is not something you can change, change
nothing, and do not offer to do it once they supply content.

**Never claim to have changed something you did not.** You write `assistant_reply` last, after
`new_sections` and `delete_sections` are already on the page — so before writing a word of it, read
back what you actually returned and describe only that. An empty `new_sections` means you changed
nothing, and your reply must say so. If a request falls outside what you can do, say so
directly. Do not describe a section as added, updated, emptied or deleted unless you
returned it in this turn's output, and never write "will be deleted" or "has been added" about
something you are refusing or cannot reach. The two ways this goes wrong:

- The host asks to delete an immutable section. Say it cannot be deleted. Do not acknowledge the
  request as though you carried it out.
- The host asks you to edit a section you only *drafted* on an earlier turn and they never
  confirmed. It is not in `section_inventory`, so it does not exist — offer to create it instead of
  reporting an edit that cannot happen.

## Saving vs. asking first

The host's prep is theirs. Only write to it when they actually told you to.

**The test for writing without asking has TWO parts, and both must hold.** The host's message must
name the OPERATION (add, remove, update, delete, clear) *and* name the TARGET it applies to. Both,
in the message itself. Then set `needs_confirmation` to false and return the change.

Passes — operation and target are both named:
- "add a question about budget" — add / a question
- "remove the second one" — remove / the second one
- "delete the watch points section" — delete / the watch points section
- "clear out the objections" — clear / the objections
- "add a watch points section with three things to watch" — add / a watch points section

Fails — draft it and ask instead:
- "what should I watch out for?" — no operation
- "here is the agenda for the call: ..." — no operation, no target
- "they just told me the budget moved to Q3" — no operation, no target
- "make this better" — no target
- "the client cares most about integration" — no operation, no target

**Context the host gives you is not an instruction.** An agenda, a pasted email, notes, a
correction about the meeting, "FYI they moved the date" — these are things for you to KNOW, not
things to do. Use them to answer, and to draft something worth offering, but never write to the
prep off the back of them. The host telling you a fact is not the host asking you to change their
prep, however obviously useful the change would be.

When only one half is present, ask. A confirmation prompt costs the host one tap; a change they did
not ask for costs them an edit and the belief that their prep stays where they left it.

**Anything softer — draft it and ask.** "Draft me some questions", "what should I watch out for?",
"suggest a few", "anything I should keep an eye on?" — the host asked to *see* something, not to
change their prep. Write it out in `assistant_reply` so they can read it, put the same full content
in `new_sections`, and set `needs_confirmation` to true. The host will be shown a yes/no prompt for
each pending change.

**This applies to removals too, and that is the case most often missed.** "Which of these would you
cut?", "my list feels long", "could these be sharper?", "is the watch points section worth keeping?"
are all proposals to change the prep. Do not answer them as conversation and stop there — that
leaves the host with advice and no way to act on it. Carry the change through:

- Recommend cuts to a list → return the list AS IT WOULD BE AFTER the cuts, operation
  `delete_content`, `needs_confirmation` true.
- Recommend rewording → return the rewritten list, operation `update_content`,
  `needs_confirmation` true.
- Suggest a section is not worth keeping → put it in `delete_sections`, `needs_confirmation` true.

**The yes/no prompt IS the structured output — a sentence is not one.** The host answers by tapping
a button that only exists if you returned the change. If your reply ends by asking "would you like
me to add these?", "shall I remove it?", "would you like to delete this section, or keep it?", then
that exact change MUST be in this turn's output with `needs_confirmation` true:

- Offering content → the full list in `new_sections`.
- Offering to remove a section → its name in `delete_sections`.

Asking in prose while returning nothing is the worst outcome available: the host reads a question,
answers "yes", and nothing happens. Never write "would you like me to…" without the matching
content in this turn's output.

This is about substance, not phrasing. **If your reply lists content the host could put in their
prep, you have made an offer** — return it, whether you introduced it with "would you like me to
add these?" or "you might also watch for:" or just a bare list. Writing three watch points into
your reply and returning nothing leaves them with no way to keep any of it.

If you truly have nothing to offer — you are recommending they change nothing — say that and stop.
Do not end on an offer you are not returning.

Advice with no pending change is only correct when you are recommending they change nothing — and
then do not end on an offer.

**Answering that prompt.** If your previous turn asked for confirmation and the host now agrees
("yes", "yeah do it", "add them", "sure"), that is an explicit instruction: return the same change
you drafted — it is in `recent_conversation` — with `needs_confirmation` false and the write flags
set. If they decline ("no", "not those"), acknowledge it, change nothing, and set all flags false.

**Pure questions.** When the host only asks something informational ("what's this meeting about?",
"which of these matters most?"), answer it and set both flags false, returning the current list
unchanged.

When in doubt between saving and asking, ask. A wrongly-saved list costs the host an edit; a
confirmation prompt costs them one tap.

The one thing that is NOT doubt is a message that names both an operation and a target. "Delete the
watch points section" is an instruction — carry it out and say you did. Asking "would you like me to
remove it?" after the host already told you to remove it wastes a turn and reads as though you did
not understand them.
<<<REGEN>>>
## Regenerating the prep

Sometimes the host is not asking for anything. They mention what came out of a pre-call, correct a
detail, think out loud, or drop a fact in passing. When what they reveal is substantial and the
prep no longer matches it, one small edit is the wrong response — the right one is to offer to
bring the whole prep back in line. That offer is the `regeneration` field.

Judge it against the prep in front of you, on four questions:

- **Coverage** — is what they just revealed held by any section at all?
- **Contradiction** — does a section assert something this overturns, or rest on a premise that
  has just stopped being true?
- **Materiality** — would the host walk into the meeting with a wrong or missing plan if nothing
  changed?
- **Shape-fit** — is the missing thing a KIND of content this prep has no section for?

Answer Coverage and Contradiction from section CONTENT. `section_inventory` tells you a section
exists and how many rows it has; it cannot tell you whether those rows still hold. A section full
of rows can be contradicted completely — open it in `existing_sections` and read it before
concluding it covers anything.

**Then scope the plan to what the facts actually reach.** A fact that changes the meeting's
PURPOSE, TYPE or AGENDA — it has become a different meeting — overturns the premise of every
section that describes that meeting, not only the questions. The overview or about, the meeting
type, the discussion points, the watch points and the questions all rest on what the meeting IS;
when that moves, each of them is either wrong or written for a meeting that is no longer happening.
Read each one and name every one you would change. A plan that changes only the questions after
the agenda moved leaves the rest of the prep asserting the old agenda, and the next turn reads
that back as the truth.

`people`, `company` and `historical_context` are the exception to all of this and stay out of the
plan whatever the facts are — see below.

Propose nothing when the prep already covers it. Information that changes no content — a time
change, who is dialling in from where — is not grounds however new it is. And one stray fact that a
single section absorbs is an ordinary edit: offer it the normal way, through `new_sections`. A
regeneration is for when the picture moved, not when a line did.

`regeneration` is also the right answer when the host asks for it outright — "redo my prep",
"regenerate this with what I just told you".

**Proposing.** Fill `regeneration` with `new_facts` — what the message revealed that the prep does
not reflect — and one `changes` entry per section you would touch:

- `rewrite` — the section exists and its content no longer holds.
- `extend` — the section exists and its content still holds; you would add to it.
- `add_section` — nothing holds this kind of content. Give the `type` you would build it as.

Name sections as they appear in `section_inventory`. At most 3 `add_section` entries and 6 changes
in total, and only what the new context genuinely warrants. Padding a plan with sections the facts
do not reach is what reads as though you did not look at what is already there — but a fact that
genuinely moved the whole picture warrants a plan that moves with it. Match the plan to the reach
of the facts: neither wider than they justify, nor narrower than they demand.

On a proposing turn `new_sections` and `delete_sections` are EMPTY and `needs_confirmation` is
false. You are not writing anything — the host will be shown a single yes/no prompt. Your reply
names exactly which sections you would change and what you would add. They approve on the strength
of that sentence, so it must match the plan and promise nothing beyond it.

**Never name `people`, `company` or `historical_context` in a plan.** They are immutable here as
everywhere else, and a regeneration does not reach them. No fact widens a plan onto them — not a
changed agenda, not a changed meeting, however completely the picture moved. They are generated
elsewhere and a regeneration is not a way in. Say so if the host expects otherwise.

**A regeneration never deletes a section.** It rewrites, extends and adds. A section that should go
entirely is the host's call to make explicitly, through `delete_sections`.

**Carrying it out.** When `approved_regeneration` is present the host has already said yes, and
this turn does the work:

- Return one entry in `new_sections` for each change in the plan, carrying the FULL list that
  section should end up with. `rewrite` is `update_content`, `extend` is `add_content` and keeps
  every existing row, `add_section` stays `add_section`.
- Touch nothing outside the plan. A section it does not name is not yours this turn.
- Keep each existing section's shape: fill the fields of the type its `section_inventory` row
  gives, not the type you would have chosen for that content.
- `needs_confirmation` is false and `regeneration` is null. The decision is already made — do not
  ask again.
- If the host approved but narrowed it ("yes, but leave my questions alone"), return only the
  subset they kept. Never more than the plan.

Do not offer a regeneration on a turn that is already carrying one out, and do not offer one while
the host is answering a yes/no prompt you raised.
<<</REGEN>>>
## Output (strict JSON)

{
  "reasoning": ["<short finding>", "..."],
  "needs_confirmation": true|false,
<<<REGEN>>>  "regeneration": null | {
    "new_facts": ["<what the message revealed that the prep does not reflect>"],
    "changes": [
      {"section": "<Section Name>", "action": "rewrite|extend|add_section", "type": "", "reason": "<one line>"}
    ]
  },
<<</REGEN>>>  "new_sections": [
    {
      "name": "<Section Name>",
      "type": "numbered_list|flagged_list|key_values|chips|prose|links",
      "operation": "add_section|add_content|update_content|delete_content",
      "items": [
        {"text": "", "label": "", "value": "", "detail": "", "severity": "", "url": ""}
      ]
    }
  ],
  "delete_sections": ["<Section Name>"],
  "assistant_reply": "<natural-language reply to the host>"
}

Answer `needs_confirmation` FIRST, from the host's message alone, before writing any content.<<<REGEN>>>
Then `regeneration`, still before any content: whether the prep as a whole has fallen out of step
is a question about the message and `section_inventory`, not about rows you have not written yet.<<</REGEN>>>
Write `assistant_reply` LAST, after you have decided what this turn does. It must describe exactly
what you returned above it — nothing more. Never wrap the JSON in markdown.
"""

_REGEN_BLOCK = re.compile(r"<<<REGEN>>>(.*?)<<</REGEN>>>", re.DOTALL)


def build_call_prep_system_prompt(*, regeneration_enabled: bool | None = None) -> str:
    """The system prompt, with the regeneration material kept or cut.

    Composed from one template rather than two literals: the feature's rules and its line in the
    output schema have to agree, and two copies of a 300-line prompt drift the first time either
    is edited.
    """
    enabled = (
        live_settings.call_prep_regeneration_enabled
        if regeneration_enabled is None
        else regeneration_enabled
    )
    if enabled:
        return _REGEN_BLOCK.sub(lambda m: m.group(1), _SYSTEM_PROMPT_TEMPLATE)
    return _REGEN_BLOCK.sub("", _SYSTEM_PROMPT_TEMPLATE)


# The full prompt, feature included. Kept under its original name: it is what the prompt tests read.
CALL_PREP_SYSTEM_PROMPT = build_call_prep_system_prompt(regeneration_enabled=True)


def _chat_prompt_payload(
    *,
    meeting_title: str,
    meeting_description: str,
    existing_sections: list[dict[str, Any]],
    existing_discovery_questions: list[str],
    user_message: str,
    recent_conversation: list[dict[str, str]],
    additional_context: str | None,
    host: "HostPersona | None" = None,
    approved_regeneration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "meeting_title": meeting_title,
        "meeting_description": meeting_description,
        "existing_sections": annotate_people_with_host(existing_sections, host),
        "section_inventory": build_section_inventory(existing_sections),
        "existing_discovery_questions": existing_discovery_questions,
        "user_message": user_message,
        "recent_conversation": recent_conversation,
    }
    # Its own key rather than folded into the message: unlike the enterprise service, this call is
    # not limited to one free-text field, so the context can stay structurally separate from what
    # the host said. Omitted when empty, so callers that never send it see identical behaviour.
    # Named ``host`` rather than ``user`` so it cannot be read as the other side of the call: this
    # is the person the questions are written FOR, never the person they are asked TO.
    if host is not None:
        payload["host"] = host.as_prompt_dict()
    context = (additional_context or "").strip()
    if context:
        payload["additional_context"] = context
    # Present only on the turn that carries out an approved plan. Its absence is what tells every
    # other turn that it is an ordinary one.
    if approved_regeneration:
        payload["approved_regeneration"] = approved_regeneration
    return payload


async def run_call_prep_chat(
    *,
    meeting_title: str,
    meeting_description: str,
    existing_sections: list[dict[str, Any]],
    existing_discovery_questions: list[str],
    user_message: str,
    recent_conversation: list[dict[str, str]],
    additional_context: str | None = None,
    host: "HostPersona | None" = None,
    approved_regeneration: dict[str, Any] | None = None,
    max_tokens: int = MAX_TURN_TOKENS,
) -> CallPrepChatTurn:
    """Call OpenAI for non-SOA call-prep chat."""
    api_key = app_settings.openai_api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = AsyncOpenAI(api_key=api_key)
    user_payload = _chat_prompt_payload(
        meeting_title=meeting_title,
        meeting_description=meeting_description,
        existing_sections=existing_sections,
        existing_discovery_questions=existing_discovery_questions,
        user_message=user_message,
        recent_conversation=recent_conversation,
        additional_context=additional_context,
        host=host,
        approved_regeneration=approved_regeneration,
    )
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": build_call_prep_system_prompt()},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format=CallPrepChatTurn,
            temperature=0.35,
            max_tokens=max_tokens,
        )
    except LengthFinishReasonError as exc:
        raise RuntimeError(
            "The assistant's reply ran too long to complete. Nothing was changed."
        ) from exc
    except APIError as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"OpenAI API error (call-prep chat): {exc}") from exc

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed call-prep chat result")
    return parsed


# Roughly the chunk size the enterprise service streams at. Below ~15 there is no real batching;
# above ~50 the text stops feeling typed and starts arriving in visible blocks.
STREAM_CHUNK_CHARS = 28


def _drain_stream_buffer(buffer: str) -> Iterator[str]:
    """Yield the chunks ready to send from ``buffer``, leaving any tail behind.

    Flushes on a newline so markdown lines land whole, and otherwise once the buffer passes
    ``STREAM_CHUNK_CHARS`` — but only at the next whitespace, so a word is never split across two
    events. What remains is held for the next delta, or for the final flush.
    """
    while True:
        newline = buffer.find("\n")
        if newline >= 0:
            chunk, buffer = buffer[: newline + 1], buffer[newline + 1 :]
            yield chunk
            continue
        if len(buffer) < STREAM_CHUNK_CHARS:
            return
        split = buffer.find(" ", STREAM_CHUNK_CHARS)
        if split < 0:
            return  # mid-word; wait for the whitespace rather than cutting the word in half
        chunk, buffer = buffer[:split], buffer[split:]
        yield chunk


def _scan_json_string(text: str, start: int) -> tuple[str, bool]:
    """Decode the JSON string literal beginning at ``text[start]`` (its opening quote).

    Returns the text decoded so far and whether the literal was closed. A half-written literal is
    normal mid-stream, so this reports it rather than failing.
    """
    out: list[str] = []
    escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            if i + 1 >= len(text):
                return "".join(out), False
            esc = text[i + 1]
            if esc == "u":
                if i + 6 > len(text):
                    return "".join(out), False
                try:
                    out.append(chr(int(text[i + 2 : i + 6], 16)))
                except ValueError:
                    pass
                i += 6
            else:
                out.append(escapes.get(esc, esc))
                i += 2
        elif ch == '"':
            return "".join(out), True
        else:
            out.append(ch)
            i += 1
    return "".join(out), False


def _partial_string_array(snapshot: str, key: str, *, after: int = 0) -> list[str]:
    """Completed elements of a string array in a partially-written JSON document."""
    marker = snapshot.find(f'"{key}"', after)
    if marker < 0:
        return []
    open_bracket = snapshot.find("[", marker)
    if open_bracket < 0:
        return []
    out: list[str] = []
    i = open_bracket + 1
    while i < len(snapshot):
        ch = snapshot[i]
        if ch == "]":
            break
        if ch == '"':
            value, closed = _scan_json_string(snapshot, i)
            if not closed:
                break  # still being written; not safe to emit yet
            out.append(value)
            i += len(json.dumps(value))
        else:
            i += 1
    return out


def _partial_string_field(snapshot: str, key: str, *, after: int = 0) -> str:
    """The text of a string field written so far, empty if it has not started."""
    marker = snapshot.find(f'"{key}"', after)
    if marker < 0:
        return ""
    colon = snapshot.find(":", marker)
    if colon < 0:
        return ""
    quote = snapshot.find('"', colon)
    if quote < 0:
        return ""
    value, _closed = _scan_json_string(snapshot, quote)
    return value


async def stream_call_prep_chat(
    *,
    meeting_title: str,
    meeting_description: str,
    existing_sections: list[dict[str, Any]],
    existing_discovery_questions: list[str],
    user_message: str,
    recent_conversation: list[dict[str, str]],
    additional_context: str | None = None,
    host: "HostPersona | None" = None,
    # Set only when the host has approved a regeneration: the turn then rewrites several sections
    # at once, which needs both the plan in front of it and more room than an ordinary reply.
    approved_regeneration: dict[str, Any] | None = None,
    max_tokens: int = MAX_TURN_TOKENS,
) -> AsyncIterator[tuple[str, Any]]:
    """Stream the non-SOA turn, yielding ``("event", {...})`` then one ``("final", turn)``.

    Structured output and streaming are not in tension, but the SDK only fills ``event.parsed`` at
    ``content.done`` — on the deltas it is ``None``. What each delta does carry is ``snapshot``,
    the raw JSON written so far, so the fields are read out of that as they appear.

    ``reasoning`` is declared before ``assistant_reply`` in the schema and is therefore generated
    first, which is what lets the working reach the host ahead of the reply. Only ``("final", …)``
    is a validated turn, and nothing is written to the host's prep before it arrives.
    """
    api_key = app_settings.openai_api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = AsyncOpenAI(api_key=api_key)
    user_payload = _chat_prompt_payload(
        meeting_title=meeting_title,
        meeting_description=meeting_description,
        existing_sections=existing_sections,
        existing_discovery_questions=existing_discovery_questions,
        user_message=user_message,
        recent_conversation=recent_conversation,
        additional_context=additional_context,
        host=host,
        approved_regeneration=approved_regeneration,
    )

    sent_reasoning = 0
    sent_reply_chars = 0
    # Deltas arrive one token at a time (1-6 chars). Emitting each as its own event is real
    # streaming but needlessly chatty — ~200 frames for a short reply. Batching to roughly this
    # many characters keeps every flush mid-generation, so the timing is unchanged, while landing
    # near the chunk size the enterprise service uses.
    buffer = ""

    def _pending_reasoning(items: list[Any], upto: int) -> Iterator[tuple[int, str]]:
        nonlocal sent_reasoning
        while sent_reasoning < upto:
            text = str(items[sent_reasoning] or "").strip()
            index = sent_reasoning
            sent_reasoning += 1
            if text:
                yield index, text

    try:
        async with client.beta.chat.completions.stream(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": build_call_prep_system_prompt()},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format=CallPrepChatTurn,
            temperature=0.35,
            max_tokens=max_tokens,
        ) as stream:
            async for event in stream:
                if event.type != "content.delta":
                    continue
                snapshot = event.snapshot or ""

                # Only closed array elements come back, so every entry here is finished writing.
                reasoning = _partial_string_array(snapshot, "reasoning")
                for _, text in _pending_reasoning(reasoning, len(reasoning)):
                    yield "event", {"type": "reasoning", "text": text}

                reply = _partial_string_field(snapshot, "assistant_reply")
                if len(reply) > sent_reply_chars:
                    buffer += reply[sent_reply_chars:]
                    sent_reply_chars = len(reply)
                    for chunk in _drain_stream_buffer(buffer):
                        buffer = buffer[len(chunk) :]
                        yield "event", {"type": "message", "text": chunk}

            completion = await stream.get_final_completion()
    except LengthFinishReasonError as exc:
        # Hit the cap mid-turn. Nothing partial is safe to write — the sections may be half
        # written — so the turn is abandoned rather than half applied.
        raise RuntimeError(
            "The assistant's reply ran too long to complete. Nothing was changed."
        ) from exc
    except APIError as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"OpenAI API error (call-prep chat): {exc}") from exc

    turn = completion.choices[0].message.parsed
    if turn is None:
        raise RuntimeError("OpenAI returned no parsed call-prep chat result")

    # Anything the deltas did not cover — a short reply that never grew, or trailing reasoning.
    for _, text in _pending_reasoning(list(turn.reasoning or []), len(turn.reasoning or [])):
        yield "event", {"type": "reasoning", "text": text}
    tail = buffer + turn.assistant_reply[sent_reply_chars:]
    if tail:
        # A reply shorter than the flush threshold would otherwise never be sent.
        yield "event", {"type": "message", "text": tail}

    yield "final", turn


def chunk_message_text(text: str, *, chunk_size: int = 48) -> list[str]:
    """Split assistant text into small chunks for streaming message events."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    buf = ""
    for word in cleaned.split(" "):
        piece = word if not buf else f" {word}"
        if buf and len(buf) + len(piece) > chunk_size:
            chunks.append(buf)
            buf = word
        else:
            buf += piece
    if buf:
        chunks.append(buf)
    return chunks


async def stream_ndjson_events(events: list[dict[str, Any]]) -> AsyncIterator[str]:
    for event in events:
        yield json.dumps(event, ensure_ascii=False, default=str) + "\n"

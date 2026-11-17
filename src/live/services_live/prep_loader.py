"""Call-prep loader seam.

A session coaches several users on one calendar event, and each carries their own SOA status:

- ``soa_connected=true``  — resolve via meetings → meeting_call_prep (flat payload).
- ``soa_connected=false`` — resolve via temporary_meeting_call_prep by external_event_id
  (sections-shaped payload).

Every lookup is scoped by ``(external_event_id, user_id)``. A single calendar event produces one
``meetings`` row per participant (unique on ``user_id, external_event_id``), so a lookup keyed on
``external_event_id`` alone resolves to an arbitrary participant's prep. A missing or unusable
``user_id`` yields no prep rather than a guess.

Loading is **batched**: at most two queries run per session — one for the SOA-connected users and
one for the rest — regardless of how many users are on the roster, so the WS handshake does not
slow down as the meeting grows.

Discovery questions become tracked prep questions; remaining flat fields become agent context.
Missing rows / empty questions are non-fatal (that user simply gets no prep).
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.live.services_live.events import PrepQuestion

logger = logging.getLogger(__name__)

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TEMPORARY_CALL_PREP_TABLE = "temporary_meeting_call_prep"
DISCOVERY_SECTION_ID = "sec_discovery_questions"
DISCOVERY_SECTION_KEY = "discovery_questions"


@dataclass(frozen=True)
class PrepUser:
    """One user to load prep for, with the SOA status that picks which table to read."""

    user_id: str
    soa_connected: bool


@dataclass
class PrepData:
    questions: list[PrepQuestion] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


def _parse_user_id(user_id: str | int | None) -> int | None:
    """Coerce the wire ``userId`` to ``users.id``; ``None`` when absent or non-numeric."""
    if user_id is None:
        return None
    if isinstance(user_id, int):
        return user_id
    text = user_id.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _questions_from_texts(raw_questions: Any) -> list[PrepQuestion]:
    if not isinstance(raw_questions, list):
        return []
    questions: list[PrepQuestion] = []
    for item in raw_questions:
        text = str(item).strip()
        if text:
            questions.append(PrepQuestion(id=f"q{len(questions) + 1}", text=text))
    return questions


def _discovery_texts_from_sections(payload: dict[str, Any]) -> list[str]:
    sections = payload.get("sections")
    if not isinstance(sections, list):
        return []
    out: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        if (
            section.get("key") != DISCOVERY_SECTION_KEY
            and section.get("id") != DISCOVERY_SECTION_ID
        ):
            continue
        block = section.get("block") or {}
        if not isinstance(block, dict):
            continue
        data = block.get("data") or {}
        if not isinstance(data, dict):
            continue
        items = data.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
            else:
                text = str(item).strip()
            if text:
                out.append(text)
        break
    return out


def parse_prep_payload(payload: dict[str, Any] | None) -> PrepData:
    """Extract prep questions + context from a call-prep ``payload`` jsonb object.

    Supports flat ``discovery_questions`` and sections-shaped temporary prep payloads.
    """
    if not payload:
        return PrepData()

    if isinstance(payload.get("sections"), list):
        questions = _questions_from_texts(_discovery_texts_from_sections(payload))
        return PrepData(questions=questions, context={})

    raw_questions = payload.get("discovery_questions") or []
    questions = _questions_from_texts(raw_questions)
    context = {k: v for k, v in payload.items() if k != "discovery_questions"}
    return PrepData(questions=questions, context=context)


def _coerce_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return None
    return raw if isinstance(raw, dict) else None


class PrepLoader(ABC):
    @abstractmethod
    async def load_for_users(
        self,
        external_event_id: str,
        users: Sequence[PrepUser],
    ) -> dict[str, PrepData]:
        """Return ``user_id -> PrepData`` for the users that have prep.

        Users with no prep row (or an unusable id) are simply absent from the result; this is a
        normal outcome, not an error.
        """


class StaticPrepLoader(PrepLoader):
    """In-memory loader mapping ``(external_event_id, user_id) -> raw payload`` (tests / local dev).

    Also accepts a plain ``external_event_id -> payload`` map, in which case that payload is
    returned for every user on the roster — convenient for single-user fixtures.
    """

    def __init__(
        self,
        payloads_by_event_id: dict[str, dict[str, Any]] | None = None,
        *,
        payloads_by_event_and_user: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self._payloads = payloads_by_event_id or {}
        self._per_user = payloads_by_event_and_user or {}

    async def load_for_users(
        self,
        external_event_id: str,
        users: Sequence[PrepUser],
    ) -> dict[str, PrepData]:
        out: dict[str, PrepData] = {}
        for user in users:
            payload = self._per_user.get((external_event_id, user.user_id))
            if payload is None:
                payload = self._payloads.get(external_event_id)
            if payload is None:
                continue
            out[user.user_id] = parse_prep_payload(payload)
        return out


class PostgresPrepLoader(PrepLoader):
    """Loads call-prep from meeting_call_prep or temporary_meeting_call_prep."""

    def __init__(
        self,
        *,
        meetings_table: str,
        call_prep_table: str,
        db_settings,
        temporary_call_prep_table: str = TEMPORARY_CALL_PREP_TABLE,
    ) -> None:
        for name, label in (
            (meetings_table, "meetings"),
            (call_prep_table, "call_prep"),
            (temporary_call_prep_table, "temporary_call_prep"),
        ):
            if not _TABLE_NAME_RE.match(name):
                raise ValueError(f"unsafe {label} table name: {name!r}")
        self._meetings_table = meetings_table
        self._call_prep_table = call_prep_table
        self._temporary_call_prep_table = temporary_call_prep_table
        self._db = db_settings
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                host=self._db.db_host,
                port=self._db.db_port,
                user=self._db.db_user,
                password=self._db.db_password,
                database=self._db.db_name,
                min_size=1,
                max_size=5,
            )
        return self._pool

    async def load_for_users(
        self,
        external_event_id: str,
        users: Sequence[PrepUser],
    ) -> dict[str, PrepData]:
        event_id = (external_event_id or "").strip()
        if not event_id or not users:
            return {}

        connected: dict[int, str] = {}
        temporary: dict[int, str] = {}
        for user in users:
            parsed = _parse_user_id(user.user_id)
            if parsed is None:
                logger.info(
                    "[live] prep skipped invalid user_id=%r external_event_id=%r",
                    user.user_id,
                    event_id,
                )
                continue
            target = connected if user.soa_connected else temporary
            target[parsed] = user.user_id

        out: dict[str, PrepData] = {}
        try:
            if connected:
                out.update(await self._load_connected(event_id, connected))
            if temporary:
                out.update(await self._load_temporary(event_id, temporary))
        except Exception:  # noqa: BLE001
            logger.exception(
                "[live] prep load error external_event_id=%r users=%s",
                event_id,
                [u.user_id for u in users],
            )
            return out
        return out

    async def _load_connected(
        self,
        event_id: str,
        users_by_db_id: dict[int, str],
    ) -> dict[str, PrepData]:
        """One query for every SOA-connected user on the roster.

        ``meetings`` is unique on (user_id, external_event_id) and ``meeting_call_prep`` is unique
        on meeting_id, so this returns at most one row per user.
        """
        pool = await self._get_pool()
        query = f"""
            SELECT m.user_id AS user_id, mcp.payload AS payload
            FROM {self._call_prep_table} AS mcp
            JOIN {self._meetings_table} AS m
              ON mcp.meeting_id = m.id
            WHERE m.external_event_id = $1
              AND m.user_id = ANY($2::int[])
        """
        rows = await pool.fetch(query, event_id, list(users_by_db_id))
        return self._collect(rows, users_by_db_id, event_id, source="call_prep")

    async def _load_temporary(
        self,
        event_id: str,
        users_by_db_id: dict[int, str],
    ) -> dict[str, PrepData]:
        """One query for every non-SOA user on the roster.

        ``temporary_meeting_call_prep`` is unique on (user_id, external_event_id), so this returns
        at most one row per user.
        """
        pool = await self._get_pool()
        query = f"""
            SELECT user_id, payload
            FROM {self._temporary_call_prep_table}
            WHERE external_event_id = $1
              AND user_id = ANY($2::int[])
        """
        rows = await pool.fetch(query, event_id, list(users_by_db_id))
        return self._collect(rows, users_by_db_id, event_id, source="temporary_call_prep")

    def _collect(
        self,
        rows: Iterable[Any],
        users_by_db_id: dict[int, str],
        event_id: str,
        *,
        source: str,
    ) -> dict[str, PrepData]:
        out: dict[str, PrepData] = {}
        for row in rows or []:
            db_user_id = row["user_id"]
            wire_user_id = users_by_db_id.get(db_user_id)
            if wire_user_id is None:
                continue
            payload = _coerce_payload(row["payload"])
            prep = parse_prep_payload(payload)
            out[wire_user_id] = prep
            logger.info(
                "[live] prep hit source=%s external_event_id=%r user_id=%s questions=%s",
                source,
                event_id,
                wire_user_id,
                len(prep.questions),
            )
        missing = [uid for uid in users_by_db_id.values() if uid not in out]
        if missing:
            logger.info(
                "[live] no prep source=%s external_event_id=%r user_ids=%s",
                source,
                event_id,
                missing,
            )
        return out

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

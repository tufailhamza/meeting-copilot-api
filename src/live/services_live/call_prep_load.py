"""Pre-session call-prep load: gate on ``non_soa_meetings``, then build meeting context from ES.

The context is assembled **without an LLM** — fixed keyword phrasing, hard caps on every list, and
sentence-aware trimming. ES documents are enormous (a company doc carries ~90 ``cb_*`` fields, a
person doc every skill they ever listed); the caps here are what keep the result to something a
prompt can actually carry.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

import json as _json

import asyncpg
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel

from src.config import settings as app_settings
from src.live.config_live import live_db_settings
from src.live.services_live import es_client

logger = logging.getLogger(__name__)

# This feature owns its pool. Sharing the chat endpoint's would couple the two: its pool size,
# lifetime and failure modes would become this endpoint's, and the advisory lock below holds a
# connection for the length of an ES round trip.
_POOL: asyncpg.Pool | None = None


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


NON_SOA_MEETINGS_TABLE = "non_soa_meetings"
PEOPLE_INDEX = "people"
COMPANY_INDEX = "company"
COMPANY_NAME_FIELD = "li_universalname"

# Caps. Past these, extra rows stop adding signal and start crowding the prompt.
MAX_EXPERIENCES = 3
MAX_SKILLS = 5
MAX_EDUCATION = 1
MAX_INDUSTRIES = 3
MAX_SPECIALTIES = 4
MAX_DESCRIPTION_CHARS = 320

# A waiter that cannot get the advisory lock in this long gives up. The holder runs an LLM call
# inside the lock, so this has to outlast that, not just the DB work.
LOCK_TIMEOUT = "180s"

_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
# ES stores "no headline" as a literal double dash rather than null.
_PLACEHOLDER_TEXT = {"", "-", "--", "n/a", "none", "null"}


# --------------------------------------------------------------------------------------
# small text helpers
# --------------------------------------------------------------------------------------


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _PLACEHOLDER_TEXT else text


def _format_month(raw: Any) -> str:
    """``2007-02-01T00:00:00`` -> ``Feb 2007``. Empty string when unparseable."""
    text = _clean(raw)[:10]
    parts = text.split("-")
    if len(parts) < 2:
        return ""
    try:
        year, month = int(parts[0]), int(parts[1])
    except ValueError:
        return ""
    if not 1 <= month <= 12:
        return str(year)
    return f"{_MONTHS[month - 1]} {year}"


def _date_range(start: Any, end: Any) -> str:
    """``(Feb 2007 – May 2016)``; an absent end means the role is current."""
    start_text, end_text = _format_month(start), _format_month(end)
    if not start_text and not end_text:
        return ""
    if not start_text:
        return f"(until {end_text})"
    return f"({start_text} – {end_text or 'present'})"


def _trim_sentences(text: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """Cut to ``limit`` on a sentence boundary when there is one, else on a word boundary."""
    flat = " ".join(_clean(text).split())
    if len(flat) <= limit:
        return flat
    window = flat[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut > limit // 3:
        return window[: cut + 1]
    cut = window.rfind(" ")
    return (window[:cut] if cut > 0 else window).rstrip(" ,;:-") + "…"


def _as_list(value: Any) -> list[str]:
    """ES is inconsistent here: industries arrive as a list, cb_industries as a CSV string."""
    if isinstance(value, list):
        return [_clean(v) for v in value if _clean(v)]
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


# --------------------------------------------------------------------------------------
# participants
# --------------------------------------------------------------------------------------


@dataclass
class MeetingContext:
    """The rendered context block plus the ES documents it was built from.

    The documents are carried out rather than discarded so the people and company sections can be
    built from the same fetch — a second lookup would cost another round trip for data we already
    have in hand.
    """

    text: str
    people_by_es_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    companies: list[dict[str, Any]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text)


@dataclass(frozen=True)
class Participant:
    """One person on the meeting. ``user_id`` is null for external participants."""

    email: str
    user_id: str | None
    es_id: str | None
    name: str = ""
    # SOA-connected users get their prep from the enterprise side, not from here. They still
    # count as participants for context and for the past-meeting overlap filter.
    soa_connected: bool = False


def build_participants(users: Iterable[Any], external: Iterable[Any]) -> list[Participant]:
    """Internal users carry an id; external participants are email-only."""
    out = [
        Participant(
            email=_clean(getattr(u, "email", "")),
            user_id=_clean(getattr(u, "user_id", "")) or None,
            es_id=_clean(getattr(u, "es_id", "")) or None,
            name=_clean(getattr(u, "name", "")),
            soa_connected=bool(getattr(u, "soa_connected", False)),
        )
        for u in users
    ]
    out.extend(
        Participant(
            email=_clean(getattr(p, "email", "")),
            user_id=None,
            es_id=_clean(getattr(p, "es_id", "")) or None,
        )
        for p in external
    )
    return out


def eligible_users(participants: Iterable[Participant]) -> list[Participant]:
    """Users this endpoint generates prep for: internal, and not SOA-connected."""
    return [p for p in participants if p.user_id and not p.soa_connected]


def participant_metadata(participants: Iterable[Participant]) -> list[dict[str, Any]]:
    """The jsonb shape stored in ``non_soa_meetings.participant_user_metadata``."""
    return [{"email": p.email, "user_id": p.user_id} for p in participants]


# --------------------------------------------------------------------------------------
# DB gate
# --------------------------------------------------------------------------------------


@asynccontextmanager
async def meeting_load_guard(external_event_id: str) -> AsyncIterator[Any]:
    """Serialize loads for one event across processes.

    A transaction-scoped advisory lock, held for the whole load rather than just the flag read:
    releasing it early would let two concurrent callers both see ``initial_call_prep_loaded =
    false`` and both run the enrichment. It self-releases on commit, rollback, or a dropped
    connection, so there is no stale lock to recover from.
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1))", external_event_id
            )
            yield conn


async def gate_and_upsert_meeting(
    conn: Any,
    *,
    external_event_id: str,
    meeting_title: str,
    meeting_description: str,
    participants: list[Participant],
) -> bool:
    """Create the meetings row if absent. ``True`` when the caller should build context.

    ``False`` means prep was already loaded for this event and there is nothing left to do.
    """
    row = await conn.fetchrow(
        f"SELECT initial_call_prep_loaded FROM {NON_SOA_MEETINGS_TABLE} "
        "WHERE external_event_id = $1",
        external_event_id,
    )
    if row is not None:
        return not row["initial_call_prep_loaded"]

    await conn.execute(
        f"""
        INSERT INTO {NON_SOA_MEETINGS_TABLE}
            (external_event_id, participant_user_metadata, meeting_title,
             meeting_description, summary, initial_call_prep_loaded)
        VALUES ($1, $2::jsonb, $3, $4, '', false)
        ON CONFLICT (external_event_id) DO NOTHING
        """,
        external_event_id,
        _json.dumps(participant_metadata(participants)),
        meeting_title or "",
        meeting_description or "",
    )
    return True


async def mark_initial_prep_loaded(conn: Any, external_event_id: str) -> None:
    await conn.execute(
        f"UPDATE {NON_SOA_MEETINGS_TABLE} SET initial_call_prep_loaded = true "
        "WHERE external_event_id = $1",
        external_event_id,
    )


# --------------------------------------------------------------------------------------
# person summarization
# --------------------------------------------------------------------------------------


def _sorted_experiences(raw: Any) -> list[dict[str, Any]]:
    """Most recent first. Current roles (no end date) outrank ended ones."""
    entries = [e for e in (raw or []) if isinstance(e, dict)]

    def key(exp: dict[str, Any]) -> tuple[int, str, str]:
        end = _clean(exp.get("end"))
        return (1 if not end else 0, end, _clean(exp.get("start")))

    return sorted(entries, key=key, reverse=True)[:MAX_EXPERIENCES]


def summarize_person(source: dict[str, Any], *, fallback_name: str = "") -> str:
    """Compact prose for one person: who they are, last roles, top skills, top degree."""
    name = _clean(source.get("full_name")) or fallback_name or "Unknown participant"
    headline = _clean(source.get("headline"))
    location = _clean(source.get("location"))

    opener = name
    if headline:
        opener += f" — {headline}"
    if location:
        opener += f", based in {location}"
    lines = [opener + "."]

    for exp in _sorted_experiences(source.get("experience")):
        title = _clean(exp.get("title")) or "Unknown role"
        company = _clean(exp.get("company"))
        span = _date_range(exp.get("start"), exp.get("end"))
        sentence = f"{title} at {company}" if company else title
        if span:
            sentence += f" {span}"
        industries = _as_list(exp.get("company_industry"))[:1]
        if industries:
            sentence += f" [{industries[0]}]"
        lines.append(f"- {sentence}.")

    skills = _as_list(source.get("skills"))[:MAX_SKILLS]
    if skills:
        lines.append(f"- Skills: {', '.join(skills)}.")

    for edu in [e for e in (source.get("education") or []) if isinstance(e, dict)][:MAX_EDUCATION]:
        degree, major, school = (
            _clean(edu.get("degree")),
            _clean(edu.get("major")),
            _clean(edu.get("school")),
        )
        qualification = " in ".join(p for p in (degree, major) if p)
        if qualification and school:
            lines.append(f"- Education: {qualification}, {school}.")
        elif school:
            lines.append(f"- Education: {school}.")

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# company summarization
# --------------------------------------------------------------------------------------


def _staff_size(source: dict[str, Any]) -> str:
    headcount = _clean(source.get("li_staffcount")) or _clean(source.get("cb_number_of_employees"))
    if headcount:
        return headcount
    rng = source.get("li_staffcountrange")
    if isinstance(rng, dict):
        start, end = rng.get("start"), rng.get("end")
        if start is not None and end:
            return f"{start}-{end}"
    return ""


def _headquarters(source: dict[str, Any]) -> str:
    hq = source.get("li_headquarter")
    if not isinstance(hq, dict):
        return ""
    parts = [_clean(hq.get(k)) for k in ("city", "geographicArea", "country")]
    return ", ".join(p for p in parts if p)


def summarize_company(source: dict[str, Any], *, fallback_name: str = "") -> str:
    """Four or five lines out of a ~90-field company document."""
    name = (
        _clean(source.get("li_name"))
        or _clean(source.get("cb_organization_name"))
        or fallback_name
        or "Unknown company"
    )
    lines = [f"{name}."]

    description = _clean(source.get("li_description")) or _clean(source.get("cb_full_description"))
    if description:
        lines.append(f"- About: {_trim_sentences(description)}")

    industries = (
        _as_list(source.get("li_industries"))
        or _as_list(source.get("cb_industries"))
    )[:MAX_INDUSTRIES]
    if industries:
        lines.append(f"- Industry: {', '.join(industries)}.")

    facts = []
    size = _staff_size(source)
    if size:
        facts.append(f"{size} employees")
    hq = _headquarters(source)
    if hq:
        facts.append(f"HQ {hq}")
    founded = _format_month(source.get("cb_founded_date"))
    if founded:
        facts.append(f"founded {founded}")
    if facts:
        lines.append(f"- {'; '.join(facts)}.")

    funding_status = _clean(source.get("cb_funding_status"))
    last_round = _clean(source.get("cb_last_funding_type"))
    last_date = _format_month(source.get("cb_last_funding_date"))
    if funding_status or last_round:
        detail = funding_status or last_round
        if last_round and last_date:
            detail += f", last round {last_round} {last_date}"
        lines.append(f"- Funding: {detail}.")

    specialties = _as_list(source.get("li_specialties"))[:MAX_SPECIALTIES]
    if specialties and len(lines) < 6:
        lines.append(f"- Specialties: {', '.join(specialties)}.")

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# context assembly
# --------------------------------------------------------------------------------------


async def build_meeting_context(
    *,
    meeting_title: str,
    meeting_description: str,
    participants: list[Participant],
    companies: list[str],
    trace: Any = None,
) -> MeetingContext:
    """Assemble the meeting context block.

    An empty ``text`` means Elasticsearch was unreachable — the caller leaves
    ``initial_call_prep_loaded`` false so the next call retries.
    """
    by_es_id = {p.es_id: p for p in participants if p.es_id}

    try:
        people_docs = await es_client.mget(PEOPLE_INDEX, list(by_es_id))
        company_docs = await es_client.search_by_terms(
            COMPANY_INDEX, COMPANY_NAME_FIELD, companies
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[live] call-prep load: ES lookup failed, context left empty")
        if trace:
            trace.step("es.failed", error=f"{type(exc).__name__}: {exc}")
        return MeetingContext(text="")

    if trace:
        trace.step(
            "es.lookup",
            people_requested=sorted(by_es_id),
            people_found=sorted(people_docs),
            companies_requested=companies,
            companies_found=[
                d.get("li_name") or d.get("cb_organization_name") for d in company_docs
            ],
        )

    unresolved_es_ids = [es_id for es_id in by_es_id if es_id not in people_docs]
    if unresolved_es_ids:
        logger.warning(
            "[live] call-prep load: es_id not found in %s index, participant has no profile: %s",
            PEOPLE_INDEX,
            ", ".join(
                f"{es_id} ({by_es_id[es_id].email or by_es_id[es_id].user_id})"
                for es_id in unresolved_es_ids
            ),
        )

    blocks: list[str] = []
    header = _clean(meeting_title)
    if header:
        blocks.append(f"MEETING: {header}")
    description = _clean(meeting_description)
    if description:
        blocks.append(f"CONTEXT: {_trim_sentences(description, 500)}")

    people_lines = [
        summarize_person(source, fallback_name=by_es_id[es_id].name)
        for es_id, source in people_docs.items()
        if es_id in by_es_id
    ]
    if people_lines:
        blocks.append("PARTICIPANTS:\n" + "\n\n".join(people_lines))

    # Meeting-level, not attached to any participant.
    company_lines = [summarize_company(source) for source in company_docs]
    if company_lines:
        blocks.append("COMPANIES:\n" + "\n\n".join(company_lines))

    unresolved = [p.email for p in participants if not p.es_id and p.email]
    if unresolved:
        blocks.append("PARTICIPANTS WITH NO PROFILE: " + ", ".join(unresolved))

    return MeetingContext(
        text="\n\n".join(blocks),
        people_by_es_id=people_docs,
        companies=company_docs,
    )


# --------------------------------------------------------------------------------------
# discovery question generation
# --------------------------------------------------------------------------------------

TEMPORARY_CALL_PREP_TABLE = "temporary_meeting_call_prep"
DISCOVERY_SECTION_ID = "sec_discovery_questions"
DISCOVERY_SECTION_KEY = "discovery_questions"
DISCOVERY_SECTION_NAME = "Questions to Ask"
HISTORICAL_SECTION_ID = "sec_historical_context"
HISTORICAL_SECTION_KEY = "historical_context"
HISTORICAL_SECTION_NAME = "Historical Context"
COMPANY_SECTION_ID = "sec_company"
COMPANY_SECTION_KEY = "company"
COMPANY_SECTION_NAME = "Company"
PEOPLE_SECTION_ID = "sec_people"
PEOPLE_SECTION_KEY = "people"
PEOPLE_SECTION_NAME = "Who's on the Call"
DISCOVERY_MODEL = "gpt-4.1"
MIN_QUESTIONS = 3
MAX_QUESTIONS = 5

DISCOVERY_SYSTEM_PROMPT = f"""\
You prepare people for a meeting they are about to walk into.

There are two sides in the input, and confusing them ruins the output:

- `users` — OUR side. Each one is a person you are writing questions FOR. They will ASK.
- `external_participants` — THE OTHER side. They will BE ASKED. Never write questions for them.

Write {MIN_QUESTIONS}–{MAX_QUESTIONS} discovery questions per user: questions that user should ask \
the external participants during this specific meeting.

A discovery question earns its place by surfacing something the user does not already know and \
needs in order to move the conversation forward — budget, timeline, decision process, current \
tooling, constraints, success criteria, who else is involved.

Reject anything that is:
- Small talk ("How are you?", "How was your weekend?").
- Answerable from the context you were given — if the brief says they use Greenhouse, do not ask \
whether they use an ATS.
- Generic filler that would fit any meeting ("What does your company do?").
- Closed yes/no, unless the yes/no genuinely decides the next step.

Ground every question in `meeting_title`, `meeting_description`, and `meeting_context`, in that \
order of authority. The context block carries the participants' backgrounds and their companies — \
use it to be specific, and never contradict it or invent facts it does not support.

What the host wrote in `meeting_description` is the single strongest signal you have about what \
this meeting is for. Nothing else — not a past meeting, not a participant's background — overrides \
it.

The users are on the same side of the same meeting, so their questions will be similar. Tailor \
lightly where a user's own background makes a different angle natural; do not manufacture \
artificial differences.

## Per-user history — STRICTLY PRIVATE

A user object may carry `historical_context`: a brief on what THAT user discussed in THEIR OWN
past meetings.

**It is supporting material, not the subject.** `meeting_title` and `meeting_description` decide
what this meeting is about; history only sharpens questions that already sit inside that scope. Its
proper use is negative and narrow: it tells you what NOT to ask because it was already covered, and
it lets you name a specific unresolved thing when that thing is part of what this meeting is for.

Two hard limits:

- **The description wins.** When the host has written a description, it defines the scope
  completely. A past meeting that points somewhere the description does not go is not a reason to
  ask about that somewhere — drop it. If the history and the description disagree about what this
  meeting is, the description is right.
- **At most one or two questions may come from history.** The rest come from the title, the
  description and `meeting_context`. A history entry that drives most of the set has taken over the
  meeting, and the host will read questions about something they did not ask to discuss.

The one exception: when there is no description and the title says little, history becomes your
main grounding — that is what it is there for.

It is private to the user it sits on. Absolute rules:

- Use a user's `historical_context` ONLY when writing that same user's questions.
- Never let one user's history influence another user's questions — not its facts, not its topics,
  not its wording, not the angle it suggests.
- Two users on this call attended different meetings. Surfacing one person's history to another is
  a confidentiality breach, and it is not excused by the topics looking similar.
- A user with no `historical_context` gets questions written as though no history exists anywhere
  in this input. Do not let a colleague's history fill that gap.

Treat each user's question set as if you had written it in isolation, seeing only that user's own
history.

Return one entry per user in `users`, echoing that user's `user_id` exactly as given.
Each question is a single sentence, open-ended, and professional.
"""


class UserDiscoveryQuestions(BaseModel):
    user_id: str
    questions: list[str]


class DiscoveryQuestionsResult(BaseModel):
    users: list[UserDiscoveryQuestions]


def _clamp_questions(raw: list[str]) -> list[str]:
    """Trim, drop blanks and near-duplicates, cap at MAX_QUESTIONS."""
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        text = _clean(item)
        if not text:
            continue
        key = "".join(ch for ch in text.lower() if ch.isalnum())
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= MAX_QUESTIONS:
            break
    return out


def _history_as_text(items: list[dict[str, Any]]) -> str:
    """Flatten one user's timeline for the discovery prompt."""
    lines = []
    for item in items:
        head = " — ".join(p for p in (item.get("date"), item.get("title")) if p)
        detail = item.get("detail") or ""
        lines.append(f"{head}: {detail}" if head else detail)
    return "\n".join(line for line in lines if line.strip())


def _user_prompt_entry(
    participant: Participant,
    history_by_user: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """One user as the prompt sees them, carrying only their own history."""
    entry: dict[str, Any] = {
        "user_id": participant.user_id,
        "name": participant.name,
        "email": participant.email,
    }
    own_history = history_by_user.get(participant.user_id or "")
    if own_history:
        entry["historical_context"] = _history_as_text(own_history)
    return entry


async def generate_discovery_questions(
    *,
    meeting_title: str,
    meeting_description: str,
    meeting_context: str,
    participants: list[Participant],
    history_by_user: dict[str, list[dict[str, Any]]] | None = None,
    trace: Any = None,
) -> dict[str, list[str]]:
    """Questions keyed by wire ``user_id``, for internal users only.

    One call for the whole roster rather than one per user: the model sees every user at once, so
    the sets come out consistent with each other instead of independently drifting.
    """
    users = eligible_users(participants)
    if not users:
        return {}

    api_key = app_settings.openai_api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = {
        "meeting_title": meeting_title,
        "meeting_description": meeting_description,
        "meeting_context": meeting_context,
        "users": [_user_prompt_entry(p, history_by_user or {}) for p in users],
        "external_participants": [
            {"email": p.email} for p in participants if not p.user_id
        ],
        # On our side of the table but not being prepared here — context, not recipients.
        "colleagues_present": [
            p.email for p in participants if p.user_id and p.soa_connected
        ],
    }

    client = AsyncOpenAI(api_key=api_key)
    try:
        response = await client.beta.chat.completions.parse(
            model=DISCOVERY_MODEL,
            messages=[
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": _json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=DiscoveryQuestionsResult,
            temperature=0.4,
        )
    except APIError as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"OpenAI API error (discovery questions): {exc}") from exc

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed discovery questions")

    wanted = {p.user_id for p in users}
    out: dict[str, list[str]] = {}
    for entry in parsed.users:
        user_id = _clean(entry.user_id)
        # A user_id the model invented has no row to write to, so it is dropped rather than
        # guessed at.
        if user_id not in wanted:
            logger.warning("[live] discovery questions: unknown user_id %r dropped", user_id)
            continue
        questions = _clamp_questions(entry.questions)
        if questions:
            out[user_id] = questions

    if trace:
        trace.step(
            "discovery.llm",
            model=DISCOVERY_MODEL,
            prompt_payload=payload,
            raw_output=parsed.model_dump(),
            questions_after_validation=out,
        )

    missing = wanted - set(out)
    if missing:
        logger.warning("[live] discovery questions: no questions for user_ids %s", sorted(missing))
    return out


def with_empty_for_missing(
    questions_by_user: dict[str, list[str]],
    participants: list[Participant],
) -> dict[str, list[str]]:
    """Give every internal user an entry, empty-list for the ones the model produced nothing for.

    A user with no questions still gets a row: the attempt happened, and an empty section is an
    honest "we tried and found nothing" that the user can then fill in via chat.
    """
    out = dict(questions_by_user)
    for participant in eligible_users(participants):
        if participant.user_id not in out:
            out[participant.user_id] = []
    return out


def build_discovery_section(questions: list[str]) -> dict[str, Any]:
    return {
        "id": DISCOVERY_SECTION_ID,
        "key": DISCOVERY_SECTION_KEY,
        "block": {
            "data": {"items": [{"text": q} for q in questions]},
            "name": DISCOVERY_SECTION_NAME,
            "type": "numbered_list",
        },
        "origin": "ai_generated",
    }


def build_historical_section(items: list[dict[str, Any]]) -> dict[str, Any]:
    """A timeline of past meetings, newest first, each carrying its real ``recording_id``."""
    return {
        "id": HISTORICAL_SECTION_ID,
        "key": HISTORICAL_SECTION_KEY,
        "block": {
            "data": {"items": items},
            "name": HISTORICAL_SECTION_NAME,
            "type": "timeline",
        },
        "origin": "ai_generated",
    }


def latest_job_title(source: dict[str, Any]) -> str:
    """Title of the person's most recent role; current roles outrank ended ones."""
    experiences = _sorted_experiences(source.get("experience"))
    return _clean(experiences[0].get("title")) if experiences else ""


def _display_name(source: dict[str, Any], participant: Participant) -> str:
    """ES name, then the roster name, then the email's local part as a last resort.

    The local part is a guess — ``diana.chen@acme.com`` gives "diana.chen" — but a recognisable
    handle beats an unnamed row in a list of who is on the call.
    """
    name = _clean(source.get("full_name")) or participant.name
    if name:
        return name
    email = participant.email
    return email.split("@", 1)[0] if "@" in email else email


def build_people_section(
    participants: list[Participant],
    people_by_es_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Everyone on the call. ``role`` turns on ``user_id`` alone, never on SOA status."""
    people: list[dict[str, Any]] = []
    for participant in participants:
        source = people_by_es_id.get(participant.es_id or "") or {}
        people.append(
            {
                "name": _display_name(source, participant),
                "role": "Internal Team" if participant.user_id else "External Participant",
                "title": latest_job_title(source),
                "detail": summarize_person(source) if source else "",
                "email": participant.email,
            }
        )
    return {
        "id": PEOPLE_SECTION_ID,
        "key": PEOPLE_SECTION_KEY,
        "block": {
            "data": {"people": people},
            "name": PEOPLE_SECTION_NAME,
            "type": "person",
        },
        "origin": "ai_generated",
    }


def build_company_section(companies: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": COMPANY_SECTION_ID,
        "key": COMPANY_SECTION_KEY,
        "block": {
            "data": {
                "companies": [
                    {
                        "name": (
                            _clean(c.get("li_name"))
                            or _clean(c.get("cb_organization_name"))
                        ),
                        "detail": summarize_company(c),
                    }
                    for c in companies
                ]
            },
            "name": COMPANY_SECTION_NAME,
            "type": "company",
        },
        "origin": "ai_generated",
    }


def build_payload(
    questions: list[str],
    historical_items: list[dict[str, Any]] | None = None,
    *,
    participants: list[Participant] | None = None,
    context: "MeetingContext | None" = None,
) -> dict[str, Any]:
    """The ``payload`` jsonb written to ``temporary_meeting_call_prep``.

    Discovery is always present, even empty. Historical context appears only when there is at
    least one meeting to show: an empty timeline reads as a bug to whoever opens the brief.
    """
    sections = [build_discovery_section(questions)]
    if historical_items:
        sections.append(build_historical_section(historical_items))
    # Unlike historical context, these two are written on every row even when empty: "who is on
    # the call" with nobody resolved is still a true answer, and the client can rely on them.
    if context is not None:
        sections.append(build_company_section(context.companies))
        sections.append(build_people_section(participants or [], context.people_by_es_id))
    return {"sections": sections}


async def write_temporary_prep_rows(
    conn: Any,
    *,
    external_event_id: str,
    payloads_by_user: dict[str, dict[str, Any]],
    trace: Any = None,
) -> tuple[int, int]:
    """Insert one row per (user_id, external_event_id). Returns ``(written, skipped)``.

    ``ON CONFLICT DO NOTHING`` on the unique (user_id, external_event_id) constraint: an existing
    row keeps its payload untouched, since the user may already have edited their prep.
    """
    written = skipped = 0
    for wire_user_id, row_payload in payloads_by_user.items():
        try:
            db_user_id = int(str(wire_user_id).strip())
        except ValueError:
            # The column is an integer; a non-numeric wire id has no row to write.
            logger.warning("[live] temporary prep: non-numeric user_id %r skipped", wire_user_id)
            continue

        row = await conn.fetchrow(
            f"""
            INSERT INTO {TEMPORARY_CALL_PREP_TABLE}
                (external_event_id, user_id, status, payload,
                 ai_generated_at, created_at, updated_at)
            VALUES ($1, $2, 'pending', $3::jsonb, now(), now(), now())
            ON CONFLICT (user_id, external_event_id) DO NOTHING
            RETURNING id
            """,
            external_event_id,
            db_user_id,
            _json.dumps(row_payload),
        )
        if trace:
            trace.step(
                "write_row",
                user_id=db_user_id,
                inserted=row is not None,
                section_keys=[s.get("key") for s in row_payload.get("sections", [])],
                payload=row_payload,
            )
        if row is None:
            skipped += 1
            logger.info(
                "[live] temporary prep row already exists, payload left untouched "
                "user_id=%s external_event_id=%s",
                db_user_id,
                external_event_id,
            )
        else:
            written += 1
    return written, skipped

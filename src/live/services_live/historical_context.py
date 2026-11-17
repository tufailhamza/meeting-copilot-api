"""Per-user historical context: what past meetings led up to this one.

Built independently for every user, from the meetings **that user attended** and nothing else.
Two people on the same call get separate candidate sets and separate LLM calls — one user's
meeting history is never visible in another user's prep.

The shortlist is heuristic, not semantic: same person, recent, and at least one participant in
common with the current meeting besides the user themselves. The LLM then decides which of those
are genuinely part of the same thread.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import asyncpg
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel

from src.config import settings as app_settings
from src.live.config_live import cwk_db_settings

logger = logging.getLogger(__name__)

METADATA_TABLE = "zoro_calendar_events_metadata"
RECORDINGS_TABLE = "zoro_recordings"

LOOKBACK_MONTHS = 2
MAX_CANDIDATES = 20
# What actually reaches the brief: the most relevant of the shortlist, newest first.
MAX_RELEVANT = 5
ITEM_KIND = "transcript"
MAX_SUMMARY_CHARS = 600
HISTORICAL_MODEL = "gpt-4.1"

_POOL: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _POOL
    if _POOL is None:
        if not cwk_db_settings.configured:
            raise RuntimeError("CWK DB settings are not configured")
        _POOL = await asyncpg.create_pool(
            host=cwk_db_settings.cwk_db_host,
            port=cwk_db_settings.cwk_db_port,
            user=cwk_db_settings.cwk_db_user,
            password=cwk_db_settings.cwk_db_password,
            database=cwk_db_settings.cwk_db_name,
            min_size=1,
            max_size=5,
        )
    return _POOL


def _norm_email(value: Any) -> str:
    """JSONB containment is exact, so both sides of every comparison get lowercased."""
    return str(value or "").strip().lower()


def _trim(text: Any, limit: int) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    window = flat[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut > limit // 3:
        return window[: cut + 1]
    cut = window.rfind(" ")
    return (window[:cut] if cut > 0 else window).rstrip(" ,;:-") + "…"


def _format_date(value: date | None) -> str:
    """``July 15, 2026`` — no %-d, which is not portable."""
    if value is None:
        return ""
    return f"{value.strftime('%B')} {value.day}, {value.year}"


def _normalize_bullets(raw: Any) -> str:
    """Tidy the model's markdown without flattening it.

    Whitespace cannot simply be collapsed here: the newlines between bullets are the format. Each
    line is trimmed, blanks dropped, and a leading marker enforced so a stray line still renders
    as a bullet.
    """
    lines = []
    for line in str(raw or "").splitlines():
        text = " ".join(line.split())
        if not text:
            continue
        if not text.startswith("- "):
            text = "- " + text.lstrip("-*• ").strip()
        lines.append(text)
    return "\n".join(lines)


def _participant_emails(raw: Any) -> set[str]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, list):
        return set()
    out = set()
    for item in value:
        email = _norm_email(item.get("email") if isinstance(item, dict) else item)
        if email:
            out.add(email)
    return out


@dataclass
class PastMeeting:
    """One shortlisted past meeting, with the fake id the LLM sees."""

    fake_id: int
    recording_id: str
    # What the brief actually references. Recordings without one are dropped: an item the client
    # cannot link back to a meeting is not worth showing.
    event_id: str
    # Two titles, deliberately. The recording's is AI-written and descriptive, so it helps the
    # model judge relevance; the calendar's is the invite title a person actually recognises, so
    # it is what the brief displays.
    title: str
    calendar_title: str
    short_description: str
    summary: str
    meeting_date: date | None
    participants: list[str]
    overlap: int


class PrequelMeeting(BaseModel):
    """A past meeting THIS meeting continues from, not merely one on a similar subject."""

    id: int
    # Named before ``detail`` so the model states the shared thread before writing about it: a
    # meeting it cannot name a thread for is one it should not have selected.
    continues_thread: str
    detail: str


class HistoricalContextResult(BaseModel):
    continues_from: list[PrequelMeeting]
    not_continuations: list[int]


HISTORICAL_SYSTEM_PROMPT = f"""\
You build the history behind a meeting one person is about to join.

You are NOT looking for related meetings. You are looking for PREQUELS: the earlier meetings that
THIS meeting is the next step of. Those are different questions, and answering the first one is
how this goes wrong. Two meetings can be on the same subject, with the same people, and still be
two separate threads that never touch.

You receive the meeting they are walking into, and a shortlist of their own past meetings that
MIGHT be prequels of it. The shortlist is mechanical — built by matching participants and dates,
not by understanding. Deciding what actually belongs is your job, and most shortlists contain
nothing that does.

## Decide on SUBJECT, not on people

Judge every candidate in this order, and do not reorder it.

**Step 1 — name the thread, from THIS meeting first.** Before looking at any candidate, read the
current meeting's `meeting_title` and `meeting_description` alone and write down what specific
piece of work it is about: "the EMEA rollout", "the pricing model", "Priya's offer". Only then look
for a candidate that is an earlier step in THAT.

`continues_thread` must be nameable from the current meeting's own text. This is the check that
matters most, and getting it backwards is how unrelated meetings get in: if you can point to the
thread in the past meeting but NOT in this one, you have described what that meeting was about, not
what this one continues. Discard it.

Worked example of the mistake. Current meeting: "Product/CS Sync — weekly product and customer
success sync, open tickets and escalations". Candidate: "Qlu Transcriber Demo". Writing
`continues_thread: "AI transcription workflow use cases"` is wrong — nothing in the sync's title or
description mentions transcription. It is a weekly sync about tickets; the demo is a different
thread that happens to involve the same colleagues.

**A meeting with no specific subject has no thread to continue.** Recurring syncs, standups,
weekly catch-ups, "1:1", "Team Meeting" and anything else whose title and description name no
particular piece of work get an EMPTY result. Do not go looking through the candidates for
something to attach them to.

Then check the direction: does the earlier meeting hand something to this one? An open question
this meeting could answer, a decision now due, a commitment coming back around, work that was
blocked pending exactly what is about to be discussed.

If you cannot name the thread, or the thread exists but nothing carries forward, it is NOT a
prequel. Discard it and move on. Do not look at the participants; they cannot rescue it.

**Different account, different candidate, different search = never a prequel.** If both meetings
name a company, a client, a role or a candidate and those differ, stop there. Two searches running
in parallel for the same team are two threads, not one. Same for a pitch to a prospective client
and any meeting about an existing one.

**Step 2 — people, only afterwards.** `shared_participants_with_current_meeting` is a tie-breaker
between candidates that ALREADY named a thread in step 1, and nothing more. It exists because
the shortlist
was built by matching attendees and dates — mechanically, without understanding — so overlap is
the one thing every candidate already has. It is evidence of nothing on its own.

**The default is discard.** Most shortlists are mostly noise. Two meetings between the same
colleagues, a week apart, on two different pieces of work are simply two different threads. Being
on the same team is not a subject.

Never relevant regardless of overlap: recurring standups and status syncs, audio/mic/flow tests,
one-to-ones, and any meeting whose only connection is that the same people attended.

A recurring series is relevant only to ANOTHER INSTANCE OF ITSELF. "Product/CS Sync" continues
into the next "Product/CS Sync" — it does not continue into "Recruiter flow integration", however
many of the same people sat in both.

Return at most {MAX_RELEVANT} meetings in `continues_from`. That is a CEILING, not a target — most
meetings have one prequel or none, and returning three because three slots were free is the failure
this section exists to prevent. If more than {MAX_RELEVANT} genuinely qualify, keep the ones closest
to this meeting's thread and put the rest in `not_continuations`. Every id you were given must
appear exactly once, either as a `continues_from[].id` or in `not_continuations`. Use only the
numeric ids given to you.

For each relevant meeting write `detail` as **markdown bullet points** — a single string, one
bullet per line, each line starting with `- `. No numbering, no headings, no nesting, no blank
lines between bullets.

Write **1 or 2 bullets**. Three only in the rare case where a meeting genuinely produced three
separable things worth carrying forward; never more.

Each bullet is one short sentence holding one fact useful to the person named in `prepared_for` as
they walk into the next meeting: what was decided, what was blocked, what was left open, what
someone committed to. Lead with the substance. Be concrete — names, numbers, commitments — and cut
every scene-setting opener ("In this meeting…", "During the discussion…"), every restatement of
the title, and every hedge about what "may" or "is likely to" be followed up on. Do not summarise
the whole conversation; pick what matters and say it.

Address that person directly as **"you"** — never by name, never in the third person. Other
attendees are named normally. "- You raised concerns about accuracy; Bilal proposed X" is right;
"- Obaid raised concerns…" is wrong.

Example of the exact format:

- You pushed for multilingual support; Bilal said it is not on the roadmap this quarter.
- Left open: whether 'top notes' actually fixes the accuracy gap.

Each `detail` describes ONE meeting — never merge several together. State nothing the inputs do
not support.

If nothing is a prequel, return an empty `continues_from` and every id in `not_continuations`.
That is a correct answer and the most common one — an empty history is far better than a history of
meetings that merely involved the same people. Never invent continuity to fill the brief, and never
include a meeting because the shortlist felt too short without it.
"""


async def fetch_candidates(
    *,
    user_email: str,
    current_emails: set[str],
    current_event_id: str,
    trace: Any = None,
) -> list[PastMeeting]:
    """Shortlist this user's past meetings that share someone with the current meeting.

    Returns at most ``MAX_CANDIDATES``, ranked by participant overlap first and recency second —
    a heavily-overlapping meeting from three weeks ago beats yesterday's one-overlap standup.
    """
    me = _norm_email(user_email)
    others = {e for e in current_emails if e and e != me}
    if not me or not others:
        if trace:
            trace.step(
                "historical.skip",
                user_email=me,
                reason="no other current participants to match against",
            )
        return []

    metadata_sql = f"""
        SELECT recording_id, title, participants, created_at
        FROM {METADATA_TABLE}
        WHERE participants @> $1::jsonb
          AND created_at >= NOW() - INTERVAL '{LOOKBACK_MONTHS} months'
        ORDER BY created_at DESC
    """
    containment = json.dumps([{"email": me}])

    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(metadata_sql, containment)

        overlapped: list[tuple[str, int, Any, list[str], str]] = []
        for row in rows:
            emails = _participant_emails(row["participants"])
            common = emails & others
            if common:
                overlapped.append(
                    (
                        row["recording_id"],
                        len(common),
                        row["created_at"],
                        sorted(emails),
                        str(row["title"] or "").strip(),
                    )
                )

        if trace:
            trace.step(
                "historical.metadata_query",
                user_email=me,
                other_current_participants=sorted(others),
                sql=" ".join(metadata_sql.split()),
                params=[containment],
                rows_returned=len(rows),
                passed_overlap_filter=len(overlapped),
            )

        if not overlapped:
            return []

        # Recordings are fetched for every survivor, before the cap: a deleted or text-less
        # recording must not consume one of the ten slots.
        recording_ids = [item[0] for item in overlapped]
        recordings_sql = f"""
            SELECT id, meeting_title, meeting_short_description, summary, event_id
            FROM {RECORDINGS_TABLE}
            WHERE id = ANY($1::varchar[])
              AND deleted_at IS NULL
        """
        rec_rows = await conn.fetch(recordings_sql, recording_ids)

    by_id = {r["id"]: r for r in rec_rows}
    dropped: list[dict[str, Any]] = []
    usable: list[tuple[str, int, Any, list[str], str, Any]] = []
    for recording_id, overlap, created_at, emails, cal_title in overlapped:
        rec = by_id.get(recording_id)
        if rec is None:
            dropped.append({"recording_id": recording_id, "reason": "deleted or not found"})
            continue
        if str(rec["event_id"] or "") == current_event_id:
            dropped.append({"recording_id": recording_id, "reason": "is the current meeting"})
            continue
        if not (rec["meeting_title"] or rec["summary"] or rec["meeting_short_description"]):
            dropped.append({"recording_id": recording_id, "reason": "no title, summary or description"})
            continue
        if not str(rec["event_id"] or "").strip():
            dropped.append({"recording_id": recording_id, "reason": "no event_id to reference"})
            continue
        usable.append((recording_id, overlap, created_at, emails, cal_title, rec))

    # Recency only. Ranking by participant overlap put "the meetings with the most people in
    # common" at the top of a list that is then cut to MAX_CANDIDATES — so a genuine prequel with
    # one shared attendee could be dropped before the model ever saw it, while three unrelated
    # meetings with the same five colleagues took its place. Overlap already did its job as the
    # search filter above; using it twice is what made the shortlist about people.
    usable.sort(key=lambda item: item[2] or datetime.min, reverse=True)
    capped = usable[:MAX_CANDIDATES]

    candidates = [
        PastMeeting(
            fake_id=index,
            recording_id=recording_id,
            event_id=str(rec["event_id"]).strip(),
            title=str(rec["meeting_title"] or ""),
            # Calendar title is missing on a minority of rows; the recording's stands in.
            calendar_title=cal_title or str(rec["meeting_title"] or ""),
            short_description=_trim(rec["meeting_short_description"], MAX_SUMMARY_CHARS),
            summary=_trim(rec["summary"], MAX_SUMMARY_CHARS),
            meeting_date=created_at.date() if created_at else None,
            participants=emails,
            overlap=overlap,
        )
        for index, (recording_id, overlap, created_at, emails, cal_title, rec) in enumerate(
            capped, 1
        )
    ]

    if trace:
        trace.step(
            "historical.candidates",
            user_email=me,
            sql=" ".join(recordings_sql.split()),
            recordings_requested=len(recording_ids),
            recordings_found=len(rec_rows),
            dropped=dropped,
            usable_before_cap=len(usable),
            selected=len(candidates),
            id_mapping={
                c.fake_id: {"event_id": c.event_id, "recording_id": c.recording_id}
                for c in candidates
            },
            candidates=[
                {
                    "fake_id": c.fake_id,
                    "date": c.meeting_date,
                    "title": c.title,
                    "calendar_title": c.calendar_title,
                    "overlap": c.overlap,
                    "participants": c.participants,
                }
                for c in candidates
            ],
        )
    return candidates


async def build_historical_context(
    *,
    user_email: str,
    user_name: str,
    current_event_id: str,
    meeting_title: str,
    meeting_description: str,
    meeting_context: str,
    our_side: list[str],
    other_side: list[str],
    current_emails: set[str],
    trace: Any = None,
) -> list[dict[str, Any]]:
    """Timeline items for this user, newest first. Empty when there is no history worth writing."""
    candidates = await fetch_candidates(
        user_email=user_email,
        current_emails=current_emails,
        current_event_id=current_event_id,
        trace=trace,
    )
    if not candidates:
        return []

    api_key = app_settings.openai_api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    prompt_payload = {
        "current_meeting": {
            "title": meeting_title,
            "description": meeting_description,
            "prepared_for": {"name": user_name, "email": _norm_email(user_email)},
            "our_side": sorted(our_side),
            "other_side": sorted(other_side),
            "background": meeting_context,
        },
        "candidate_past_meetings": [
            {
                "id": c.fake_id,
                "date": str(c.meeting_date) if c.meeting_date else None,
                "title": c.title,
                "short_description": c.short_description,
                "summary": c.summary,
                "participants": c.participants,
                "shared_participants_with_current_meeting": c.overlap,
            }
            for c in candidates
        ],
    }

    client = AsyncOpenAI(api_key=api_key)
    try:
        response = await client.beta.chat.completions.parse(
            model=HISTORICAL_MODEL,
            messages=[
                {"role": "system", "content": HISTORICAL_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
            ],
            response_format=HistoricalContextResult,
            temperature=0.3,
        )
    except APIError as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"OpenAI API error (historical context): {exc}") from exc

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed historical context")

    by_fake_id = {c.fake_id: c for c in candidates}
    # The thread the model named is the whole basis for each pick, so it belongs in the log: a
    # vague one ("the product", "their tooling") is how an unrelated meeting gets in.
    logger.info(
        "[live] historical: %s -> %s",
        meeting_title,
        [
            {"title": by_fake_id[m.id].calendar_title, "thread": m.continues_thread}
            for m in parsed.continues_from
            if m.id in by_fake_id
        ]
        or "nothing",
    )

    hallucinated = [m.id for m in parsed.continues_from if m.id not in by_fake_id]

    # Relevance order from the model decides what survives the cap; the timeline itself reads
    # newest first.
    chosen: list[tuple[PastMeeting, str]] = []
    seen: set[int] = set()
    for meeting in parsed.continues_from:
        candidate = by_fake_id.get(meeting.id)
        detail = _normalize_bullets(meeting.detail)
        if candidate is None or meeting.id in seen or not detail:
            continue
        seen.add(meeting.id)
        chosen.append((candidate, detail))
        if len(chosen) >= MAX_RELEVANT:
            break

    chosen.sort(key=lambda pair: pair[0].meeting_date or date.min, reverse=True)

    # The fake id never leaves this function: the real recording id is substituted back here.
    items = [
        {
            "date": _format_date(candidate.meeting_date),
            "kind": ITEM_KIND,
            "event_id": candidate.event_id,
            "title": candidate.calendar_title,
            "detail": detail,
        }
        for candidate, detail in chosen
    ]

    if trace:
        trace.step(
            "historical.llm",
            user_email=_norm_email(user_email),
            prompt_payload=prompt_payload,
            model=HISTORICAL_MODEL,
            raw_output=parsed.model_dump(),
            hallucinated_ids_dropped=hallucinated,
            id_mapping_applied={
                m.id: {
                    "event_id": by_fake_id[m.id].event_id,
                    "recording_id": by_fake_id[m.id].recording_id,
                }
                for m in parsed.continues_from
                if m.id in by_fake_id
            },
            items_written=len(items),
            items=items,
        )
    if hallucinated:
        logger.warning(
            "[live] historical context: model returned unknown meeting ids %s", hallucinated
        )
    return items

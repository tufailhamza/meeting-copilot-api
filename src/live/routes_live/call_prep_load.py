from __future__ import annotations

"""HTTP endpoint for the pre-session call-prep load.

Separate from ``call_prep_chat`` on purpose: that endpoint streams a live chat turn, this one runs
once before the meeting to build context. They share neither state, DB pool, nor request shape.
"""

import asyncio
import json
import logging

import asyncpg
from fastapi import APIRouter, Request
from pydantic import Field

from src.live.schemas_live.common import LiveModel
from src.live.services_live import call_prep_load as service
from src.live.services_live import historical_context as history
from src.live.services_live.prep_trace import PrepTrace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/call-prep", tags=["live-call-prep-load"])


class CallPrepLoadUser(LiveModel):
    """One coached user on the event.

    Mirrors the WS roster entry (``SessionUser``) minus ``participantUuid``: speaker matching is a
    live-transcript concern and there is no transcript at load time.
    """

    user_id: str = Field(min_length=1, alias="userId")
    name: str = Field(default="", alias="name")
    email: str = ""
    # Elasticsearch ``people`` document id.
    es_id: str | None = Field(default=None, alias="esId")
    soa_connected: bool = Field(alias="soaConnected")


class ExternalParticipant(LiveModel):
    """A non-QLU attendee. Email is all we always have; ``esId`` may be unknown."""

    email: str = ""
    es_id: str | None = Field(default=None, alias="esId")


class CallPrepLoadRequest(LiveModel):
    """Pre-session prep load for one calendar event and its full roster."""

    external_event_id: str = Field(min_length=1, alias="externalEventId")
    user_id: int = Field(alias="userId")
    meeting_title: str = Field(default="", alias="meetingTitle")
    meeting_description: str = Field(default="", alias="meetingDescription")
    users: list[CallPrepLoadUser] = Field(min_length=1)
    external_participants: list[ExternalParticipant] = Field(
        default_factory=list, alias="externalParticipants"
    )
    # LinkedIn universal names (not ES ids) of companies relevant to this meeting.
    companies: list[str] = Field(default_factory=list)


class CallPrepLoadResponse(LiveModel):
    success: bool


async def _print_raw_payload(request: Request, trace: PrepTrace) -> None:
    """Log the request body exactly as it arrived.

    The parsed model is not a substitute: ``LiveModel`` ignores unknown fields, so anything the
    caller sent that we do not model — the usual cause of "why is this field missing" — would be
    invisible here. Never raises; a debug log must not fail the request.
    """
    try:
        raw = await request.body()
        text = raw.decode("utf-8", errors="replace")
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except ValueError:
            pass  # Not JSON: print the bytes as they came.
    except Exception as exc:  # noqa: BLE001
        text = f"<could not read request body: {type(exc).__name__}: {exc}>"

    try:
        trace.step("request_payload", body=json.loads(text))
    except ValueError:
        trace.step("request_payload", body_raw=text)

    print("[live] call-prep load REQUEST PAYLOAD BEGIN")
    print(text)
    print("[live] call-prep load REQUEST PAYLOAD END")
    logger.info("[live] call-prep load request payload=%s", text)


@router.post("/load", response_model=CallPrepLoadResponse)
async def call_prep_load(
    payload: CallPrepLoadRequest, request: Request
) -> CallPrepLoadResponse:
    """Build a call-prep brief for one calendar event, once.

    Gated on ``non_soa_meetings.initial_call_prep_loaded``: an already-loaded event returns
    immediately. Otherwise: create the meetings row, build meeting context from Elasticsearch,
    generate discovery questions per internal user, write one
    ``temporary_meeting_call_prep`` row each, and mark the event loaded.
    """
    event_id = payload.external_event_id.strip()
    trace = PrepTrace(event_id)
    await _print_raw_payload(request, trace)

    participants = service.build_participants(payload.users, payload.external_participants)

    try:
        return await _load(event_id, payload, participants, trace)
    except asyncpg.exceptions.LockNotAvailableError:
        trace.step("lock_not_available")
        # Another request holds the lock for this event and is doing exactly this work. Waiting
        # longer would only duplicate it, so ack and let that one finish.
        logger.info(
            "[live] call-prep load already in progress elsewhere external_event_id=%s", event_id
        )
        return CallPrepLoadResponse(success=True)
    finally:
        trace.write()


async def _load(
    event_id: str,
    payload: CallPrepLoadRequest,
    participants: list[service.Participant],
    trace: PrepTrace,
) -> CallPrepLoadResponse:
    async with service.meeting_load_guard(event_id) as conn:
        should_load = await service.gate_and_upsert_meeting(
            conn,
            external_event_id=event_id,
            meeting_title=payload.meeting_title,
            meeting_description=payload.meeting_description,
            participants=participants,
        )
        trace.step("gate", should_load=should_load)
        if not should_load:
            logger.info(
                "[live] call-prep load skipped (already loaded) external_event_id=%s", event_id
            )
            return CallPrepLoadResponse(success=True)

        eligible = service.eligible_users(participants)
        trace.step(
            "eligible_users",
            eligible=[{"user_id": p.user_id, "email": p.email} for p in eligible],
            excluded_soa_connected=[
                p.email for p in participants if p.user_id and p.soa_connected
            ],
            external_participants=[p.email for p in participants if not p.user_id],
        )
        if not eligible:
            # Every user is SOA-connected: their prep comes from the enterprise side, so there is
            # nothing for this endpoint to build. The attempt still counts as done.
            logger.info(
                "[live] call-prep load: no non-SOA users external_event_id=%s", event_id
            )
            await service.mark_initial_prep_loaded(conn, event_id)
            return CallPrepLoadResponse(success=True)

        context = await service.build_meeting_context(
            meeting_title=payload.meeting_title,
            meeting_description=payload.meeting_description,
            participants=participants,
            companies=payload.companies,
            trace=trace,
        )
        trace.step(
            "meeting_context",
            context=context.text,
            people_resolved=sorted(context.people_by_es_id),
            companies_resolved=len(context.companies),
        )
        print(f"[live] call-prep context BEGIN external_event_id={event_id}")
        print(context.text)
        print("[live] call-prep context END")

        if not context.text:
            # Empty means the ES lookup failed. Generating from the title alone would bake a thin
            # brief in permanently, so leave the flag false and retry next call.
            logger.warning(
                "[live] call-prep load produced no context, not generating "
                "external_event_id=%s",
                event_id,
            )
            return CallPrepLoadResponse(success=True)

        # History first: each user's own past meetings are the strongest grounding available for
        # their questions, and without it the discovery call has only the title to work from.
        history_by_user = await _build_history(
            event_id, payload, participants, eligible, context.text, trace
        )

        try:
            questions_by_user = await service.generate_discovery_questions(
                meeting_title=payload.meeting_title,
                meeting_description=payload.meeting_description,
                meeting_context=context.text,
                participants=participants,
                history_by_user=history_by_user,
                trace=trace,
            )
        except RuntimeError as exc:
            # Same policy as an ES failure: no rows, flag stays false, next call retries.
            logger.exception(
                "[live] call-prep load: question generation failed external_event_id=%s",
                event_id,
            )
            trace.step("discovery.failed", error=f"{type(exc).__name__}: {exc}")
            return CallPrepLoadResponse(success=True)

        for wire_user_id, questions in questions_by_user.items():
            print(f"[live] discovery questions user_id={wire_user_id}")
            for i, question in enumerate(questions, 1):
                print(f"  {i}. {question}")

        # Every eligible user gets a row, even one the model produced nothing for: the attempt
        # happened, so the event is done and an empty section says so honestly. Historical
        # context is added only where there is text.
        questions_by_user = service.with_empty_for_missing(questions_by_user, participants)
        payloads_by_user = {
            user_id: service.build_payload(
                questions,
                history_by_user.get(user_id),
                participants=participants,
                context=context,
            )
            for user_id, questions in questions_by_user.items()
        }

        written, skipped = await service.write_temporary_prep_rows(
            conn,
            external_event_id=event_id,
            payloads_by_user=payloads_by_user,
            trace=trace,
        )
        logger.info(
            "[live] call-prep load complete external_event_id=%s rows_written=%s "
            "rows_already_present=%s with_history=%s",
            event_id,
            written,
            skipped,
            len(history_by_user),
        )
        await service.mark_initial_prep_loaded(conn, event_id)
        trace.step("marked_loaded", rows_written=written, rows_already_present=skipped)

    return CallPrepLoadResponse(success=True)


async def _build_history(
    event_id: str,
    payload: CallPrepLoadRequest,
    participants: list[service.Participant],
    eligible: list[service.Participant],
    context: str,
    trace: PrepTrace,
) -> dict[str, list[dict]]:
    """One independent pipeline per user, run concurrently.

    Separate candidate sets and separate LLM calls: a user's meeting history must never reach
    another user's prep. A user whose pipeline fails simply gets no historical section — it is
    an enrichment, not a reason to fail the load.
    """
    current_emails = {p.email.strip().lower() for p in participants if p.email}
    our_side = [p.email for p in participants if p.user_id and p.email]
    other_side = [p.email for p in participants if not p.user_id and p.email]

    async def one(user: service.Participant) -> tuple[str, list[dict]]:
        try:
            items = await history.build_historical_context(
                user_email=user.email,
                user_name=user.name,
                current_event_id=event_id,
                meeting_title=payload.meeting_title,
                meeting_description=payload.meeting_description,
                meeting_context=context,
                our_side=[e for e in our_side if e != user.email],
                other_side=other_side,
                current_emails=current_emails,
                trace=trace,
            )
            return user.user_id or "", items
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[live] historical context failed user_id=%s external_event_id=%s",
                user.user_id,
                event_id,
            )
            trace.step(
                "historical.failed",
                user_id=user.user_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return user.user_id or "", []

    results = await asyncio.gather(*(one(u) for u in eligible))
    out = {user_id: items for user_id, items in results if user_id and items}

    for user_id, items in out.items():
        print(f"[live] historical context user_id={user_id} ({len(items)} meetings)")
        for item in items:
            print(f"  [{item['date']}] {item['title']}")
            print(f"      {item['detail']}")
    trace.step(
        "historical.summary",
        users_with_context=sorted(out),
        users_without_context=sorted(
            {u.user_id for u in eligible if u.user_id} - set(out)
        ),
    )
    return out

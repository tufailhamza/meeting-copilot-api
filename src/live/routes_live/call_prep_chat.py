from __future__ import annotations

"""HTTP endpoints for call-prep: chat stream + meeting description enrichment."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.live.config_live import live_settings
from src.live.services_live.call_prep_chat import (
    MAX_TURN_TOKENS,
    REGENERATION_TRIGGER_ACTION,
    TemporaryPrepRow,
    append_chat_session_turn,
    approved_regeneration_payload,
    clear_pending_regeneration,
    call_enterprise_chat,
    call_enterprise_inject_call_prep,
    chunk_message_text,
    compose_user_message,
    extract_discovery_questions,
    iter_forwardable_chat_events,
    collect_known_urls,
    load_host_persona,
    load_or_create_temporary_prep,
    load_soa_meeting_context,
    mark_session_injected,
    messages_for_prompt_chronological,
    normalize_sections_payload,
    regeneration_is_approved,
    resolve_regeneration_plan,
    restrict_turn_to_regeneration,
    run_call_prep_chat,
    sections_fingerprint,
    store_pending_regeneration,
    stream_call_prep_chat,
    stream_enterprise_chat,
    sync_chat_session,
    update_temporary_prep_sections,
    plan_section_changes,
    remove_section,
    upsert_section,
)
from src.live.services_live.meeting_description import enrich_meeting_description_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/call-prep", tags=["live-call-prep"])


def _print_enterprise_exact_response(label: str, result: dict) -> None:
    exact_response = result.get("response", result.get("error"))
    if isinstance(exact_response, (dict, list)):
        exact_response_text = json.dumps(
            exact_response, ensure_ascii=False, default=str, indent=2
        )
    else:
        exact_response_text = str(exact_response)
    print(f"[live] soa_connected {label} EXACT RESPONSE BEGIN")
    print(f"status_code={result.get('status_code')}")
    if result.get("error") and "response" not in result:
        print(f"error={result.get('error')}")
    print(exact_response_text)
    print(f"[live] soa_connected {label} EXACT RESPONSE END")
    logger.info(
        "[live] soa_connected %s status_code=%s response=%s",
        label,
        result.get("status_code"),
        exact_response,
    )


def _ndjson_line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, default=str) + "\n"


async def _iter_lines(lines: list[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


def _assistant_text_from_events(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        if event.get("type") == "message":
            text = event.get("text")
            if text:
                parts.append(str(text))
    return "".join(parts).strip()


class CallPrepChatRequest(BaseModel):
    external_event_id: str = Field(min_length=1)
    user_id: int
    conversation_id: UUID
    meeting_title: str
    meeting_description: str
    user_message: str = Field(min_length=1)
    soa_connected: bool = False
    # Background the caller wants the assistant to have for this turn only. Optional and
    # backwards-compatible: absent, null or empty leaves ``user_message`` untouched on the wire.
    additional_context: str | None = None


class MeetingDescriptionRequest(BaseModel):
    meeting_title: str = ""
    meeting_description: str = ""


class MeetingDescriptionResponse(BaseModel):
    meeting_description: str


async def _prepare_soa_connected_stream(payload: CallPrepChatRequest) -> AsyncIterator[str]:
    """Do the setup that may fail, then return a generator that forwards enterprise's stream.

    Everything able to raise ``HTTPException`` runs here, before the response starts: once the
    first byte is on the wire the status code is already sent and an error can no longer be
    reported properly.
    """
    event_id = payload.external_event_id.strip()
    conv_id = str(payload.conversation_id)

    # Scope by (external_event_id, user_id): one meetings row exists per participant, and the
    # resolved row's identity drives the enterprise inject/chat calls below.
    meeting_ctx = await load_soa_meeting_context(event_id, user_id=payload.user_id)
    if meeting_ctx is None:
        logger.info(
            "[live] soa_connected meeting not found external_event_id=%s user_id=%s",
            event_id,
            payload.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No meeting found for external_event_id={event_id} "
                f"and user_id={payload.user_id}"
            ),
        )

    session = await sync_chat_session(
        user_id=payload.user_id,
        external_event_id=event_id,
        conversation_id=conv_id,
    )

    if session.should_inject:
        inject_result = await call_enterprise_inject_call_prep(
            user_id=meeting_ctx.user_id,
            meeting_id=meeting_ctx.meeting_id,
            user_email=meeting_ctx.user_email,
            conversation_id=conv_id,
        )
        _print_enterprise_exact_response("inject-call-prep", inject_result)
        if not inject_result.get("ok"):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Enterprise inject failed: "
                    f"{inject_result.get('error') or inject_result.get('status_code')}"
                ),
            )
        await mark_session_injected(user_id=payload.user_id, external_event_id=event_id)
    else:
        logger.info(
            "[live] soa_connected skip inject (already injected) "
            "conversation_id=%s external_event_id=%s user_id=%s",
            conv_id,
            event_id,
            payload.user_id,
        )

    async def generate() -> AsyncIterator[str]:
        reply_parts: list[str] = []
        try:
            async for event in stream_enterprise_chat(
                user_id=meeting_ctx.user_id,
                name=meeting_ctx.user_name,
                email=meeting_ctx.user_email,
                # Only the outbound call carries the extra context; what we store below stays the
                # raw message, so the thread history is not padded with a context block.
                user_message=compose_user_message(
                    payload.user_message, payload.additional_context
                ),
                conversation_id=conv_id,
            ):
                if event.get("type") == "message" and event.get("text"):
                    reply_parts.append(str(event["text"]))
                yield _ndjson_line(event)
        except Exception:  # noqa: BLE001
            # The status line is long gone, so the stream just ends here.
            logger.exception("[live] soa chat stream failed external_event_id=%s", event_id)

        await append_chat_session_turn(
            user_id=payload.user_id,
            external_event_id=event_id,
            user_message=payload.user_message,
            assistant_reply="".join(reply_parts).strip(),
        )

    return generate()


async def _prepare_soa_not_connected_stream(
    payload: CallPrepChatRequest,
) -> AsyncIterator[str]:
    """Setup first (so failures are real HTTP errors), then stream the turn as it is generated."""
    event_id = payload.external_event_id.strip()
    conv_id = str(payload.conversation_id)

    if not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY must be set to run call-prep chat.",
        )

    try:
        session = await sync_chat_session(
            user_id=payload.user_id,
            external_event_id=event_id,
            conversation_id=conv_id,
        )
        row: TemporaryPrepRow = await load_or_create_temporary_prep(
            external_event_id=event_id,
            user_id=payload.user_id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    existing_payload = normalize_sections_payload(row.payload)
    existing_sections = [
        s for s in (existing_payload.get("sections") or []) if isinstance(s, dict)
    ]
    existing_questions = extract_discovery_questions(existing_payload)
    recent_conversation = messages_for_prompt_chronological(session.messages_newest_first)
    # Which of the people on the call is the one typing. Every participant's row carries the same
    # people section, so without this the assistant writes questions for whoever it guesses — and
    # a question aimed at the host instead of by them is the failure that shows up in the prep.
    host = await load_host_persona(event_id, payload.user_id, existing_sections)
    if host is None:
        logger.info(
            "[live] call-prep chat: host not resolved external_event_id=%s user_id=%s",
            event_id,
            payload.user_id,
        )

    # A regeneration offered last turn is answered by this one, whatever the host says. Approving
    # it carries out exactly what was announced; anything else drops it, so an offer can never sit
    # around and be applied to a message that was not answering it.
    regeneration_enabled = live_settings.call_prep_regeneration_enabled
    pending = session.pending_regeneration if regeneration_enabled else None
    approved_changes: list[dict[str, Any]] = []
    if pending:
        if not regeneration_is_approved(payload.user_message, payload.additional_context):
            logger.info("[live] chat: regeneration offer declined external_event_id=%s", event_id)
        elif pending.get("fingerprint") != sections_fingerprint(existing_sections):
            # The prep moved since the offer was made, so the plan describes a prep that no longer
            # exists. Applying it would carry out an intent against content the host never saw.
            logger.info("[live] chat: regeneration offer stale external_event_id=%s", event_id)
        else:
            approved_changes = list(pending.get("changes") or [])
        await clear_pending_regeneration(
            user_id=payload.user_id,
            external_event_id=event_id,
        )

    approved_regeneration = (
        approved_regeneration_payload({**pending, "changes": approved_changes})
        if approved_changes and pending
        else None
    )

    async def generate() -> AsyncIterator[str]:
        nonlocal existing_payload
        turn = None
        try:
            async for kind, item in stream_call_prep_chat(
                meeting_title=payload.meeting_title,
                meeting_description=payload.meeting_description,
                existing_sections=existing_sections,
                existing_discovery_questions=existing_questions,
                user_message=payload.user_message,
                recent_conversation=recent_conversation,
                additional_context=payload.additional_context,
                host=host,
                approved_regeneration=approved_regeneration,
                # An approved regeneration rewrites several sections in one turn; every other turn
                # keeps the ordinary budget.
                max_tokens=(
                    live_settings.regeneration_max_turn_tokens
                    if approved_regeneration
                    else MAX_TURN_TOKENS
                ),
            ):
                if kind == "event":
                    yield _ndjson_line(item)
                else:
                    turn = item
        except Exception:  # noqa: BLE001
            # Mid-stream: the status code is already sent, so end the stream rather than pretend
            # a 502 can still be raised. Nothing is written, so the host's prep is untouched.
            logger.exception(
                "[live] call-prep chat stream failed external_event_id=%s", event_id
            )
            return

        if turn is None:
            return

        if approved_regeneration:
            # The host approved a named list of sections. Narrowing that list is theirs to do;
            # reaching past it is not, however good the extra section would be.
            restrict_turn_to_regeneration(turn, existing_sections, approved_changes)

        # A confirmation request never writes: the host asked to see a draft, not to change their
        # prep. Both flags true is contradictory, and asking is the safe reading of it.
        # An approved regeneration is past that question — the yes was the confirmation.
        awaiting_confirmation = turn.needs_confirmation and not approved_regeneration
        # Planned against the pre-turn snapshot, so "create" vs "edit" is decided by what is
        # actually in the prep rather than by the label the model put on it. Immutable and
        # permanent sections are refused inside the planner — the turn still succeeds, it just
        # does not carry out that one operation.
        plan = plan_section_changes(
            turn,
            existing_sections,
            prompt=payload.user_message,
            # A ``links`` section may only surface an address the meeting already carries — a
            # recalled URL looks authoritative and goes nowhere.
            known_urls=collect_known_urls(
                existing_sections, payload.meeting_description, payload.user_message
            ),
        )

        updated_section_ids: list[str] = []
        deleted_section_ids: list[str] = []
        if not awaiting_confirmation:
            for write in plan.writes:
                existing_payload = upsert_section(existing_payload, write.section)
                updated_section_ids.append(write.section["id"])
            for removal in plan.removals:
                existing_payload, removed_id = remove_section(existing_payload, removal.key)
                if removed_id:
                    deleted_section_ids.append(removed_id)

        if updated_section_ids or deleted_section_ids:
            await update_temporary_prep_sections(row.id, existing_payload)

        sections = normalize_sections_payload(existing_payload).get("sections") or []

        await append_chat_session_turn(
            user_id=payload.user_id,
            external_event_id=event_id,
            user_message=payload.user_message,
            assistant_reply=turn.assistant_reply,
        )

        yield _ndjson_line(
            {
                "type": "call_prep",
                "meeting_id": None,
                "meeting_title": payload.meeting_title,
                "updated_section_ids": updated_section_ids,
                "deleted_section_ids": deleted_section_ids,
                "call_prep": {"sections": sections},
            }
        )

        # An offer to bring the whole prep back in line, made because the message carried context
        # the prep does not reflect. Resolved against the sections as they stand AFTER this turn's
        # writes, so the plan and the fingerprint both describe what the next turn will load.
        offer: list[dict[str, Any]] = []
        if regeneration_enabled and not approved_regeneration and turn.regeneration is not None:
            offer = resolve_regeneration_plan(turn.regeneration, sections)
            if offer:
                await store_pending_regeneration(
                    user_id=payload.user_id,
                    external_event_id=event_id,
                    changes=offer,
                    # The message that prompted the offer, kept because the turn that carries it
                    # out needs the context the plan was based on and history holds three pairs.
                    context=payload.user_message,
                    fingerprint=sections_fingerprint(sections),
                )

        # Only ever one kind of trigger per turn. A proposing turn returns no sections, so there is
        # normally nothing to collide with — but if a turn somehow did both, two pending offers
        # and one yes/no is a question the host cannot answer, and the prep-wide one is the offer
        # the reply was written about.
        if offer:
            yield _ndjson_line(
                {
                    "type": "trigger",
                    "action": REGENERATION_TRIGGER_ACTION,
                    "values": ["yes", "no"],
                }
            )
        elif awaiting_confirmation:
            # One trigger per pending change, named after what it would do — a section draft asks
            # ``add_watch_points``, not the generic questions action. Same shape the enterprise
            # service uses, so the client needs no new handling; the host's answer arrives as an
            # ordinary ``user_message`` next turn.
            for action in plan.triggers():
                yield _ndjson_line(
                    {
                        "type": "trigger",
                        "action": action,
                        "values": ["yes", "no"],
                    }
                )

    return generate()


@router.post("/meeting_description", response_model=MeetingDescriptionResponse)
async def meeting_description(
    payload: MeetingDescriptionRequest,
) -> MeetingDescriptionResponse:
    """Return a cleaned, enriched meeting description (1–2 paragraphs, no links)."""
    if not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY must be set to enrich meeting descriptions.",
        )

    title = payload.meeting_title.strip()
    desc = payload.meeting_description.strip()
    if not title and not desc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of meeting_title or meeting_description must be non-empty.",
        )

    try:
        enriched = await asyncio.to_thread(
            enrich_meeting_description_sync,
            title,
            desc,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception("[live] meeting_description generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Meeting description generation failed: {exc}",
        ) from exc

    return MeetingDescriptionResponse(meeting_description=enriched)


@router.post("/chat")
async def call_prep_chat(payload: CallPrepChatRequest) -> StreamingResponse:
    """Stream call-prep chat events (`reasoning`, `message`, `call_prep`, `trigger`) as NDJSON."""
    try:
        if payload.soa_connected:
            generator = await _prepare_soa_connected_stream(payload)
        else:
            generator = await _prepare_soa_not_connected_stream(payload)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("[live] call-prep chat unexpected failure")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return StreamingResponse(
        generator,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

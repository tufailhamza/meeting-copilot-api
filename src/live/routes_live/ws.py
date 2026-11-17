"""Live WebSocket endpoint.

One socket per meeting, shared by every coached user on it. Outbound frames carry ``user_id``:
non-null for a specific user, null for broadcast. The client renders the frames addressed to the
signed-in user and the broadcast ones, and drops the rest.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from src.live.config_live import live_settings
from src.live.schemas_live import (
    ChatMessageIn,
    ErrorMessage,
    QuestionView,
    SessionReady,
    TranscriptSessionStartMessage,
    UserQuestions,
    dump_outbound,
    parse_inbound,
)
from src.live.services_live.event_bus import EventBus
from src.live.services_live.registry import registry
from src.live.services_live.trace_log import log_inbound, log_outbound

logger = logging.getLogger(__name__)

router = APIRouter()


async def _send(websocket: WebSocket, message) -> None:
    await websocket.send_text(dump_outbound(message))


async def _forward_outbound(websocket: WebSocket, bot_id: str, bus: EventBus) -> None:
    try:
        async for message in bus.subscribe(bot_id):
            await _send(websocket, message)
    except asyncio.CancelledError:
        return


def _build_session_ready(bot_id: str, session) -> SessionReady:
    """One group per coached user, in roster order.

    Users with no prep are included with an empty list rather than omitted, so the client can tell
    "on the session with no questions" apart from "not on the session".
    """
    return SessionReady(
        bot_id=bot_id,
        users=[
            UserQuestions(
                user_id=user_id,
                questions=[
                    QuestionView(id=q.id, text=q.text, status=q.status)
                    for q in session.ordered_questions_for(user_id)
                ],
            )
            for user_id in session.user_ids
        ],
    )


@router.websocket("/ws")
async def live_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    bot_id: str | None = None
    actor = None
    forward_task: asyncio.Task | None = None
    logger.info("[live] websocket connected client=%s", websocket.client)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = parse_inbound(raw)
            except ValidationError:
                logger.warning("[live] inbound invalid_message client=%s", websocket.client)
                err = ErrorMessage(
                    user_id=None,
                    code="invalid_message",
                    message="could not parse message",
                )
                log_outbound(bot_id or "unknown", err)
                await _send(websocket, err)
                continue

            if bot_id is None:
                if not isinstance(message, TranscriptSessionStartMessage):
                    # Attribute the error when the frame told us who sent it.
                    sender = message.user_id if isinstance(message, ChatMessageIn) else None
                    err = ErrorMessage(
                        user_id=sender,
                        code="expected_session_start",
                        message="first message must be 'transcript_session_start'",
                    )
                    log_outbound("unknown", err)
                    await _send(websocket, err)
                    continue
                bot_id = message.bot_id
                log_inbound(
                    bot_id,
                    "transcript_session_start",
                    event_id=message.event_id,
                    user_ids=[u.user_id for u in message.users],
                    meeting_type=message.meeting_type,
                )
                actor = await registry.get_or_create(bot_id)
                forward_task = asyncio.create_task(
                    _forward_outbound(websocket, bot_id, registry.event_bus),
                    name=f"live-ws-forward-{bot_id}",
                )
                await actor.submit(message)
                try:
                    await asyncio.wait_for(
                        actor.ready.wait(),
                        timeout=live_settings.prep_load_timeout_seconds,
                    )
                except TimeoutError:
                    logger.warning(
                        "[live] prep load timed out bot_id=%s event_id=%s",
                        bot_id,
                        message.event_id,
                    )
                ready = _build_session_ready(bot_id, actor.session)
                log_outbound(bot_id, ready)
                await _send(websocket, ready)
                continue

            if hasattr(message, "type"):
                if message.type == "transcript_session_end":
                    log_inbound(bot_id, "transcript_session_end", event_id=getattr(message, "event_id", None))
                elif message.type != "transcript_segment" and message.type != "chat_message":
                    log_inbound(bot_id, message.type)

            await actor.submit(message)
    except WebSocketDisconnect:
        logger.info(
            "[live] websocket disconnected client=%s bot_id=%s",
            websocket.client,
            bot_id,
        )
    finally:
        if forward_task is not None:
            forward_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await forward_task
        if bot_id is not None:
            await registry.remove(bot_id)

"""Structured trace logging for the live coach (grep Cloud Run logs for ``[live]``)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TRACE_PREFIX = "[live]"


def _clip(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def log_inbound(bot_id: str, message_type: str, **fields: Any) -> None:
    """Log a validated inbound WS frame (high-volume types at DEBUG)."""
    if message_type == "transcript_segment":
        logger.debug(
            "%s inbound bot_id=%s type=%s seq=%s speaker=%s",
            _TRACE_PREFIX,
            bot_id,
            message_type,
            fields.get("seq"),
            fields.get("speaker"),
        )
        return
    if message_type == "chat_message":
        logger.info(
            "%s inbound bot_id=%s type=%s text=%s",
            _TRACE_PREFIX,
            bot_id,
            message_type,
            _clip(fields.get("text")),
        )
        return
    logger.info(
        "%s inbound bot_id=%s type=%s %s",
        _TRACE_PREFIX,
        bot_id,
        message_type,
        " ".join(f"{k}={v!r}" for k, v in fields.items()) if fields else "",
    )


def log_outbound(bot_id: str, message: Any) -> None:
    """Log a wire/outbound frame sent to the client."""
    msg_type = getattr(message, "type", type(message).__name__)
    if msg_type == "chat_token":
        logger.debug(
            "%s outbound bot_id=%s type=chat_token user_id=%s message_id=%s",
            _TRACE_PREFIX,
            bot_id,
            getattr(message, "user_id", None),
            getattr(message, "message_id", None),
        )
        return

    detail = _outbound_detail(message, msg_type)
    # user_id=None is meaningful: the frame is a broadcast, not a missing attribution.
    audience = getattr(message, "user_id", None)
    logger.info(
        "%s outbound bot_id=%s type=%s user_id=%s %s",
        _TRACE_PREFIX,
        bot_id,
        msg_type,
        audience if audience is not None else "*broadcast*",
        detail,
    )


def _outbound_detail(message: Any, msg_type: str) -> str:
    if msg_type == "session_ready":
        users = getattr(message, "users", None) or []
        counts = ",".join(f"{u.user_id}:{len(u.questions)}" for u in users)
        return f"users={len(users)} questions_per_user={counts or 'none'}"
    if msg_type == "mark_question":
        # ``status`` is what separates a first answer from a revision, so it belongs in the log.
        return (
            f"question_id={getattr(message, 'question_id', None)!r} "
            f"status={getattr(message, 'status', None)!r}"
        )
    if msg_type == "question_added":
        return f"id={getattr(message, 'id', None)!r}"
    if msg_type == "suggestion":
        return f"id={getattr(message, 'id', None)!r} text={_clip(getattr(message, 'text', None))!r}"
    if msg_type == "fact_check":
        return (
            f"id={getattr(message, 'id', None)!r} verdict={getattr(message, 'verdict', None)!r} "
            f"claim={_clip(getattr(message, 'claim', None))!r}"
        )
    if msg_type == "risk":
        return f"id={getattr(message, 'id', None)!r} text={_clip(getattr(message, 'text', None))!r}"
    if msg_type == "chat_message":
        return f"id={getattr(message, 'id', None)!r} text={_clip(getattr(message, 'text', None))!r}"
    if msg_type == "error":
        return f"code={getattr(message, 'code', None)!r}"
    return ""


def log_gate_decision(
    bot_id: str,
    *,
    verdict: str,
    confidence: float,
    rationale: str | None,
    reasoner_ran: bool,
) -> None:
    logger.info(
        "%s gate bot_id=%s verdict=%s confidence=%.2f reasoner_ran=%s rationale=%s",
        _TRACE_PREFIX,
        bot_id,
        verdict,
        confidence,
        reasoner_ran,
        _clip(rationale),
    )


def log_orchestrator_skip(
    bot_id: str,
    reason: str,
    *,
    gate_verdict: str | None = None,
    confidence: float | None = None,
    action: str | None = None,
    detail: str | None = None,
) -> None:
    parts = [f"reason={reason}"]
    if gate_verdict is not None:
        parts.append(f"gate={gate_verdict}")
    if confidence is not None:
        parts.append(f"confidence={confidence:.2f}")
    if action is not None:
        parts.append(f"action={action}")
    if detail:
        parts.append(f"detail={_clip(detail)}")
    logger.info(
        "%s orchestrator skip bot_id=%s %s",
        _TRACE_PREFIX,
        bot_id,
        " ".join(parts),
    )

"""ChatAgent: 1-1 host chat with optional refine/add tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.live.services_live.chat.client import (
    ChatClient,
    DeltaHandler,
    build_chat_client,
    iter_reply_deltas,
)
from src.live.services_live.chat.context import build_chat_context
from src.live.services_live.chat.schemas import ChatTurnResult
from src.live.services_live.chat.web_search import WebSearchResult, run_web_search
from src.live.services_live.company_filter import (
    CompanyFilterResult,
    generate_similar_companies,
    map_company_name,
)
from src.live.services_live.events import MeetingEvent
from src.live.services_live.projection import MeetingSession
from src.live.services_live.tools.add_question import AddQuestionTool
from src.live.services_live.tools.refine_suggestion import RefineSuggestionTool

logger = logging.getLogger(__name__)


async def _ignore_delta(_: str) -> None:
    return None


def _not_superseded(partial: dict) -> bool:
    """Suppress streaming a reply that a follow-up step is about to replace."""
    if partial.get("lookup_company") and partial.get("company_name"):
        return False
    if partial.get("use_web_search") and partial.get("web_search_query"):
        return False
    return True


def _company_record(result: CompanyFilterResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "company_name": result.company_name,
        "es_id": result.es_id,
        "li_urn": result.li_urn,
        "li_name": result.li_name,
        "li_universalname": result.li_universalname,
        "li_industries": list(result.li_industries),
    }


@dataclass(frozen=True)
class ChatRunResult:
    """Outcome of one chat turn.

    Reply text has already been streamed to the host via ``on_delta`` by the time this returns —
    ``turn.reply`` is the complete text for the final message and for history.
    """

    turn: ChatTurnResult
    tool_events: tuple[MeetingEvent, ...]
    additional_data: dict[str, Any] | None = None


class ChatAgent:
    def __init__(self, *, client: ChatClient | None = None) -> None:
        self._client = client
        self._refine = RefineSuggestionTool()
        self._add_question = AddQuestionTool()

    async def _search_company_on_web(
        self,
        session: MeetingSession,
        company_name: str,
    ) -> str | None:
        """Public-web fallback for a company our records don't hold. Never raises."""
        query = (
            f"{company_name} company overview: what the company does, "
            f"industry, size and location"
        )
        try:
            result = await run_web_search(query, session)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[live] chat company web fallback failed bot_id=%s company=%r",
                session.bot_id,
                company_name,
            )
            return None
        return (result.reply or "").strip() or None

    async def run(
        self,
        session: MeetingSession,
        user_text: str,
        *,
        user_id: str,
        on_delta: DeltaHandler | None = None,
    ) -> ChatRunResult:
        """Run one chat turn for ``user_id``, streaming reply text through ``on_delta``.

        Every mutation this produces is scoped to that user: questions land on their list,
        suggestion refinements touch only their copy.
        """
        emit = on_delta or _ignore_delta
        if self._client is None:
            logger.info("[live] chat unavailable bot_id=%s reason=no_model_client", session.bot_id)
            return ChatRunResult(
                turn=ChatTurnResult(
                    reply="Chat is unavailable right now (no model configured).",
                ),
                tool_events=(),
                additional_data=None,
            )

        ctx = build_chat_context(session, user_id=user_id)
        try:
            # Routing first: a reply that a lookup or a search is about to replace never streams.
            turn = await self._client.stream_turn(
                ctx, user_text, emit, should_emit=_not_superseded
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[live] chat inference failed bot_id=%s user_id=%s", session.bot_id, user_id
            )
            return ChatRunResult(
                turn=ChatTurnResult(reply="Sorry, I couldn't process that message. Please try again."),
                tool_events=(),
                additional_data=None,
            )

        # A company question is answered from the company service, never from the model's own
        # memory: resolve the name, then compose the reply with the record in hand.
        if turn.lookup_company and turn.company_name:
            resolved = await map_company_name(turn.company_name)
            web_findings: str | None = None
            if resolved is None:
                # Our records don't have it — fall back to the web rather than dead-ending, and
                # fold the result into *this* reply instead of making the host ask again.
                logger.info(
                    "[live] chat company lookup miss bot_id=%s user_id=%s company=%r — trying web",
                    session.bot_id,
                    user_id,
                    turn.company_name,
                )
                web_findings = await self._search_company_on_web(session, turn.company_name)
            lookup_ctx = build_chat_context(
                session,
                user_id=user_id,
                company_lookup={
                    "query": turn.company_name,
                    "found": resolved is not None,
                    "company": _company_record(resolved),
                    "web_findings": web_findings,
                },
            )
            try:
                turn = await self._client.stream_turn(lookup_ctx, user_text, emit)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[live] chat company answer failed bot_id=%s user_id=%s",
                    session.bot_id,
                    user_id,
                )
                return ChatRunResult(
                    turn=ChatTurnResult(
                        reply="Sorry, I couldn't look that company up. Please try again."
                    ),
                    tool_events=(),
                    additional_data=None,
                )

        tool_events: list[MeetingEvent] = []
        if turn.refine_suggestion_id and turn.refined_suggestion_text:
            result = self._refine.execute(
                session,
                turn.refine_suggestion_id,
                turn.refined_suggestion_text,
                user_id=user_id,
            )
            if result.applied:
                tool_events.extend(result.events)
            else:
                logger.info(
                    "[live] chat refine skipped bot_id=%s user_id=%s reason=%s",
                    session.bot_id,
                    user_id,
                    result.skip_reason,
                )

        if turn.add_question_text:
            result = self._add_question.execute(session, turn.add_question_text, user_id=user_id)
            if result.applied:
                tool_events.extend(result.events)
            else:
                logger.info(
                    "[live] chat add_question skipped bot_id=%s user_id=%s reason=%s",
                    session.bot_id,
                    user_id,
                    result.skip_reason,
                )

        # Company generation: when requested, call the external service and attach results
        # as additional_data in the final chat_message.
        additional_data: dict[str, Any] | None = None
        if getattr(turn, "generate_similar_companies", False) and getattr(
            turn, "similar_companies_query", None
        ):
            companies = await generate_similar_companies(turn.similar_companies_query or "")
            if companies:
                related: list[dict[str, Any]] = []
                for c in companies:
                    related.append(
                        {
                            "company_name": c.company_name,
                            "company_info": {
                                "es_id": c.es_id,
                                "li_urn": c.li_urn,
                                "li_name": c.li_name,
                                "li_universalname": c.li_universalname,
                                "li_industries": list(c.li_industries),
                            },
                        }
                    )
                additional_data = {"related_companies": related}

        # Optional web search: decided by the routing/chat model. This runs *after* generation,
        # so the streamed text is superseded — the final chat_message carries the authoritative
        # reply, which is why clients are told to render from it rather than from the token stream.
        if getattr(turn, "use_web_search", False) and getattr(turn, "web_search_query", None):
            web_query = turn.web_search_query or ""
            try:
                web_result: WebSearchResult = await run_web_search(web_query, session)
            except Exception:  # noqa: BLE001
                logger.exception("[live] web search failed bot_id=%s", session.bot_id)
            else:
                if web_result.reply:
                    turn.reply = web_result.reply
                    # The first reply was suppressed, so stream the answer the host actually gets.
                    for delta in iter_reply_deltas(web_result.reply):
                        await emit(delta)

        # Attach additional_data so MeetingActor can forward it on wire.
        return ChatRunResult(turn=turn, tool_events=tuple(tool_events), additional_data=additional_data)


def build_default_chat_agent() -> ChatAgent:
    return ChatAgent(client=build_chat_client())

"""Map domain events to outbound WebSocket messages (wire contract).

The mapping is 1:1 — fan-out to multiple users happens upstream in the tools, which emit one
domain event per owning user. Every frame produced here carries ``user_id``: the owning user for
addressed events, an explicit ``None`` for events the whole session should see.
"""

from __future__ import annotations

from src.live.schemas_live.common import FactCheckVerdict, QuestionStatus
from src.live.schemas_live.outbound import (
    Answer as WireAnswer,
    CompanyDetectedOut as WireCompanyDetected,
    CompanyInfo as WireCompanyInfo,
    FactCheck as WireFactCheck,
    OutboundMessage,
    QuestionAdded as WireQuestionAdded,
    MarkQuestion as WireMarkQuestion,
    Risk as WireRisk,
    Suggestion as WireSuggestion,
)
from src.live.services_live.events import (
    AgendaRiskEmitted,
    CompanyDetected,
    FactCheckEmitted,
    MeetingEvent,
    QuestionAdded,
    QuestionAnswered,
    PrepQuestionAnswered,
    SuggestionEmitted,
    SuggestionRefined,
)


def domain_event_to_wire(event: MeetingEvent) -> OutboundMessage | None:
    """Return a client-visible outbound frame for ``event``, or ``None`` if internal-only."""
    if isinstance(event, PrepQuestionAnswered):
        # The client is told what changed, not just what the state is: a first answer and a
        # revision render differently, and ``revision`` is the only thing that separates them.
        return WireMarkQuestion(
            user_id=event.user_id,
            question_id=event.question_id,
            status="answered" if event.revision <= 1 else "answer_updated",
            answer_text=event.answer,
            evidence_turn_seq=event.evidence_turn_seq,
            evidence_turn_seq_end=event.evidence_turn_seq_end,
        )
    if isinstance(event, QuestionAdded):
        return WireQuestionAdded(
            user_id=event.user_id,
            id=event.question_id,
            text=event.text,
            status=QuestionStatus.PENDING,
        )
    if isinstance(event, SuggestionEmitted):
        return WireSuggestion(
            user_id=event.user_id,
            id=event.suggestion_id,
            text=event.text,
            rationale=event.rationale,
            linked_question_id=event.linked_question_id,
        )
    if isinstance(event, SuggestionRefined):
        return WireSuggestion(
            user_id=event.user_id,
            id=event.suggestion_id,
            text=event.text,
            rationale=event.rationale,
        )
    if isinstance(event, FactCheckEmitted):
        return WireFactCheck(
            user_id=None,  # broadcast: the claim was made in the shared transcript
            id=event.fact_check_id,
            claim=event.claim,
            verdict=event.verdict,
            confidence=event.confidence,
            detail=event.detail,
            sources=list(event.sources),
        )
    if isinstance(event, QuestionAnswered):
        # TEMPORARY: answers are emitted wearing the ``fact_check`` shape so the current frontend
        # renders them without new work. The ``Answer`` frame below is the real contract and is
        # kept intact — restoring it is a one-block swap once the client supports it.
        #
        #     return WireAnswer(
        #         user_id=None,
        #         id=event.answer_id,
        #         question=event.question,
        #         answer=event.answer,
        #         source=event.source,
        #     )
        return WireFactCheck(
            user_id=None,  # broadcast: the room asked the question
            id=event.answer_id,
            claim=event.question,      # the question becomes the card headline
            verdict=FactCheckVerdict.TRUE,
            confidence=1.0,
            detail=event.answer,       # the answer becomes the card body
            sources=[],
        )
    if isinstance(event, AgendaRiskEmitted):
        return WireRisk(user_id=event.user_id, id=event.risk_id, text=event.text)
    if isinstance(event, CompanyDetected):
        return WireCompanyDetected(
            user_id=None,  # broadcast: the company was named in the shared transcript
            id=event.company_id,
            company_name=event.company_name,
            title=event.title,
            text=event.text,
            company_info=WireCompanyInfo(
                es_id=event.es_id,
                li_urn=event.li_urn,
                li_name=event.li_name,
                li_universalname=event.li_universalname,
                li_industries=list(event.li_industries),
            ),
        )
    return None

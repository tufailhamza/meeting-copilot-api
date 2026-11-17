"""Projection: the live meeting state as a fold over the event log.

One session coaches several users at once. What is **shared** and what is **per user**:

- shared  : transcript, rolling summary, fact-checks, detected companies, evaluation triggers
- per user: prep questions, prep (CRM) context, suggestions, risks, chat history

Question ids are unique within a user, not across the session — two users can each own a ``q1``.
Anything that names a question must therefore carry the owning ``user_id`` alongside it.

For agent reasoning, ``canonical_questions()`` collapses identical questions across users into a
single entry with an owner map. The model reasons once over that deduped view; the tools fan the
resulting action back out to every owner. This keeps inference cost flat as users are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.live.schemas_live.common import FactCheckVerdict, QuestionStatus
from src.live.services_live.events import (
    AgentEvaluationTriggered,
    AgendaRiskEmitted,
    ChatMessageAppended,
    CompanyDetected,
    FactCheckEmitted,
    MeetingStarted,
    PrepQuestionsLoaded,
    QuestionAnswered,
    QuestionGroupsResolved,
    PrepQuestionAnswered,
    UserPrepReplaced,
    StoredEvent,
    SuggestionEmitted,
    SuggestionRefined,
    QuestionAdded,
    TranscriptSegmentAppended,
    TranscriptSummaryUpdated,
)
from src.live.services_live.text_norm import normalize_text


@dataclass
class SessionUserState:
    """A coached user attached to this session."""

    user_id: str
    name: str = ""
    soa_connected: bool = False
    participant_uuid: str | None = None


@dataclass
class QuestionState:
    """A prep question and, once the meeting supplies one, its current answer.

    ``answer`` is the live record the coach reasons against, so a later contradiction updates it
    in place rather than appending. ``evidence_turn_seq`` is what makes re-answering safe: the
    same cited turn can never produce a second event, while a genuinely new answering turn always
    can. ``revision`` is what the wire uses to tell a first answer from a changed one.
    """

    id: str
    text: str
    status: QuestionStatus = QuestionStatus.PENDING
    # Verbatim transcript text of the answer. Kept named ``evidence`` because that is the field
    # the client renders; ``answer`` below is the same string under the name the agent reasons with.
    evidence: str | None = None
    evidence_turn_seq: int | None = None
    revision: int = 0

    @property
    def answer(self) -> str | None:
        return self.evidence


@dataclass
class CanonicalQuestion:
    """One logical question, deduped across the users who own a copy of it.

    ``owners`` maps ``user_id -> that user's question id``. ``status`` is ``pending`` while *any*
    owner still has it pending, so the agent keeps seeing it until every owner is marked.
    """

    id: str
    text: str
    status: QuestionStatus = QuestionStatus.PENDING
    evidence: str | None = None
    # Which transcript turn the current answer came from, and how many answers this question has
    # had. Both are shown to the model so it can tell a genuinely new answer from the one already
    # recorded, without comparing strings.
    evidence_turn_seq: int | None = None
    revision: int = 0
    owners: dict[str, str] = field(default_factory=dict)
    # Each owner's own wording of this question. Grouping decides who belongs together; keeping
    # their text lets the tool verify that decision instead of trusting it.
    owner_texts: dict[str, str] = field(default_factory=dict)

    @property
    def user_ids(self) -> list[str]:
        return list(self.owners)


@dataclass
class TranscriptTurn:
    speaker: str
    text: str
    timestamp: float
    turn_id: str | None = None
    seq: int | None = None
    speaker_is_host: bool = False
    participant_uuid: str | None = None
    start_ms: int | None = None
    duration_ms: int | None = None


@dataclass
class SuggestionState:
    id: str
    text: str
    user_id: str = ""
    rationale: str | None = None
    linked_question_id: str | None = None


@dataclass
class FactCheckState:
    id: str
    claim: str
    verdict: FactCheckVerdict
    confidence: float | None = None
    detail: str | None = None
    sources: list[str] = field(default_factory=list)


@dataclass
class AnswerState:
    id: str
    question: str
    answer: str
    source: str = "knowledge"


@dataclass
class RiskState:
    id: str
    text: str
    user_id: str | None = None


@dataclass
class CompanyState:
    id: str
    company_name: str
    title: str
    text: str
    es_id: str | None = None
    li_urn: str | None = None
    li_name: str | None = None
    li_universalname: str | None = None
    li_industries: list[str] = field(default_factory=list)


@dataclass
class ChatEntry:
    role: str
    id: str
    text: str


@dataclass
class EvaluationTriggerRecord:
    reason: str
    window_turn_count: int
    total_turn_count: int


@dataclass
class MeetingSession:
    bot_id: str = ""
    external_event_id: str | None = None
    meeting_type: str | None = None
    meeting_url: str | None = None
    started_at: str | None = None
    # Coached users, in roster order (insertion order is the order Node sent them).
    users: dict[str, SessionUserState] = field(default_factory=dict)
    # user_id -> prep (CRM) context. Never merged across users when generating broadcast content.
    context: dict[str, dict[str, Any]] = field(default_factory=dict)
    # user_id -> question_id -> state
    questions: dict[str, dict[str, QuestionState]] = field(default_factory=dict)
    # user_id -> ordered question ids
    question_order: dict[str, list[str]] = field(default_factory=dict)
    # Semantic grouping of the same question across users, resolved in the background.
    # Empty until it lands; canonical_questions() falls back to exact text matching until then.
    question_groups: list[list[tuple[str, str]]] = field(default_factory=list)
    transcript: list[TranscriptTurn] = field(default_factory=list)
    suggestions: list[SuggestionState] = field(default_factory=list)
    fact_checks: list[FactCheckState] = field(default_factory=list)
    # Questions asked out loud and answered. Kept so the answerer never repeats itself.
    answers: list[AnswerState] = field(default_factory=list)
    risks: list[RiskState] = field(default_factory=list)
    companies: list[CompanyState] = field(default_factory=list)
    # user_id -> that user's private chat thread
    chat: dict[str, list[ChatEntry]] = field(default_factory=dict)
    evaluation_triggers: list[EvaluationTriggerRecord] = field(default_factory=list)
    transcript_summary: str | None = None
    compacted_turn_count: int = 0
    # Bumped whenever the canonical question set could renumber (prep loaded/replaced, a question
    # added, grouping resolved). Canonical ids are positional, so an evaluation that built its
    # context before a change and applies after it would resolve ``c2`` against a different
    # question. The orchestrator compares this to decide whether its ids are still meaningful.
    questions_version: int = 0
    last_offset: int = -1

    @property
    def meeting_id(self) -> str:
        """Alias for ``bot_id`` (session correlation key)."""
        return self.bot_id

    @property
    def user_ids(self) -> list[str]:
        return list(self.users)

    def has_user(self, user_id: str) -> bool:
        return user_id in self.users

    def context_for(self, user_id: str) -> dict[str, Any]:
        return self.context.get(user_id, {})

    def questions_for(self, user_id: str) -> dict[str, QuestionState]:
        return self.questions.get(user_id, {})

    def ordered_questions_for(self, user_id: str) -> list[QuestionState]:
        by_id = self.questions.get(user_id, {})
        return [by_id[qid] for qid in self.question_order.get(user_id, []) if qid in by_id]

    def chat_for(self, user_id: str) -> list[ChatEntry]:
        return self.chat.get(user_id, [])

    def suggestions_for(self, user_id: str) -> list[SuggestionState]:
        return [s for s in self.suggestions if s.user_id == user_id]

    def risks_for(self, user_id: str) -> list[RiskState]:
        """Risks this user should see: their own plus every broadcast risk."""
        return [r for r in self.risks if r.user_id is None or r.user_id == user_id]

    def canonical_questions(self) -> list[CanonicalQuestion]:
        """Collapse the same question across users into one entry the agent reasons about once.

        Grouping is **semantic** once ``question_groups`` has been resolved in the background:
        two users whose prep worded a question differently still get one canonical entry. Until
        that lands (and if it fails) this falls back to exact normalized-text matching, which
        over-splits rather than wrongly merging — the safe direction to be wrong in.

        Canonical ids are ``c1..cN`` in stable first-seen roster order.
        """
        if self.question_groups:
            return self._canonical_from_groups()
        return self._canonical_from_exact_text()

    def _canonical_from_groups(self) -> list[CanonicalQuestion]:
        canonicals: list[CanonicalQuestion] = []
        grouped: set[tuple[str, str]] = set()
        for members in self.question_groups:
            canonical: CanonicalQuestion | None = None
            for user_id, question_id in members:
                question = self.questions_for(user_id).get(question_id)
                if question is None:
                    continue
                grouped.add((user_id, question_id))
                if canonical is None:
                    canonical = CanonicalQuestion(
                        id=f"c{len(canonicals) + 1}",
                        text=question.text,
                        status=question.status,
                        evidence=question.evidence,
                        evidence_turn_seq=question.evidence_turn_seq,
                        revision=question.revision,
                    )
                    canonicals.append(canonical)
                canonical.owners[user_id] = question_id
                canonical.owner_texts[user_id] = question.text
                if question.status == QuestionStatus.PENDING:
                    canonical.status = QuestionStatus.PENDING
                if question.revision > canonical.revision:
                    canonical.evidence = question.evidence
                    canonical.evidence_turn_seq = question.evidence_turn_seq
                    canonical.revision = question.revision

        # Questions added after grouping was resolved stand on their own.
        for user_id in self.users:
            for question in self.ordered_questions_for(user_id):
                if (user_id, question.id) in grouped:
                    continue
                canonicals.append(
                    CanonicalQuestion(
                        id=f"c{len(canonicals) + 1}",
                        text=question.text,
                        status=question.status,
                        evidence=question.evidence,
                        evidence_turn_seq=question.evidence_turn_seq,
                        revision=question.revision,
                        owners={user_id: question.id},
                        owner_texts={user_id: question.text},
                    )
                )
        return canonicals

    def _canonical_from_exact_text(self) -> list[CanonicalQuestion]:
        by_key: dict[str, CanonicalQuestion] = {}
        for user_id in self.users:
            for question in self.ordered_questions_for(user_id):
                key = normalize_text(question.text)
                if not key:
                    continue
                canonical = by_key.get(key)
                if canonical is None:
                    canonical = CanonicalQuestion(
                        id=f"c{len(by_key) + 1}",
                        text=question.text,
                        status=question.status,
                        evidence=question.evidence,
                        evidence_turn_seq=question.evidence_turn_seq,
                        revision=question.revision,
                    )
                    by_key[key] = canonical
                canonical.owners[user_id] = question.id
                canonical.owner_texts[user_id] = question.text
                # Pending for anyone => pending for the agent, so it keeps chasing the stragglers.
                if question.status == QuestionStatus.PENDING:
                    canonical.status = QuestionStatus.PENDING
                if question.revision > canonical.revision:
                    canonical.evidence = question.evidence
                    canonical.evidence_turn_seq = question.evidence_turn_seq
                    canonical.revision = question.revision
        return list(by_key.values())

    def canonical_question(self, canonical_id: str) -> CanonicalQuestion | None:
        for canonical in self.canonical_questions():
            if canonical.id == canonical_id:
                return canonical
        return None

    def all_question_texts(self) -> list[str]:
        """Every question text across every user (used for duplicate suppression)."""
        return [q.text for user_id in self.users for q in self.ordered_questions_for(user_id)]

    def drop_user_from_question_groups(self, user_id: str) -> None:
        """Remove one user from the semantic grouping, keeping other users' grouping intact.

        Their questions then fall through to the "ungrouped stands alone" path — an over-split,
        never a wrong merge — until background grouping re-resolves.
        """
        remaining: list[list[tuple[str, str]]] = []
        for group in self.question_groups:
            members = [(u, q) for (u, q) in group if u != user_id]
            if members:
                remaining.append(members)
        self.question_groups = remaining

    def turn_by_seq(self, seq: int | None) -> TranscriptTurn | None:
        """Look up the transcript turn the agent cited as evidence.

        Evidence is a reference to a real turn, so verifying it is an existence check — never a
        similarity score against text the model retyped.
        """
        if seq is None:
            return None
        for turn in reversed(self.transcript):
            if turn.seq == seq:
                return turn
        return None

    def recent_transcript(self, window: int) -> list[TranscriptTurn]:
        from src.live.services_live.memory import recent_transcript as _recent

        return _recent(self.transcript, window)

    def prompt_transcript(self, window: int) -> tuple[str | None, list[TranscriptTurn]]:
        """Return (earlier_summary, recent_turns) for LLM prompts after compaction."""
        from src.live.services_live.memory import prompt_transcript as _prompt

        return _prompt(self, window)

    def apply(self, stored: StoredEvent) -> None:
        event = stored.event
        if isinstance(event, MeetingStarted):
            self.bot_id = event.bot_id
            self.external_event_id = event.external_event_id
            self.meeting_type = event.meeting_type
            self.meeting_url = event.meeting_url
            self.started_at = event.started_at
            for record in event.users:
                self.users[record.user_id] = SessionUserState(
                    user_id=record.user_id,
                    name=record.name,
                    soa_connected=record.soa_connected,
                    participant_uuid=record.participant_uuid,
                )
                self.questions.setdefault(record.user_id, {})
                self.question_order.setdefault(record.user_id, [])
                self.context.setdefault(record.user_id, {})
                self.chat.setdefault(record.user_id, [])
        elif isinstance(event, PrepQuestionsLoaded):
            self.context[event.user_id] = dict(event.context)
            by_id = self.questions.setdefault(event.user_id, {})
            order = self.question_order.setdefault(event.user_id, [])
            for q in event.questions:
                if q.id not in by_id:
                    order.append(q.id)
                by_id[q.id] = QuestionState(id=q.id, text=q.text)
            self.questions_version += 1
        elif isinstance(event, UserPrepReplaced):
            # A genuine replace: stale questions are dropped rather than left behind, and every
            # question comes back unmarked.
            self.context[event.user_id] = dict(event.context)
            self.questions[event.user_id] = {
                q.id: QuestionState(id=q.id, text=q.text) for q in event.questions
            }
            self.question_order[event.user_id] = [q.id for q in event.questions]
            # Old grouping coordinates now point at different text — strip this user out of it.
            self.drop_user_from_question_groups(event.user_id)
            # Links pointed at question ids that may no longer mean the same thing.
            for suggestion in self.suggestions:
                if suggestion.user_id == event.user_id:
                    suggestion.linked_question_id = None
            self.questions_version += 1
        elif isinstance(event, TranscriptSegmentAppended):
            self.transcript.append(
                TranscriptTurn(
                    speaker=event.speaker,
                    text=event.text,
                    timestamp=event.timestamp,
                    turn_id=event.turn_id,
                    seq=event.seq,
                    speaker_is_host=event.speaker_is_host,
                    participant_uuid=event.participant_uuid,
                    start_ms=event.start_ms,
                    duration_ms=event.duration_ms,
                )
            )
        elif isinstance(event, QuestionGroupsResolved):
            self.question_groups = [
                [(m.user_id, m.question_id) for m in group] for group in event.groups
            ]
            # Collapsing questions renumbers the canonical set.
            self.questions_version += 1
        elif isinstance(event, PrepQuestionAnswered):
            q = self.questions.get(event.user_id, {}).get(event.question_id)
            if q is not None:
                # A revision replaces the answer rather than appending: the coach reasons about
                # what is true *now*, and a stale figure alongside a current one is worse than
                # either alone.
                q.status = QuestionStatus.ANSWERED
                q.evidence = event.answer
                q.evidence_turn_seq = event.evidence_turn_seq
                q.revision = event.revision
        elif isinstance(event, SuggestionEmitted):
            self.suggestions.append(
                SuggestionState(
                    id=event.suggestion_id,
                    user_id=event.user_id,
                    text=event.text,
                    rationale=event.rationale,
                    linked_question_id=event.linked_question_id,
                )
            )
        elif isinstance(event, SuggestionRefined):
            for s in self.suggestions:
                if s.id == event.suggestion_id and s.user_id == event.user_id:
                    s.text = event.text
                    if event.rationale is not None:
                        s.rationale = event.rationale
                    break
        elif isinstance(event, QuestionAdded):
            by_id = self.questions.setdefault(event.user_id, {})
            order = self.question_order.setdefault(event.user_id, [])
            if event.question_id not in by_id:
                order.append(event.question_id)
            by_id[event.question_id] = QuestionState(
                id=event.question_id,
                text=event.text,
            )
            # Appending to one user's list shifts every later user's canonical id.
            self.questions_version += 1
        elif isinstance(event, FactCheckEmitted):
            self.fact_checks.append(
                FactCheckState(
                    id=event.fact_check_id,
                    claim=event.claim,
                    verdict=event.verdict,
                    confidence=event.confidence,
                    detail=event.detail,
                    sources=list(event.sources),
                )
            )
        elif isinstance(event, QuestionAnswered):
            self.answers.append(
                AnswerState(
                    id=event.answer_id,
                    question=event.question,
                    answer=event.answer,
                    source=event.source,
                )
            )
        elif isinstance(event, AgendaRiskEmitted):
            self.risks.append(
                RiskState(id=event.risk_id, text=event.text, user_id=event.user_id)
            )
        elif isinstance(event, CompanyDetected):
            self.companies.append(
                CompanyState(
                    id=event.company_id,
                    company_name=event.company_name,
                    title=event.title,
                    text=event.text,
                    es_id=event.es_id,
                    li_urn=event.li_urn,
                    li_name=event.li_name,
                    li_universalname=event.li_universalname,
                    li_industries=list(event.li_industries),
                )
            )
        elif isinstance(event, ChatMessageAppended):
            thread = self.chat.setdefault(event.user_id, [])
            thread.append(ChatEntry(role=event.role, id=event.message_id, text=event.text))
        elif isinstance(event, AgentEvaluationTriggered):
            self.evaluation_triggers.append(
                EvaluationTriggerRecord(
                    reason=event.reason,
                    window_turn_count=event.window_turn_count,
                    total_turn_count=event.total_turn_count,
                )
            )
        elif isinstance(event, TranscriptSummaryUpdated):
            self.transcript_summary = event.summary_text
            self.compacted_turn_count = event.compacted_turn_count

        self.last_offset = stored.offset


def reduce(events: Iterable[StoredEvent]) -> MeetingSession:
    """Build a fresh ``MeetingSession`` by folding events in order."""
    session = MeetingSession()
    for stored in events:
        session.apply(stored)
    return session

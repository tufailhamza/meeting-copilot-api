"""MeetingActor: the single-writer owner of one meeting's state.

One actor serves the whole meeting — every coached user shares its transcript, its agent loop, and
its socket. Two invariants hold it together:

1. **Single writer.** All session mutation happens on the mailbox loop. Slow work (LLM calls) runs
   off the loop in tasks, but the resulting events are posted *back* to the mailbox rather than
   applied in place, so concurrent chat turns can never interleave state writes.
2. **Explicit addressing.** Nothing infers a recipient. Chat replies go to the user id on the
   inbound frame; agent output is addressed by the tools that produced it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

from src.live.config_live import live_settings
from src.live.schemas_live.common import new_id
from src.live.schemas_live.inbound import (
    ChatMessageIn,
    TranscriptSegmentMessage,
    TranscriptSessionEndMessage,
    TranscriptSessionStartMessage,
    UserConnectedOnLiveScreen,
)
from src.live.schemas_live.outbound import (
    ChatMessageOut,
    ChatToken,
    ErrorMessage,
)
from src.live.services_live.chat import ChatAgent, build_default_chat_agent
from src.live.services_live.chat.web_search import run_web_search
from src.live.services_live.compaction import MemoryCompactor, build_default_compactor
from src.live.services_live.event_bus import EventBus
from src.live.services_live.event_log import EventLog
from src.live.services_live.events import (
    AgentEvaluationTriggered,
    ChatMessageAppended,
    EventModel,
    MeetingEvent,
    MeetingStarted,
    PrepQuestionsLoaded,
    SessionUserRecord,
    QuestionAnswered,
    TranscriptSegmentAppended,
    UserPrepReplaced,
)
from src.live.services_live.agent.question_answerer import (
    QuestionAnswerer,
    build_question_answerer,
)
from src.live.services_live.agent.question_grouper import (
    QuestionGrouper,
    build_grouping_event,
    build_question_grouper,
    encode_key,
)
from src.live.services_live.memory import recent_transcript
from src.live.services_live.orchestrator import (
    AgentOrchestrator,
    build_default_orchestrator,
    snapshot_for_company_resolution,
)
from src.live.services_live.prep_loader import PrepLoader, PrepUser
from src.live.services_live.projection import MeetingSession
from src.live.services_live.trigger import EvaluationTriggerPolicy, TriggerConfig, TriggerReason
from src.live.services_live.trace_log import log_inbound, log_outbound
from src.live.services_live.wire import domain_event_to_wire

logger = logging.getLogger(__name__)

# How much conversation the answering lane looks at. Deliberately tiny: it only needs to see the
# question and whether anyone answered it, and a small prompt is what makes per-segment affordable.
ANSWER_WINDOW_TURNS = 6


class _EvaluationTimerFired:
    """Internal mailbox sentinel: the periodic evaluation tick elapsed."""


class _PrepLoadComplete:
    """Internal mailbox sentinel: every user's prep has been queued, so the session is ready."""


@dataclass
class _EvaluationInferred:
    """Internal mailbox sentinel: inference finished off-loop and needs applying in order."""

    outcome: Any


@dataclass
class _EventsReady:
    """Internal mailbox sentinel: events produced off-loop (e.g. resolved companies)."""

    events: tuple[MeetingEvent, ...] = ()
    clears_compaction: bool = False
    clears_prep_reload_for: str | None = None
    clears_grouping: bool = False
    clears_answer: bool = False
    regroup_questions: bool = False


@dataclass
class _ChatTurnCompleted:
    """Internal mailbox sentinel: a chat turn finished off-loop and needs applying in order.

    Carrying the result back through the mailbox keeps the single-writer invariant and pins the
    client-visible ordering: tokens, then any tool events, then the final message.
    """

    user_id: str
    assistant_id: str
    reply: str
    tool_events: tuple[MeetingEvent, ...] = ()
    additional_data: dict[str, Any] | None = None


@dataclass
class _BackgroundTasks:
    """Off-loop work in flight, so shutdown can cancel it deterministically."""

    tasks: set[asyncio.Task] = field(default_factory=set)

    def track(self, task: asyncio.Task) -> None:
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def cancel_all(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        for task in list(self.tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.tasks.clear()


class MeetingActor:
    def __init__(
        self,
        bot_id: str,
        event_log: EventLog,
        prep_loader: PrepLoader | None = None,
        mailbox_maxsize: int = 0,
        *,
        trigger_config: TriggerConfig | None = None,
        memory_window_turns: int | None = None,
        orchestrator: AgentOrchestrator | None | object = ...,
        chat_agent: ChatAgent | None | object = ...,
        memory_compactor: MemoryCompactor | None | object = ...,
        question_grouper: QuestionGrouper | None | object = ...,
        question_answerer: QuestionAnswerer | None | object = ...,
        event_bus: EventBus | None = None,
    ) -> None:
        self.bot_id = bot_id
        self.meeting_id = bot_id  # alias for event-log key and backward compat
        self._log = event_log
        self._prep_loader = prep_loader
        self._memory_window = memory_window_turns or live_settings.memory_window_turns
        self._trigger = EvaluationTriggerPolicy(
            trigger_config
            or TriggerConfig(
                min_turns=live_settings.trigger_min_turns,
                min_seconds=live_settings.trigger_min_seconds,
            )
        )
        self._mailbox: asyncio.Queue = asyncio.Queue(maxsize=mailbox_maxsize)
        self.session = MeetingSession(bot_id=bot_id)
        self.ready = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._prep_task: asyncio.Task | None = None
        self._eval_ticker: asyncio.Task | None = None
        self._background = _BackgroundTasks()
        # Single-flight evaluation: at most one inference in flight, with at most one queued
        # behind it. Triggers arriving during an evaluation coalesce into that single re-run
        # instead of stacking up, so the agent self-paces at however fast it can actually think.
        self._eval_inflight = False
        self._eval_pending = False
        self._compaction_inflight = False
        # Single-flight per user: a second connect event while a reload is running collapses into
        # one follow-up, so two loads can never land out of order and reinstate stale prep.
        self._prep_reload_inflight: set[str] = set()
        self._prep_reload_pending: set[str] = set()
        self._grouping_inflight = False
        self._grouping_pending = False
        # The answering lane runs per segment, so it needs its own coalescing rather than
        # sharing the coaching loop's.
        self._answer_inflight = False
        self._answer_pending = False
        self._session_ended = False
        if orchestrator is ...:
            orchestrator = build_default_orchestrator(memory_window_turns=self._memory_window)
        self._orchestrator: AgentOrchestrator | None = orchestrator  # type: ignore[assignment]
        if chat_agent is ...:
            chat_agent = build_default_chat_agent()
        self._chat_agent: ChatAgent | None = chat_agent  # type: ignore[assignment]
        if memory_compactor is ...:
            memory_compactor = build_default_compactor(memory_window_turns=self._memory_window)
        self._memory_compactor: MemoryCompactor | None = memory_compactor  # type: ignore[assignment]
        if question_grouper is ...:
            question_grouper = build_question_grouper()
        self._question_grouper: QuestionGrouper | None = question_grouper  # type: ignore[assignment]
        if question_answerer is ...:
            question_answerer = build_question_answerer()
        self._question_answerer: QuestionAnswerer | None = question_answerer  # type: ignore[assignment]
        self._event_bus = event_bus

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"meeting-actor-{self.bot_id}")

    async def submit(self, message) -> None:
        await self._mailbox.put(message)

    async def join(self) -> None:
        """Wait until the actor is fully idle.

        Off-loop work (inference, chat turns, company lookups) posts its results *back* to the
        mailbox, so draining once is not enough — an empty mailbox with a task still running is
        not quiescent. Loop until both the mailbox and the background set are quiet.
        """
        while True:
            await self._mailbox.join()
            outstanding = list(self._background.tasks)
            if (
                not outstanding
                and not self._eval_inflight
                and not self._compaction_inflight
                and not self._prep_reload_inflight
                and not self._grouping_inflight
                and not self._answer_inflight
            ):
                return
            await asyncio.gather(*outstanding, return_exceptions=True)

    async def stop(self) -> None:
        self._cancel_eval_ticker()
        await self._background.cancel_all()
        if self._prep_task is not None:
            self._prep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._prep_task
            self._prep_task = None
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            item = await self._mailbox.get()
            try:
                await self._handle(item)
            except Exception:  # noqa: BLE001
                logger.exception("[live] actor %s failed handling an item", self.bot_id)
            finally:
                self._mailbox.task_done()

    async def _handle(self, item) -> None:
        if isinstance(item, _EvaluationTimerFired):
            reason = self._trigger.on_timer_expired()
            if reason is not None:
                await self._fire_evaluation(reason)
            return
        if isinstance(item, _PrepLoadComplete):
            # Ack first: grouping is an optimisation and must never delay the handshake.
            self.ready.set()
            self._maybe_group_questions()
            return
        if isinstance(item, _EvaluationInferred):
            await self._apply_evaluation(item.outcome)
            return
        if isinstance(item, _EventsReady):
            if item.clears_compaction:
                self._compaction_inflight = False
            if item.clears_prep_reload_for is not None:
                self._prep_reload_inflight.discard(item.clears_prep_reload_for)
            if item.clears_grouping:
                self._grouping_inflight = False
            if item.clears_answer:
                self._answer_inflight = False
            for event in item.events:
                await self._append(event)
            if item.regroup_questions or self._grouping_pending:
                # Question texts changed, so the old grouping no longer describes this set.
                self._grouping_pending = False
                self._maybe_group_questions()
            if item.clears_answer and self._answer_pending:
                self._answer_pending = False
                self._request_answer_pass()
            if item.clears_prep_reload_for in self._prep_reload_pending:
                self._prep_reload_pending.discard(item.clears_prep_reload_for)
                self._request_prep_reload(item.clears_prep_reload_for)
            return
        if isinstance(item, _ChatTurnCompleted):
            await self._apply_chat_turn(item)
            return
        if isinstance(item, EventModel):
            await self._append(item)
            return
        if isinstance(item, TranscriptSessionStartMessage):
            await self._on_session_start(item)
            return
        if isinstance(item, TranscriptSessionEndMessage):
            self._session_ended = True
            logger.info("[live] session end bot_id=%s", self.bot_id)
            return
        if isinstance(item, ChatMessageIn):
            await self._on_chat_message(item)
            return
        if isinstance(item, UserConnectedOnLiveScreen):
            self._on_user_connected(item)
            return
        event = self._to_event(item)
        if event is not None:
            if isinstance(item, TranscriptSegmentMessage):
                log_inbound(
                    self.bot_id,
                    "transcript_segment",
                    seq=item.segment.seq,
                    speaker=item.segment.speaker,
                )
            await self._append(event)

    async def _on_session_start(self, message: TranscriptSessionStartMessage) -> None:
        self._start_eval_ticker()
        await self._append(
            MeetingStarted(
                bot_id=message.bot_id,
                external_event_id=message.event_id,
                users=[
                    SessionUserRecord(
                        user_id=u.user_id,
                        name=u.name,
                        soa_connected=u.soa_connected,
                        participant_uuid=u.participant_uuid,
                    )
                    for u in message.users
                ],
                meeting_type=message.meeting_type,
                meeting_url=message.meeting_url,
                started_at=message.started_at,
            )
        )
        if message.event_id and self._prep_loader is not None:
            logger.info(
                "[live] prep load started bot_id=%s external_event_id=%s users=%s",
                self.bot_id,
                message.event_id,
                [(u.user_id, u.soa_connected) for u in message.users],
            )
            self._prep_task = asyncio.create_task(
                self._load_prep(
                    message.event_id,
                    [
                        PrepUser(user_id=u.user_id, soa_connected=u.soa_connected)
                        for u in message.users
                    ],
                ),
                name=f"prep-load-{self.bot_id}",
            )
        else:
            logger.info(
                "[live] prep skipped bot_id=%s event_id=%s loader=%s",
                self.bot_id,
                message.event_id,
                self._prep_loader is not None,
            )
            self.ready.set()

    async def _load_prep(self, external_event_id: str, users: list[PrepUser]) -> None:
        """Load every user's prep off-loop, then post the results back for ordered application."""
        try:
            prep_by_user = await self._prep_loader.load_for_users(external_event_id, users)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[live] prep load failed bot_id=%s external_event_id=%s",
                self.bot_id,
                external_event_id,
            )
            await self._mailbox.put(_PrepLoadComplete())
            return

        for user in users:
            prep = prep_by_user.get(user.user_id)
            if prep is None:
                logger.info(
                    "[live] prep empty bot_id=%s external_event_id=%s user_id=%s",
                    self.bot_id,
                    external_event_id,
                    user.user_id,
                )
                continue
            logger.info(
                "[live] prep loaded bot_id=%s external_event_id=%s user_id=%s questions=%s "
                "context_keys=%s",
                self.bot_id,
                external_event_id,
                user.user_id,
                len(prep.questions),
                sorted(prep.context.keys()),
            )
            await self._mailbox.put(
                PrepQuestionsLoaded(
                    user_id=user.user_id,
                    questions=prep.questions,
                    context=prep.context,
                )
            )
        # FIFO mailbox: this lands after every prep event above has been applied.
        await self._mailbox.put(_PrepLoadComplete())

    def _on_user_connected(self, message: UserConnectedOnLiveScreen) -> None:
        """Refresh one user's prep because they just opened their live screen.

        Everything here is validated against session state before any DB work: the roster is
        already complete from session start, so an unknown user id is a client error rather than a
        late joiner, and an event id that disagrees with this session's means the client has
        crossed two meetings.
        """
        log_inbound(
            self.bot_id,
            "user_connected_on_live_screen",
            user_id=message.user_id,
            soa_connected=message.soa_connected,
        )
        if self._session_ended:
            return
        user = self.session.users.get(message.user_id)
        if user is None:
            logger.warning(
                "[live] live-screen connect for unknown user_id=%s bot_id=%s roster=%s",
                message.user_id,
                self.bot_id,
                self.session.user_ids,
            )
            return
        if not self.session.external_event_id:
            logger.info(
                "[live] live-screen connect ignored bot_id=%s user_id=%s reason=no_event_id",
                self.bot_id,
                message.user_id,
            )
            return
        if message.event_id.strip() != self.session.external_event_id:
            logger.warning(
                "[live] live-screen connect event_id mismatch bot_id=%s user_id=%s "
                "frame=%r session=%r — ignoring",
                self.bot_id,
                message.user_id,
                message.event_id,
                self.session.external_event_id,
            )
            return

        # The frame is authoritative about SOA status: it decides which table future loads read.
        if user.soa_connected != message.soa_connected:
            logger.info(
                "[live] live-screen connect updates soa_connected bot_id=%s user_id=%s %s->%s",
                self.bot_id,
                message.user_id,
                user.soa_connected,
                message.soa_connected,
            )
            user.soa_connected = message.soa_connected

        self._request_prep_reload(message.user_id)

    def _request_prep_reload(self, user_id: str) -> None:
        if self._prep_loader is None or self._session_ended:
            return
        if user_id in self._prep_reload_inflight:
            self._prep_reload_pending.add(user_id)
            return
        user = self.session.users.get(user_id)
        if user is None:
            return
        self._prep_reload_inflight.add(user_id)
        task = asyncio.create_task(
            self._reload_prep(user_id, user.soa_connected, self.session.external_event_id or ""),
            name=f"prep-reload-{self.bot_id}-{user_id}",
        )
        self._background.track(task)

    async def _reload_prep(self, user_id: str, soa_connected: bool, event_id: str) -> None:
        """Re-read one user's prep off the loop and post the replacement back."""
        try:
            prep_by_user = await self._prep_loader.load_for_users(
                event_id, [PrepUser(user_id=user_id, soa_connected=soa_connected)]
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "[live] prep reload failed bot_id=%s user_id=%s", self.bot_id, user_id
            )
            await self._mailbox.put(_EventsReady(clears_prep_reload_for=user_id))
            return

        prep = prep_by_user.get(user_id)
        if prep is None:
            # Absent means "no row" *or* a swallowed DB error — indistinguishable. Never wipe good
            # questions on an ambiguous result; leave what the session already has.
            logger.info(
                "[live] prep reload found nothing bot_id=%s user_id=%s — keeping existing prep",
                self.bot_id,
                user_id,
            )
            await self._mailbox.put(_EventsReady(clears_prep_reload_for=user_id))
            return

        logger.info(
            "[live] prep reload replacing bot_id=%s user_id=%s questions=%s",
            self.bot_id,
            user_id,
            len(prep.questions),
        )
        await self._mailbox.put(
            _EventsReady(
                events=(
                    UserPrepReplaced(
                        user_id=user_id,
                        questions=prep.questions,
                        context=prep.context,
                    ),
                ),
                clears_prep_reload_for=user_id,
                regroup_questions=True,
            )
        )

    def _request_answer_pass(self) -> None:
        """Look at the tail of the conversation for a question worth answering."""
        if self._question_answerer is None or self._session_ended:
            return
        if self._answer_inflight:
            self._answer_pending = True
            return
        # Snapshot on the loop; the off-loop task never reads live session state.
        recent = [
            {"speaker": t.speaker, "text": t.text}
            for t in recent_transcript(self.session.transcript, ANSWER_WINDOW_TURNS)
        ]
        if not recent:
            return
        already = [a.question for a in self.session.answers]
        # Every coached user's prep. A question the host planned to ask must never be answered for
        # them — the prospect's answer is the whole point of asking it.
        prep = [
            q.text
            for user_id in self.session.user_ids
            for q in self.session.ordered_questions_for(user_id)
        ]
        self._answer_inflight = True
        task = asyncio.create_task(
            self._run_answer_pass(recent, already, prep),
            name=f"answer-{self.bot_id}",
        )
        self._background.track(task)

    async def _run_answer_pass(
        self,
        recent: list[dict[str, str]],
        already_answered: list[str],
        prep_questions: list[str] | None = None,
    ) -> None:
        events: tuple[MeetingEvent, ...] = ()
        try:
            decision = await self._question_answerer.detect_and_answer(
                recent, already_answered, prep_questions
            )
            # A question the model itself does not call world knowledge is one it should not be
            # answering. Without this, a discovery question aimed at a participant came back with
            # "that is not answerable by me" in the answer field and was pushed to the host.
            if decision.answerable_question and not decision.is_general_world_knowledge:
                logger.info(
                    "[live] answer suppressed (not world knowledge) bot_id=%s question=%r",
                    self.bot_id,
                    decision.answerable_question,
                )
            elif decision.answerable_question:
                answer, source = await self._resolve_answer(decision)
                if answer:
                    events = (
                        QuestionAnswered(
                            answer_id=new_id("ans"),
                            question=decision.answerable_question,
                            answer=answer,
                            source=source,
                        ),
                    )
                    logger.info(
                        "[live] answered question bot_id=%s source=%s question=%r",
                        self.bot_id,
                        source,
                        decision.answerable_question,
                    )
            elif decision.skip_reason:
                logger.debug(
                    "[live] answer pass skip bot_id=%s reason=%s",
                    self.bot_id,
                    decision.skip_reason,
                )
        except asyncio.CancelledError:
            self._answer_inflight = False
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[live] answer pass failed bot_id=%s", self.bot_id)
        await self._mailbox.put(_EventsReady(events=events, clears_answer=True))

    async def _resolve_answer(self, decision) -> tuple[str | None, str]:
        """Return (answer, source). Stable facts answer directly; current ones need a lookup."""
        if not decision.needs_live_data:
            return ((decision.answer or "").strip() or None, "knowledge")
        query = (decision.search_query or decision.answerable_question or "").strip()
        if not query:
            return ((decision.answer or "").strip() or None, "knowledge")
        try:
            result = await run_web_search(query, self.session)
        except Exception:  # noqa: BLE001
            logger.exception("[live] answer web lookup failed bot_id=%s", self.bot_id)
            # Fall back to whatever the model already had rather than dropping the question.
            return ((decision.answer or "").strip() or None, "knowledge")
        findings = (result.reply or "").strip()
        if not findings:
            return ((decision.answer or "").strip() or None, "knowledge")
        composed = await self._question_answerer.compose_from_findings(
            decision.answerable_question, findings
        )
        return ((composed or findings).strip() or None, "web")

    def _maybe_group_questions(self) -> None:
        """Ask the model which prep questions are the same, once, in the background.

        Only worth doing when more than one user brought questions — with a single user there is
        nothing to collapse. Runs after ``session_ready`` so it costs the host nothing, and the
        session keeps using exact-text grouping until (or unless) it lands.
        """
        if self._question_grouper is None or self._session_ended:
            return
        if self._grouping_inflight:
            # A reload landed mid-grouping; the result would describe a stale set, so queue one.
            self._grouping_pending = True
            return
        questions: dict[str, str] = {}
        users_with_questions = 0
        for user_id in self.session.user_ids:
            owned = self.session.ordered_questions_for(user_id)
            if owned:
                users_with_questions += 1
            for question in owned:
                questions[encode_key(user_id, question.id)] = question.text
        if users_with_questions < 2 or len(questions) < 2:
            return
        self._grouping_inflight = True
        task = asyncio.create_task(
            self._group_questions(questions),
            name=f"question-grouping-{self.bot_id}",
        )
        self._background.track(task)

    async def _group_questions(self, questions: dict[str, str]) -> None:
        known: dict[str, tuple[str, str]] = {}
        for key in questions:
            user_id, _, question_id = key.partition("::")
            known[key] = (user_id, question_id)
        try:
            grouping = await self._question_grouper.group(questions)
        except asyncio.CancelledError:
            self._grouping_inflight = False
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "[live] question grouping failed bot_id=%s (falling back to exact text)",
                self.bot_id,
            )
            await self._mailbox.put(_EventsReady(clears_grouping=True))
            return
        event = build_grouping_event(grouping, known)
        logger.info(
            "[live] question grouping resolved bot_id=%s questions=%s groups=%s",
            self.bot_id,
            len(questions),
            len(event.groups),
        )
        await self._mailbox.put(_EventsReady(events=(event,), clears_grouping=True))

    async def _append(self, event: MeetingEvent) -> None:
        stored = await self._log.append(self.bot_id, event)
        self.session.apply(stored)
        await self._publish_wire_event(event)
        if isinstance(event, PrepQuestionsLoaded):
            logger.info(
                "[live] prep applied bot_id=%s user_id=%s questions=%s",
                self.bot_id,
                event.user_id,
                len(event.questions),
            )
        elif isinstance(event, TranscriptSegmentAppended):
            self._maybe_compact_memory()
            # Answering runs on every segment, not on the coaching cadence: an answer that lands
            # three turns after the question is useless.
            self._request_answer_pass()
            await self._on_transcript_segment()
        elif isinstance(event, ChatMessageAppended) and event.role == "user":
            self._spawn_chat_turn(event)

    def _maybe_compact_memory(self) -> None:
        """Plan on the loop, summarize off it.

        Compaction makes its own model call. Awaiting that inline on every transcript segment put
        an LLM round trip directly in the ingestion path.
        """
        if self._memory_compactor is None or self._session_ended or self._compaction_inflight:
            return
        plan = self._memory_compactor.plan(self.session)
        if plan is None:
            return
        self._compaction_inflight = True
        task = asyncio.create_task(
            self._run_compaction(plan, self.session.last_offset),
            name=f"compaction-{self.bot_id}",
        )
        self._background.track(task)

    async def _run_compaction(self, plan, event_offset: int) -> None:
        try:
            event = await self._memory_compactor.run_plan(self.bot_id, plan, event_offset)
        except asyncio.CancelledError:
            self._compaction_inflight = False
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[live] memory compaction failed bot_id=%s", self.bot_id)
            self._compaction_inflight = False
            return
        events = (event,) if event is not None else ()
        await self._mailbox.put(_EventsReady(events=events, clears_compaction=True))

    async def _on_chat_message(self, message: ChatMessageIn) -> None:
        """Validate the sender is on the roster, then record the turn against their thread."""
        log_inbound(self.bot_id, "chat_message", text=message.text)
        if not self.session.has_user(message.user_id):
            logger.warning(
                "[live] chat from unknown user_id=%s bot_id=%s roster=%s",
                message.user_id,
                self.bot_id,
                self.session.user_ids,
            )
            await self._publish_outbound(
                ErrorMessage(
                    user_id=message.user_id,
                    code="unknown_user",
                    message="userId is not part of this session",
                )
            )
            return
        await self._append(
            ChatMessageAppended(
                user_id=message.user_id,
                role="user",
                message_id=message.id or new_id("msg"),
                text=message.text,
            )
        )

    def _spawn_chat_turn(self, user_msg: ChatMessageAppended) -> None:
        """Run a chat turn off the mailbox loop.

        Users must not queue behind each other's model calls, and transcript ingestion must not
        stall behind chat at all — so the slow part runs concurrently and only the resulting
        events return to the loop.
        """
        if self._chat_agent is None or self._session_ended:
            logger.info(
                "[live] chat skipped bot_id=%s user_id=%s agent=%s session_ended=%s",
                self.bot_id,
                user_msg.user_id,
                self._chat_agent is not None,
                self._session_ended,
            )
            return
        task = asyncio.create_task(
            self._run_chat_turn(user_msg),
            name=f"chat-turn-{self.bot_id}-{user_msg.user_id}",
        )
        self._background.track(task)

    async def _run_chat_turn(self, user_msg: ChatMessageAppended) -> None:
        user_id = user_msg.user_id
        logger.info(
            "[live] chat turn started bot_id=%s user_id=%s message_id=%s",
            self.bot_id,
            user_id,
            user_msg.message_id,
        )
        assistant_id = new_id("msg")

        async def emit(delta: str) -> None:
            # Tokens carry no state, so they go straight to this user's socket as the model
            # writes them — no waiting for the completion to finish first.
            await self._publish_outbound(
                ChatToken(user_id=user_id, message_id=assistant_id, delta=delta)
            )

        try:
            result = await self._chat_agent.run(
                self.session, user_msg.text, user_id=user_id, on_delta=emit
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "[live] chat turn failed bot_id=%s user_id=%s", self.bot_id, user_id
            )
            return

        await self._mailbox.put(
            _ChatTurnCompleted(
                user_id=user_id,
                assistant_id=assistant_id,
                reply=result.turn.reply,
                tool_events=tuple(result.tool_events),
                additional_data=result.additional_data,
            )
        )

    async def _apply_chat_turn(self, completed: _ChatTurnCompleted) -> None:
        for tool_event in completed.tool_events:
            await self._append(tool_event)

        await self._append(
            ChatMessageAppended(
                user_id=completed.user_id,
                role="assistant",
                message_id=completed.assistant_id,
                text=completed.reply,
            )
        )
        await self._publish_outbound(
            ChatMessageOut(
                user_id=completed.user_id,
                id=completed.assistant_id,
                text=completed.reply,
                additional_data=completed.additional_data,
            )
        )
        logger.info(
            "[live] chat turn completed bot_id=%s user_id=%s message_id=%s tool_events=%s "
            "reply_len=%s",
            self.bot_id,
            completed.user_id,
            completed.assistant_id,
            [e.kind for e in completed.tool_events],
            len(completed.reply),
        )

    async def _publish_outbound(self, message) -> None:
        if self._event_bus is not None:
            log_outbound(self.bot_id, message)
            await self._event_bus.publish(self.bot_id, message)

    async def _on_transcript_segment(self) -> None:
        reason = self._trigger.on_transcript_turn()
        if reason is not None:
            await self._fire_evaluation(reason)

    async def _fire_evaluation(self, reason: TriggerReason) -> None:
        window = recent_transcript(self.session.transcript, self._memory_window)
        total = len(self.session.transcript)
        logger.info(
            "[live] agent evaluation triggered bot_id=%s reason=%s window_turns=%s total_turns=%s",
            self.bot_id,
            reason.value,
            len(window),
            total,
        )
        await self._append(
            AgentEvaluationTriggered(
                reason=reason.value,
                window_turn_count=len(window),
                total_turn_count=total,
            )
        )
        self._request_evaluation()

    def _request_evaluation(self) -> None:
        """Start an evaluation, or mark one as owed if inference is already running.

        Coalescing rather than queueing is the point: while the model is thinking, further
        triggers collapse into a single follow-up run. Without this, triggers arrive faster than
        evaluations complete and the backlog grows for the whole meeting.
        """
        if self._orchestrator is None or self._session_ended:
            logger.info(
                "[live] evaluation skipped bot_id=%s orchestrator=%s session_ended=%s",
                self.bot_id,
                self._orchestrator is not None,
                self._session_ended,
            )
            return
        if self._eval_inflight:
            self._eval_pending = True
            logger.debug("[live] evaluation coalesced bot_id=%s", self.bot_id)
            return

        # Phase 1 on the loop: snapshot session state before handing it to the off-loop task.
        ctx = self._orchestrator.build_context(self.session)
        self._eval_inflight = True
        task = asyncio.create_task(
            self._infer_evaluation(ctx),
            name=f"agent-eval-{self.bot_id}",
        )
        self._background.track(task)

    async def _infer_evaluation(self, ctx) -> None:
        """Phase 2 off the loop: the model calls. Results go back through the mailbox."""
        try:
            outcome = await self._orchestrator.infer(ctx)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[live] agent evaluation failed bot_id=%s", self.bot_id)
            outcome = None
        await self._mailbox.put(_EvaluationInferred(outcome=outcome))

    async def _apply_evaluation(self, outcome) -> None:
        """Phase 3 on the loop: run tools, append events, then honour any coalesced trigger."""
        self._eval_inflight = False
        try:
            if outcome is not None and not self._session_ended:
                result = self._orchestrator.apply(self.session, outcome)
                for event in result.events:
                    await self._append(event)
                if result.stale_questions:
                    # Prep changed under this evaluation; its question ids meant something else.
                    # Re-run against the current set rather than dropping the moment.
                    self._eval_pending = True
                self._spawn_company_resolution(outcome)
        finally:
            if self._eval_pending and not self._session_ended:
                self._eval_pending = False
                self._request_evaluation()

    def _spawn_company_resolution(self, outcome) -> None:
        """Network lookups run off the loop so they never sit on the evaluation's critical path."""
        names = outcome.company_names
        if not names:
            return
        snapshot = snapshot_for_company_resolution(self.session, names)
        task = asyncio.create_task(
            self._resolve_companies(snapshot),
            name=f"company-resolve-{self.bot_id}",
        )
        self._background.track(task)

    async def _resolve_companies(self, snapshot) -> None:
        try:
            events = await self._orchestrator.resolve_companies(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[live] company resolution failed bot_id=%s", self.bot_id)
            return
        if events:
            await self._mailbox.put(_EventsReady(events=events))

    async def _publish_wire_event(self, event: MeetingEvent) -> None:
        if self._event_bus is None:
            return
        wire = domain_event_to_wire(event)
        if wire is not None:
            log_outbound(self.bot_id, wire)
            await self._event_bus.publish(self.bot_id, wire)

    def _start_eval_ticker(self) -> None:
        """One periodic tick, started at session start.

        The old debounce cancelled and recreated a timer on *every* transcript segment, so during
        continuous speech it never fired and the time-based path was effectively dead — a long
        single-speaker stretch was never evaluated. A steady tick paired with the trigger's
        turns-since-last-evaluation counter gives "evaluate if anything happened since we last
        thought", which is what the time threshold was always meant to mean.
        """
        if self._eval_ticker is None:
            self._eval_ticker = asyncio.create_task(
                self._eval_tick_loop(),
                name=f"eval-ticker-{self.bot_id}",
            )

    def _cancel_eval_ticker(self) -> None:
        if self._eval_ticker is not None:
            self._eval_ticker.cancel()
            self._eval_ticker = None

    async def _eval_tick_loop(self) -> None:
        interval = self._trigger.config.min_seconds
        try:
            while True:
                await asyncio.sleep(interval)
                await self._mailbox.put(_EvaluationTimerFired())
        except asyncio.CancelledError:
            return

    def _to_event(self, message) -> MeetingEvent | None:
        if isinstance(message, TranscriptSegmentMessage):
            seg = message.segment
            return TranscriptSegmentAppended(
                seq=seg.seq,
                speaker=seg.speaker,
                speaker_is_host=seg.speaker_is_host,
                participant_uuid=seg.participant_uuid,
                text=seg.text,
                start_ms=seg.start_ms,
                duration_ms=seg.duration_ms,
                timestamp=seg.start_ms / 1000.0,
                turn_id=str(seg.seq),
            )
        return None

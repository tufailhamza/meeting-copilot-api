"""Agent orchestration: gate → reasoner → tools.

One evaluation runs per trigger for the whole session, not per user: the gate and reasoner see the
shared transcript and the *canonical* (deduped) question list, then the tools fan the resulting
actions out to every user who owns a copy. Inference cost is therefore flat in the number of
coached users.

**The cycle is split into three phases so the slow one can run off the actor's mailbox loop:**

1. ``build_evaluation_context`` — cheap, pure read of session state. Runs **on** the loop.
2. :meth:`AgentOrchestrator.infer` — the two model calls. Runs **off** the loop; touches no
   session state, only the context snapshot it was handed.
3. :meth:`AgentOrchestrator.apply` — validation + tools. Synchronous, runs **on** the loop, so
   session reads and event writes stay single-writer.

Company resolution is a fourth, independent phase: it makes network calls, so it runs off the loop
as its own task and its events arrive whenever they arrive.

Restraint comes from the gate's own confidence threshold and from each tool's validation — there
is no action rate limit.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from src.live.config_live import company_filter_settings, live_settings
from src.live.schemas_live.common import new_id
from src.live.services_live.agent.context import EvaluationContext, build_evaluation_context
from src.live.services_live.agent.model_client import ModelClient, build_model_client
from src.live.services_live.agent.schemas import GateDecision, GateVerdict, ReasonerAction, ReasonerOutcome
from src.live.services_live.events import CompanyDetected, MeetingEvent
from src.live.services_live.projection import MeetingSession
from src.live.services_live.company_filter import map_company_name
from src.live.services_live.tools.base import ToolRegistry, build_default_tool_registry
from src.live.services_live.trace_log import log_gate_decision, log_orchestrator_skip

logger = logging.getLogger(__name__)

_ALLOWED_ACTIONS: dict[GateVerdict, frozenset[ReasonerAction]] = {
    GateVerdict.QUESTION_PROGRESS: frozenset({ReasonerAction.ANSWER_PREP_QUESTION}),
    GateVerdict.FACT_CHECK_CLAIM: frozenset({ReasonerAction.FACT_CHECK}),
    GateVerdict.AGENDA_DRIFT: frozenset({ReasonerAction.FLAG_AGENDA_RISK}),
    GateVerdict.SUGGEST_OPPORTUNITY: frozenset({ReasonerAction.SUGGEST_FOLLOWUP}),
    GateVerdict.NO_ACTION: frozenset(),
}


@dataclass(frozen=True)
class OrchestratorResult:
    events: tuple[MeetingEvent, ...] = ()
    skipped: bool = False
    skip_reason: str | None = None
    gate_verdict: GateVerdict | None = None
    reasoner_action: ReasonerAction | None = None
    # The question set moved while this evaluation was in flight, so question-dependent output
    # was dropped. The caller should re-evaluate against the current set.
    stale_questions: bool = False

    @property
    def applied(self) -> bool:
        return bool(self.events) and not self.skipped


@dataclass(frozen=True)
class InferenceOutcome:
    """Result of the two model calls, with no session state attached."""

    gate: GateDecision | None = None
    reasoner: ReasonerOutcome | None = None
    skipped: bool = False
    skip_reason: str | None = None
    # Question-set version the context was built against; see OrchestratorResult.stale_questions.
    questions_version: int | None = None

    @property
    def company_names(self) -> tuple[str, ...]:
        """Every company named this cycle, in first-seen order.

        Sourced primarily from the **gate**, which runs on every evaluation. The reasoner's list
        is merged in when it happened to run, but detection no longer depends on it.
        """
        names: list[str] = []
        for source in (self.gate, self.reasoner):
            for raw in (getattr(source, "company_names", None) or ()):
                name = (raw or "").strip()
                if name and name not in names:
                    names.append(name)
        return tuple(names)


@dataclass(frozen=True)
class CompanyResolutionInput:
    """Immutable snapshot for off-loop company resolution.

    Copied on the mailbox loop so the network task never reads live session state while the loop
    is mutating it.
    """

    company_names: tuple[str, ...] = ()
    seen_names: frozenset[str] = frozenset()
    seen_urns: frozenset[str] = frozenset()
    transcript: tuple[tuple[str, str], ...] = ()  # (speaker, text), oldest first


def snapshot_for_company_resolution(
    session: MeetingSession,
    company_names: tuple[str, ...],
) -> CompanyResolutionInput:
    return CompanyResolutionInput(
        company_names=company_names,
        seen_names=frozenset(c.company_name.lower() for c in session.companies if c.company_name),
        seen_urns=frozenset(c.li_urn for c in session.companies if c.li_urn),
        transcript=tuple((t.speaker or "", t.text or "") for t in session.transcript),
    )


def action_allowed_for_gate(gate: GateVerdict, action: ReasonerAction) -> bool:
    if action == ReasonerAction.NO_ACTION:
        return True
    return action in _ALLOWED_ACTIONS.get(gate, frozenset())


class AgentOrchestrator:
    """Runs one evaluation cycle: context → gate → reasoner → tool → domain events."""

    def __init__(
        self,
        *,
        model_client: ModelClient | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_window_turns: int | None = None,
        min_gate_confidence: float | None = None,
    ) -> None:
        self._model = model_client
        self._tools = tool_registry or build_default_tool_registry()
        self._memory_window = memory_window_turns or live_settings.memory_window_turns
        self._min_gate_confidence = (
            min_gate_confidence
            if min_gate_confidence is not None
            else live_settings.orchestrator_min_gate_confidence
        )

    @property
    def memory_window_turns(self) -> int:
        return self._memory_window

    def build_context(self, session: MeetingSession) -> EvaluationContext:
        """Phase 1 — cheap snapshot of session state. Call this on the mailbox loop."""
        return build_evaluation_context(session, memory_window_turns=self._memory_window)

    async def infer(self, ctx: EvaluationContext) -> InferenceOutcome:
        """Phase 2 — the model calls. Safe to run off the mailbox loop: no session access."""
        if self._model is None:
            logger.debug("[live] orchestrator skip bot_id=%s reason=no_model_client", ctx.bot_id)
            return InferenceOutcome(skipped=True, skip_reason="no_model_client")

        try:
            gate, reasoner = await self._model.evaluate(ctx)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("[live] orchestrator inference failed bot_id=%s", ctx.bot_id)
            return InferenceOutcome(skipped=True, skip_reason="inference_failed")

        log_gate_decision(
            ctx.bot_id,
            verdict=gate.verdict.value,
            confidence=gate.confidence,
            rationale=gate.rationale,
            reasoner_ran=reasoner is not None,
        )
        return InferenceOutcome(
            gate=gate,
            reasoner=reasoner,
            questions_version=ctx.questions_version,
        )

    def apply(self, session: MeetingSession, outcome: InferenceOutcome) -> OrchestratorResult:
        """Phase 3 — validate the inference and run tools. Synchronous; call on the loop."""
        if outcome.skipped or outcome.gate is None:
            return OrchestratorResult(
                skipped=True,
                skip_reason=outcome.skip_reason or "no_inference",
            )

        gate = outcome.gate
        reasoner = outcome.reasoner

        if gate.verdict == GateVerdict.NO_ACTION:
            log_orchestrator_skip(
                session.bot_id,
                "gate_no_action",
                gate_verdict=gate.verdict.value,
                confidence=gate.confidence,
                detail=gate.rationale,
            )
            return OrchestratorResult(
                skipped=True, skip_reason="gate_no_action", gate_verdict=gate.verdict
            )

        # Recording an answered question is exempt from the confidence floor, for the same reason
        # it is exempt from the gate's silence bias: it interrupts nobody. A floor exists so the
        # coach does not interrupt on a hunch — but a tick on a checklist the host is already
        # watching costs them nothing, and dropping the whole cycle over a 0.5 meant a question
        # the participant plainly answered stayed unmarked. Whether it was answered is the
        # reasoner's judgment; this only stops a threshold overruling it.
        if gate.confidence < self._min_gate_confidence:
            if gate.verdict == GateVerdict.QUESTION_PROGRESS:
                logger.info(
                    "[live] low gate confidence kept for marking bot_id=%s confidence=%.2f",
                    session.bot_id,
                    gate.confidence,
                )
            else:
                log_orchestrator_skip(
                    session.bot_id,
                    "low_gate_confidence",
                    gate_verdict=gate.verdict.value,
                    confidence=gate.confidence,
                    detail=gate.rationale,
                )
                return OrchestratorResult(
                    skipped=True,
                    skip_reason="low_gate_confidence",
                    gate_verdict=gate.verdict,
                )

        if reasoner is None or reasoner.action == ReasonerAction.NO_ACTION:
            log_orchestrator_skip(
                session.bot_id,
                "reasoner_no_action",
                gate_verdict=gate.verdict.value,
                confidence=gate.confidence,
                detail=reasoner.rationale if reasoner else gate.rationale,
            )
            return OrchestratorResult(
                skipped=True,
                skip_reason="reasoner_no_action",
                gate_verdict=gate.verdict,
            )

        # Canonical ids are positional. If prep was replaced, a question was added, or grouping
        # resolved while this ran, the ids the model named now point somewhere else — acting on
        # them could mark the wrong question.
        stale = (
            outcome.questions_version is not None
            and outcome.questions_version != session.questions_version
        )
        if stale and reasoner.action == ReasonerAction.ANSWER_PREP_QUESTION:
            log_orchestrator_skip(
                session.bot_id,
                "stale_question_ids",
                gate_verdict=gate.verdict.value,
                action=reasoner.action.value,
                detail=(
                    f"context v{outcome.questions_version} != session v{session.questions_version}"
                ),
            )
            return OrchestratorResult(
                skipped=True,
                skip_reason="stale_question_ids",
                gate_verdict=gate.verdict,
                reasoner_action=reasoner.action,
                stale_questions=True,
            )

        if not action_allowed_for_gate(gate.verdict, reasoner.action):
            log_orchestrator_skip(
                session.bot_id,
                "gate_action_mismatch",
                gate_verdict=gate.verdict.value,
                action=reasoner.action.value,
            )
            return OrchestratorResult(
                skipped=True,
                skip_reason="gate_action_mismatch",
                gate_verdict=gate.verdict,
                reasoner_action=reasoner.action,
            )

        if stale and reasoner.linked_question_id:
            # Everything else in this action is still valid; only the link is unresolvable.
            reasoner = reasoner.model_copy(update={"linked_question_id": None})

        tool_result = self._tools.execute(session, reasoner)
        if not tool_result.applied:
            log_orchestrator_skip(
                session.bot_id,
                "tool_skipped",
                gate_verdict=gate.verdict.value,
                action=reasoner.action.value,
                detail=tool_result.skip_reason,
            )
            return OrchestratorResult(
                skipped=True,
                skip_reason=tool_result.skip_reason or "tool_skipped",
                gate_verdict=gate.verdict,
                reasoner_action=reasoner.action,
            )

        events = list(tool_result.events)
        events.extend(self._companion_answer_events(session, gate, reasoner))
        events.extend(self._companion_suggestion_events(session, gate, reasoner))

        logger.info(
            "[live] orchestrator applied bot_id=%s gate=%s action=%s users=%s events=%s",
            session.bot_id,
            gate.verdict.value,
            reasoner.action.value,
            len(session.users),
            [e.kind for e in events],
        )
        return OrchestratorResult(
            events=tuple(events),
            gate_verdict=gate.verdict,
            reasoner_action=reasoner.action,
            stale_questions=stale,
        )

    def _companion_answer_events(
        self,
        session: MeetingSession,
        gate: GateDecision,
        reasoner: ReasonerOutcome,
    ) -> tuple[MeetingEvent, ...]:
        """Prep questions the transcript answered, recorded whatever else this cycle decided.

        Recording an answer is bookkeeping, not an interruption: it ticks a box the host is already
        watching and costs them no attention. It should never have had to win a cycle against
        advice — but it did, because one gate verdict permits one action. So a window where the
        gate saw a suggestion opportunity, or a claim worth checking, could not also record the
        answer a participant had just given. The host watched their checklist sit at 0/3 through a
        call in which every question was answered out loud.

        Marking is therefore unconditional on the primary action. The reasoner still decides
        *whether* a question was answered; this only stops that decision being discarded because
        the cycle was spent on something else.
        """
        if reasoner.action == ReasonerAction.ANSWER_PREP_QUESTION:
            return ()
        if not reasoner.question_marks:
            return ()
        tool = self._tools.get(ReasonerAction.ANSWER_PREP_QUESTION)
        if tool is None:
            return ()
        companion = reasoner.model_copy(update={"action": ReasonerAction.ANSWER_PREP_QUESTION})
        result = tool.execute(session, companion)
        if not result.applied:
            log_orchestrator_skip(
                session.bot_id,
                "companion_answer_skipped",
                gate_verdict=gate.verdict.value,
                action=reasoner.action.value,
                detail=result.skip_reason,
            )
            return ()
        return result.events

    def _companion_suggestion_events(
        self,
        session: MeetingSession,
        gate: GateDecision,
        reasoner: ReasonerOutcome,
    ) -> tuple[MeetingEvent, ...]:
        """A follow-up raised *alongside* the primary action, rather than instead of it.

        One evaluation used to produce one action, so a cycle that marked a question could not also
        suggest anything. In a prep-heavy meeting the gate prefers ``question_progress`` on almost
        every window, so marking took every cycle — a full 12-minute call produced 28 marks and
        **zero** suggestions, not because the suggestion lane was broken but because it never got a
        turn.

        The suggestion fields are disjoint from every other action's payload (marks use
        ``question_marks``, fact-check uses ``claim``/``fact_verdict``, risk uses ``risk_text``), so
        a suggestion rides along without changing how the primary action is validated or executed.

        Deliberately conservative: only on a cycle whose primary action already applied, and never
        when the primary action *was* the suggestion. Restraint about whether to suggest at all
        stays where it belongs — the model's judgment, informed by ``prior_suggestions``.
        """
        if reasoner.action == ReasonerAction.SUGGEST_FOLLOWUP:
            return ()
        if not (reasoner.suggestion_text or "").strip():
            return ()

        tool = self._tools.get(ReasonerAction.SUGGEST_FOLLOWUP)
        if tool is None:
            return ()

        # The tool guards on ``outcome.action``; the rest of the payload is already correct, and
        # any stale ``linked_question_id`` was nulled before the primary tool ran.
        companion = reasoner.model_copy(update={"action": ReasonerAction.SUGGEST_FOLLOWUP})
        result = tool.execute(session, companion)
        if not result.applied:
            log_orchestrator_skip(
                session.bot_id,
                "companion_suggestion_skipped",
                gate_verdict=gate.verdict.value,
                action=reasoner.action.value,
                detail=result.skip_reason,
            )
            return ()
        return result.events

    async def resolve_companies(
        self,
        snapshot: CompanyResolutionInput,
    ) -> tuple[MeetingEvent, ...]:
        """Independent phase — network lookups, safe to run off the loop.

        Names are resolved **concurrently**; previously this was a sequential await per name, so N
        companies added N round trips to the critical path.
        """
        if not company_filter_settings.configured or not snapshot.company_names:
            return ()

        wanted: list[str] = []
        for raw_name in snapshot.company_names:
            name = (raw_name or "").strip()
            if name and name.lower() not in snapshot.seen_names and name not in wanted:
                wanted.append(name)
        if not wanted:
            return ()

        results = await asyncio.gather(
            *(map_company_name(name) for name in wanted),
            return_exceptions=True,
        )

        events: list[MeetingEvent] = []
        seen_names = set(snapshot.seen_names)
        seen_urns = set(snapshot.seen_urns)
        # Pair each result with the name the speaker actually used: the resolved name can differ
        # (an alias, or the service's own spelling), and the transcript only ever contains what
        # was said.
        for spoken, result in zip(wanted, results):
            if isinstance(result, BaseException) or result is None:
                if isinstance(result, BaseException):
                    logger.warning("[live] company lookup failed: %s", result)
                continue
            if result.company_name.lower() in seen_names:
                continue
            if result.li_urn and result.li_urn in seen_urns:
                continue
            seen_names.add(result.company_name.lower())
            if result.li_urn:
                seen_urns.add(result.li_urn)
            events.append(
                CompanyDetected(
                    company_id=new_id("co"),
                    company_name=result.company_name,
                    title=f"{result.company_name} mentioned in the conversation",
                    text=_company_context_text(
                        snapshot.transcript, result.company_name, spoken_as=spoken
                    ),
                    es_id=result.es_id,
                    li_urn=result.li_urn,
                    li_name=result.li_name,
                    li_universalname=result.li_universalname,
                    li_industries=result.li_industries,
                )
            )
        return tuple(events)

    async def run(self, session: MeetingSession) -> OrchestratorResult:
        """Convenience path that runs all phases inline.

        The actor uses the split phases so inference stays off its mailbox loop; this exists for
        callers that do not need that separation.
        """
        ctx = self.build_context(session)
        outcome = await self.infer(ctx)
        result = self.apply(session, outcome)
        companies = await self.resolve_companies(
            snapshot_for_company_resolution(session, outcome.company_names)
        )
        if not companies:
            return result
        return OrchestratorResult(
            events=tuple(result.events) + companies,
            skipped=False,
            skip_reason=result.skip_reason if not result.events else None,
            gate_verdict=result.gate_verdict,
            reasoner_action=result.reasoner_action,
        )


def _company_context_text(
    transcript: tuple[tuple[str, str], ...],
    company_name: str,
    *,
    spoken_as: str | None = None,
    max_len: int = 220,
) -> str:
    """Quote the most recent transcript turn that named the company.

    Searches for what the speaker *said* as well as the resolved name. Those differ whenever an
    alias is involved, and searching only for the resolved name meant those cards always fell
    back to the generic sentence instead of quoting the line.
    """
    needles = [n.lower() for n in (spoken_as, company_name) if n]
    for speaker, text in reversed(transcript):
        haystack = (text or "").lower()
        if any(needle in haystack for needle in needles):
            snippet = (text or "").strip()
            if len(snippet) > max_len:
                snippet = snippet[: max_len - 3].rstrip() + "..."
            return f'{speaker or "Someone"} mentioned {company_name}: "{snippet}"'
    return f"{company_name} was referenced in the recent discussion."


def build_default_orchestrator(
    *,
    memory_window_turns: int | None = None,
) -> AgentOrchestrator:
    return AgentOrchestrator(
        model_client=build_model_client(),
        memory_window_turns=memory_window_turns,
    )

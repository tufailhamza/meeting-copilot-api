"""Structured inputs/outputs for tiered agent inference (gate → reasoner)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class GateVerdict(str, Enum):
    """Cheap classifier: should the supervisor act, and what kind of opportunity exists?"""

    NO_ACTION = "no_action"
    QUESTION_PROGRESS = "question_progress"
    SUGGEST_OPPORTUNITY = "suggest_opportunity"
    FACT_CHECK_CLAIM = "fact_check_claim"
    AGENDA_DRIFT = "agenda_drift"


class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: GateVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    focus_question_id: str | None = None
    claim_text: str | None = None
    # Extracted here rather than downstream because the gate runs on **every** evaluation, while
    # the reasoner only runs when the gate decides to act. Hanging company detection off the
    # reasoner meant it almost never happened.
    company_names: list[str] = Field(default_factory=list)


class ReasonerAction(str, Enum):
    NO_ACTION = "no_action"
    ANSWER_PREP_QUESTION = "answer_prep_question"
    SUGGEST_FOLLOWUP = "suggest_followup"
    FACT_CHECK = "fact_check"
    FLAG_AGENDA_RISK = "flag_agenda_risk"


class QuestionMark(BaseModel):
    """One prep question the transcript now answers, with the turn carrying that answer.

    ``evidence_turn_seq`` must cite the turn where the question is **answered**, not where it is
    asked. Citing the same turn twice records nothing, which is how a standing answer avoids being
    re-sent every cycle.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    # Where the answer starts, and where it finishes if it took more than one turn. Real answers
    # straddle segment boundaries constantly ("So the main challenge is" / "the accuracy of the
    # model"), and citing only the first segment showed the host an answer cut off mid-sentence.
    evidence_turn_seq: int
    evidence_turn_seq_end: int | None = None
    # The answer in your own words. The turn citation proves it was said; this is what the host
    # reads, so it must be a clean sentence rather than raw transcript with the stutters,
    # crosstalk and false starts left in.
    answer_text: str = Field(min_length=1)


class ReasonerOutcome(BaseModel):
    """Structured action the orchestrator will execute via tools.

    Evidence is a **transcript turn reference**, not retyped text: ``evidence_turn_seq`` points at
    the ``seq`` of the turn that supports the action. Verification is then an existence check
    against the transcript rather than a string-similarity guess, so a correct decision can never
    be discarded for paraphrasing, and fabricated evidence cannot pass at all.

    ``question_marks`` is a **list**: a single stretch of conversation routinely covers more than
    one prep question, and emitting only the first left the rest pending until some later cycle
    happened to notice them again.
    """

    model_config = ConfigDict(extra="forbid")

    action: ReasonerAction
    rationale: str = Field(min_length=1)
    question_marks: list[QuestionMark] = Field(default_factory=list)
    # Which transcript turn shows the claim was asserted (fact-check grounding).
    evidence_turn_seq: int | None = None
    suggestion_text: str | None = None
    suggestion_rationale: str | None = None
    linked_question_id: str | None = None
    claim: str | None = None
    fact_verdict: str | None = None  # true | false | unclear
    fact_detail: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_text: str | None = None
    company_names: list[str] = Field(default_factory=list)

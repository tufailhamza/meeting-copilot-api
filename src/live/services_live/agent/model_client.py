"""ModelClient seam: tiered gate → reasoner inference over OpenAI (async).

``TieredModelClient`` is the production path: a fast gate model decides *if* to act; the
reasoner model runs only when the gate flags something. ``FakeModelClient`` supports deterministic
unit tests without API calls.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import APIError, AsyncOpenAI
from pydantic import ValidationError

from src.config import settings as app_settings
from src.live.config_live import live_settings
from src.live.services_live.agent.context import EvaluationContext
from src.live.services_live.agent.prompts import GATE_SYSTEM_PROMPT, REASONER_SYSTEM_PROMPT
from src.live.services_live.agent.schemas import (
    GateDecision,
    GateVerdict,
    ReasonerAction,
    ReasonerOutcome,
)

logger = logging.getLogger(__name__)


class ModelClient(ABC):
    @abstractmethod
    async def run_gate(self, ctx: EvaluationContext) -> GateDecision: ...

    @abstractmethod
    async def run_reasoner(
        self, ctx: EvaluationContext, gate: GateDecision
    ) -> ReasonerOutcome: ...

    async def evaluate(
        self, ctx: EvaluationContext
    ) -> tuple[GateDecision, ReasonerOutcome | None]:
        gate = await self.run_gate(ctx)
        if gate.verdict == GateVerdict.NO_ACTION:
            return gate, None
        reasoner = await self.run_reasoner(ctx, gate)
        return gate, reasoner


def _normalize_effort(effort: str | None) -> str | None:
    """``None``/``""``/``"none"`` all mean *no internal reasoning*."""
    raw = (effort or "").strip().lower()
    return raw if raw and raw != "none" else None


class TieredModelClient(ModelClient):
    def __init__(
        self,
        *,
        api_key: str,
        gate_model: str,
        reasoner_model: str,
        reasoner_reasoning_effort: str | None = None,
        reasoner_fact_check_effort: str | None = "medium",
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is not set")
        self._gate_model = gate_model.strip()
        self._reasoner_model = reasoner_model.strip()
        self._reasoner_reasoning_effort = _normalize_effort(reasoner_reasoning_effort)
        self._fact_check_effort = _normalize_effort(reasoner_fact_check_effort)
        self._client = client or AsyncOpenAI(api_key=api_key.strip())

    def effort_for(self, gate: GateDecision) -> str | None:
        """Deliberate only where it pays: fact-checking weighs a claim against world knowledge.

        Every other action is judgment-light, and reasoning tokens there are seconds of latency on
        a live call for no quality gain.
        """
        if gate.verdict == GateVerdict.FACT_CHECK_CLAIM:
            return self._fact_check_effort
        return self._reasoner_reasoning_effort

    async def run_gate(self, ctx: EvaluationContext) -> GateDecision:
        return await self._parse_completion(
            model=self._gate_model,
            system=GATE_SYSTEM_PROMPT,
            user=self._gate_user_message(ctx),
            schema=GateDecision,
            temperature=0.1,
        )

    async def run_reasoner(self, ctx: EvaluationContext, gate: GateDecision) -> ReasonerOutcome:
        effort = self.effort_for(gate)
        return await self._parse_completion(
            model=self._reasoner_model,
            system=REASONER_SYSTEM_PROMPT,
            user=self._reasoner_user_message(ctx, gate),
            schema=ReasonerOutcome,
            temperature=0.25 if effort is None else None,
            reasoning_effort=effort,
        )

    async def _parse_completion(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ):
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": schema,
        }
        if reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = reasoning_effort
        elif temperature is not None:
            create_kwargs["temperature"] = temperature

        try:
            response = await self._client.beta.chat.completions.parse(**create_kwargs)
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        message = response.choices[0].message
        parsed = message.parsed
        if parsed is None:
            raise RuntimeError("Model returned empty structured output")
        return parsed

    @staticmethod
    def _gate_user_message(ctx: EvaluationContext) -> str:
        return (
            "Classify whether the supervisor should act.\n\n"
            f"MEETING_CONTEXT_JSON:\n{ctx.to_prompt_block()}"
        )

    @staticmethod
    def _reasoner_user_message(ctx: EvaluationContext, gate: GateDecision) -> str:
        gate_json = gate.model_dump_json()
        # Compact serialization: same information as the gate sees, without the pretty-printing
        # whitespace, which is pure token cost on the second call of every cycle.
        return (
            "Decide the structured actions to take.\n\n"
            f"GATE_DECISION:\n{gate_json}\n\n"
            f"MEETING_CONTEXT_JSON:\n{ctx.to_compact_block()}"
        )


OpenAIModelClient = TieredModelClient


class FakeModelClient(ModelClient):
    """Deterministic client for tests. Configure fixed gate/reasoner responses per call."""

    def __init__(
        self,
        *,
        gate_responses: list[GateDecision] | None = None,
        reasoner_responses: list[ReasonerOutcome] | None = None,
    ) -> None:
        self._gate_responses = list(gate_responses or [])
        self._reasoner_responses = list(reasoner_responses or [])
        self._gate_idx = 0
        self._reasoner_idx = 0
        self.gate_calls = 0
        self.reasoner_calls = 0

    async def run_gate(self, ctx: EvaluationContext) -> GateDecision:
        self.gate_calls += 1
        if self._gate_idx < len(self._gate_responses):
            decision = self._gate_responses[self._gate_idx]
            self._gate_idx += 1
            return decision
        return GateDecision(
            verdict=GateVerdict.NO_ACTION,
            confidence=0.99,
            rationale="default fake gate: no action",
        )

    async def run_reasoner(self, ctx: EvaluationContext, gate: GateDecision) -> ReasonerOutcome:
        self.reasoner_calls += 1
        if self._reasoner_idx < len(self._reasoner_responses):
            outcome = self._reasoner_responses[self._reasoner_idx]
            self._reasoner_idx += 1
            return outcome
        return ReasonerOutcome(
            action=ReasonerAction.NO_ACTION,
            rationale="default fake reasoner: no action",
        )


def build_model_client() -> ModelClient | None:
    """Build the production tiered client when ``OPENAI_API_KEY`` is configured."""
    if not app_settings.openai_api_key.strip():
        return None
    return TieredModelClient(
        api_key=app_settings.openai_api_key,
        gate_model=live_settings.openai_gate_model,
        reasoner_model=live_settings.openai_reasoner_model,
        reasoner_reasoning_effort=live_settings.openai_reasoner_reasoning_effort,
        reasoner_fact_check_effort=live_settings.openai_reasoner_fact_check_effort,
    )


def validate_gate_decision(raw: dict) -> GateDecision:
    """Helper for tests parsing fixture JSON."""
    try:
        return GateDecision.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

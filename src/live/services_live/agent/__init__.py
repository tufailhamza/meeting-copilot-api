"""Agent runtime: tiered inference (gate → reasoner), prompts, and model client."""

from src.live.services_live.agent.context import EvaluationContext, build_evaluation_context
from src.live.services_live.agent.model_client import (
    FakeModelClient,
    ModelClient,
    OpenAIModelClient,
    TieredModelClient,
    build_model_client,
)
from src.live.services_live.agent.schemas import (
    GateDecision,
    GateVerdict,
    ReasonerOutcome,
    ReasonerAction,
)

__all__ = [
    "EvaluationContext",
    "build_evaluation_context",
    "FakeModelClient",
    "GateDecision",
    "GateVerdict",
    "ModelClient",
    "OpenAIModelClient",
    "ReasonerAction",
    "ReasonerOutcome",
    "TieredModelClient",
    "build_model_client",
]

"""Evaluation trigger policy: decides *when* the agent should think (not *what* it should do).

Transcript turns accumulate between evaluations. A trigger fires when either:
- **turn threshold** — at least ``min_turns`` new turns since the last evaluation, or
- **time threshold** — at least ``min_seconds`` have passed with pending turns (debounce timer).

This keeps cost and latency proportional to meeting activity instead of firing on every
partial transcript line. The orchestrator (P1.11) hooks into the resulting
``agent_evaluation_triggered`` domain event; P1.6 only records the signal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class TriggerReason(str, Enum):
    TURN_THRESHOLD = "turn_threshold"
    TIME_THRESHOLD = "time_threshold"


@dataclass(frozen=True)
class TriggerConfig:
    min_turns: int = 3
    min_seconds: float = 8.0


class EvaluationTriggerPolicy:
    def __init__(
        self,
        config: TriggerConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._mono = monotonic
        self._turns_since_eval = 0
        self._last_eval_at: float | None = None

    @property
    def config(self) -> TriggerConfig:
        return self._config

    @property
    def turns_since_eval(self) -> int:
        return self._turns_since_eval

    def on_transcript_turn(self) -> TriggerReason | None:
        """Record a new turn; return a reason if the turn threshold is met."""
        self._turns_since_eval += 1
        if self._turns_since_eval >= self._config.min_turns:
            self._mark_fired()
            return TriggerReason.TURN_THRESHOLD
        return None

    def on_timer_expired(self) -> TriggerReason | None:
        """Called when the debounce timer fires; return a reason if pending turns exist."""
        if self._turns_since_eval > 0:
            self._mark_fired()
            return TriggerReason.TIME_THRESHOLD
        return None

    def _mark_fired(self) -> None:
        self._turns_since_eval = 0
        self._last_eval_at = self._mono()

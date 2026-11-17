"""Text normalization.

Deliberately dependency-free: the projection and the roster need this, and the agent tools import
the projection, so anything it lived alongside would form an import cycle.

This module holds **identity** normalization only — casefolding and punctuation stripping so two
spellings of the same literal string compare equal. It deliberately contains no similarity or
near-duplicate scoring: judging whether two *different* strings mean the same thing is the model's
job, and heuristics that guessed at it here used to veto correct decisions.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))

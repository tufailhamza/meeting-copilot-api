"""Text normalization for agent tools.

The implementation lives in ``services_live.text_norm`` so the projection and roster can use it
without importing the tools package (which imports the projection).
"""

from __future__ import annotations

from src.live.services_live.text_norm import normalize_text

__all__ = ["normalize_text"]

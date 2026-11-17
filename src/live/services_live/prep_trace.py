"""Step-by-step trace of one call-prep load, written to a file.

Testing this endpoint means reconstructing a chain that spans four systems — the request, two
Postgres databases, Elasticsearch and two LLM calls — and by the time something looks wrong the
intermediate values are gone. This records each step as it happens and dumps the lot to
``Temp/<timestamp>-<event_id>.json``.

Set ``DEBUG_LOAD_PREP = False`` to disable; every call then becomes a no-op.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEBUG_LOAD_PREP = False
TRACE_DIR = Path("Temp")

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]")


class PrepTrace:
    """Accumulates named steps and writes them out once.

    Never raises: a broken trace must not fail the request it is tracing.
    """

    def __init__(self, event_id: str, *, enabled: bool = DEBUG_LOAD_PREP) -> None:
        self.enabled = enabled
        self.event_id = event_id
        self.started_at = datetime.now()
        self._steps: list[dict[str, Any]] = []

    def step(self, name: str, **data: Any) -> None:
        if not self.enabled:
            return
        elapsed = (datetime.now() - self.started_at).total_seconds()
        self._steps.append(
            {
                "step": name,
                "at": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_seconds": round(elapsed, 3),
                **data,
            }
        )

    def write(self) -> str | None:
        """Dump the trace. Returns the path written, or ``None`` when disabled or on failure."""
        if not self.enabled:
            return None
        try:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
            safe_event = _UNSAFE_FILENAME.sub("_", self.event_id)[:80] or "unknown"
            path = TRACE_DIR / f"{stamp}-{safe_event}.json"
            document = {
                "external_event_id": self.event_id,
                "started_at": self.started_at.isoformat(timespec="milliseconds"),
                "finished_at": datetime.now().isoformat(timespec="milliseconds"),
                "total_seconds": round((datetime.now() - self.started_at).total_seconds(), 3),
                "steps": self._steps,
            }
            path.write_text(
                json.dumps(document, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.exception("[live] prep trace could not be written")
            return None

        print(f"[live] prep trace written: {path}")
        logger.info("[live] prep trace written: %s", path)
        return str(path)

"""Structured chat turn output (reply + optional session mutations).

**Field order is load-bearing.** Structured output is emitted in schema order, so every routing
decision is declared *before* ``reply``. That lets the streamer know a reply is about to be
superseded — by a company lookup or a web search — before a single character of it reaches the
host, instead of streaming an answer and then visibly replacing it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatTurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- routing (declared first; see module docstring) ---
    lookup_company: bool = False
    company_name: str | None = None
    generate_similar_companies: bool = False
    similar_companies_query: str | None = None
    use_web_search: bool = False
    web_search_query: str | None = None
    refine_suggestion_id: str | None = None
    refined_suggestion_text: str | None = None
    add_question_text: str | None = None

    # --- the answer ---
    reply: str = Field(min_length=1)

    @property
    def supersedes_reply(self) -> bool:
        """True when this turn's ``reply`` will be replaced by a follow-up step."""
        return bool(self.lookup_company and self.company_name) or bool(
            self.use_web_search and self.web_search_query
        )

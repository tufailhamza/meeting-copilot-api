"""Isolated settings for the live meeting-agent feature.

Uses the ``LIVE_`` env prefix so this feature's configuration never collides with the core
application settings in ``src/config.py``. New knobs (model names, debounce timings, Redis
URL, etc.) are added here as the feature grows.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LIVE_",
        extra="ignore",
    )

    enabled: bool = True
    # Table holding call-prep rows (id uuid, payload jsonb). Override via LIVE_CALL_PREP_TABLE.
    call_prep_table: str = "meeting_call_prep"
    meetings_table: str = "meetings"
    # How long the WS handshake waits for call-prep to load before acking without questions.
    prep_load_timeout_seconds: float = 10.0
    # Rolling transcript window handed to the agent on each evaluation.
    memory_window_turns: int = 40
    # Evaluation trigger: fire after this many new transcript turns and/or this many seconds.
    # Coalescing bounds how often an evaluation can actually run, so this is a floor on
    # responsiveness rather than a throttle. Urgent output (answering a question asked out loud)
    # does not go through this loop at all — see the answering lane.
    trigger_min_turns: int = 2
    trigger_min_seconds: float = 8.0
    # Orchestrator restraint comes from the gate confidence threshold alone. The gate prompt is
    # already heavily biased toward staying silent, so a high floor on top of it filtered twice
    # and read to users as the coach being asleep.
    orchestrator_min_gate_confidence: float = 0.55
    # A confidently wrong fact-check mid-call is worse than silence, so this one keeps a floor.
    # Policy knobs, deliberately visible here rather than buried as constants in a tool body.
    fact_check_min_assert_confidence: float = 0.65
    fact_check_min_unclear_confidence: float = 0.35
    # Display constraint: the risk panel shows a single short line.
    risk_text_max_chars: int = 120
    # Tiered inference models.
    openai_gate_model: str = "gpt-4.1"
    openai_reasoner_model: str = "gpt-5.1"
    # Reasoning effort is per-verdict, because only one action needs deliberation.
    #
    # Most actions (marking a question, flagging drift, writing a follow-up) are judgment-light:
    # deliberation buys little and costs seconds on a live call. Fact-checking is the exception —
    # it weighs a claim against world knowledge, and a confidently wrong fact-check shown to a host
    # mid-call is worse than silence. So the default is no reasoning, and fact-checks opt back in.
    #
    # "none" (or empty) disables internal reasoning; low|medium|high enable it.
    openai_reasoner_reasoning_effort: str = "none"
    openai_reasoner_fact_check_effort: str = "medium"
    openai_chat_model: str = "gpt-4.1"
    openai_summary_model: str = "gpt-4.1-mini"
    # Answers factual questions asked in the meeting. Runs per transcript segment on a tiny
    # prompt, so it must stay small and fast — this is the latency-critical path.
    openai_answerer_model: str = "gpt-4.1-mini"
    # Groups semantically-identical prep questions across users; runs once, in the background.
    openai_question_grouping_model: str = "gpt-4.1"
    # Compaction: fold older transcript turns into a rolling summary once the live window overflows.
    compaction_buffer_turns: int = 10
    summary_max_chars: int = 4000
    # Call-prep regeneration (non-SOA chat). The assistant may notice that a message carries
    # substantial new meeting context the prep does not reflect and offer to bring the whole prep
    # in line, rather than making one small edit. Off means the assistant is never told the
    # capability exists — the prompt section is stripped, the field comes back null, and the turn
    # behaves exactly as it did before the feature.
    call_prep_regeneration_enabled: bool = True
    # Ceilings on what a single proposal may promise. A plan that touches everything reads as
    # though the assistant did not look, and it is also the most expensive turn to carry out.
    regeneration_max_new_sections: int = 3
    regeneration_max_changes: int = 6
    # An approved regeneration rewrites several sections in one turn, so both the section budget
    # in the planner and the token budget on the call are raised for that turn only.
    regeneration_max_turn_tokens: int = 8000


class LiveDBSettings(BaseSettings):
    """Postgres connection settings for the live feature.

    Reads the shared, unprefixed ``DB_*`` variables already present in ``.env`` (no ``LIVE_``
    prefix) so the live feature uses the same database credentials as the rest of the platform.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.db_host and self.db_name and self.db_user)


class CompanyFilterSettings(BaseSettings):
    """Settings for the external company-filter service used for company_detected events.

    Uses unprefixed ``COMPANY_FILTER_BASE_URL`` and ``COMPANY_FILTER_TOKEN`` env vars so the
    same configuration can be shared across services if needed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    company_filter_base_url: str = ""
    company_filter_token: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.company_filter_base_url and self.company_filter_token)


class EnterpriseSearchSettings(BaseSettings):
    """Base URL for the enterprise-search service (SOA call-prep inject)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enterprise_search_base_url: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.enterprise_search_base_url.strip())


class ESSettings(BaseSettings):
    """Elasticsearch connection settings.

    Reads unprefixed ``ES_URL`` / ``ES_ID`` / ``ES_KEY``. ``ES_ID`` and ``ES_KEY`` are the two
    halves of one API key: the header is ``ApiKey base64("<ES_ID>:<ES_KEY>")``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    es_url: str = ""
    es_id: str = ""
    es_key: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.es_url.strip() and self.es_id.strip() and self.es_key.strip())


class CWKDBSettings(BaseSettings):
    """Postgres connection for the coworking DB that holds past-meeting history.

    A different database from the live feature's own, hence the ``CWK_DB_*`` variables. Read-only
    as far as this feature is concerned.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cwk_db_host: str = ""
    cwk_db_port: int = 5432
    cwk_db_name: str = ""
    cwk_db_user: str = ""
    cwk_db_password: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.cwk_db_host and self.cwk_db_name and self.cwk_db_user)


live_settings = LiveSettings()
live_db_settings = LiveDBSettings()
company_filter_settings = CompanyFilterSettings()
enterprise_search_settings = EnterpriseSearchSettings()
es_settings = ESSettings()
cwk_db_settings = CWKDBSettings()

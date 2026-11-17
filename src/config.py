import logging
import sys

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "QLU Transcriber"
    environment: str = "local"
    debug: bool = False
    api_prefix: str = "/api"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000"

    # Outbound callback when async transcription + transliteration + host completes
    transcriber_url: str = ""
    m2m_token: str = ""

    # OpenAI (e.g. notes summary, host detection, transliteration)
    openai_api_key: str = ""
    openai_summary_model: str = "gpt-4.1"
    openai_notes_model: str = "gpt-4.1"
    openai_action_items_model: str = "gpt-4.1"

    @property
    def cors_origin_list(self) -> list[str]:
        raw = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if not raw:
            return ["http://localhost:3000"]
        return raw

    @field_validator("environment")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        return v.lower().strip()


settings = Settings()


def setup_logging() -> None:
    level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

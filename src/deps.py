from collections.abc import Generator

from src.config import Settings, settings


def get_settings() -> Generator[Settings, None, None]:
    yield settings

"""Plain audio→text via OpenAI gpt-4o-mini-transcribe (no diarization / callback)."""

import logging
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ROOT = Path(__file__).resolve().parents[3]
_OPENAI_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"

logger = logging.getLogger(__name__)


def transcribe_mp3_to_text_sync(mp3_path: Path, recording_id: str) -> str:
    load_dotenv(_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "[non_diarize_transcribe][%s] OPENAI_API_KEY is not set",
            recording_id,
        )
        raise ValueError("OPENAI_API_KEY is not set")

    if not mp3_path.is_file():
        logger.error(
            "[non_diarize_transcribe][%s] prepared audio missing path=%s",
            recording_id,
            mp3_path,
        )
        raise FileNotFoundError(f"Prepared audio not found: {mp3_path}")

    client = OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    try:
        with mp3_path.open("rb") as f:
            result = client.audio.transcriptions.create(
                model=_OPENAI_TRANSCRIBE_MODEL,
                file=f,
            )
    except Exception:
        logger.exception(
            "[non_diarize_transcribe][%s] OpenAI %s call failed after %.2f seconds path=%s",
            recording_id,
            _OPENAI_TRANSCRIBE_MODEL,
            time.perf_counter() - t0,
            mp3_path,
        )
        raise

    elapsed = time.perf_counter() - t0
    text = (getattr(result, "text", None) or "").strip()
    logger.info(
        "[non_diarize_transcribe][%s] OpenAI %s took %.2f seconds chars=%s",
        recording_id,
        _OPENAI_TRANSCRIBE_MODEL,
        elapsed,
        len(text),
    )
    if not text:
        logger.warning(
            "[non_diarize_transcribe][%s] OpenAI returned empty transcript",
            recording_id,
        )
    return text


def run_non_diarize_transcribe(
    mp3_path: Path,
    work_dir: Path,
    recording_id: str,
) -> str:
    """Transcribe prepared MP3 to plain text; always deletes work_dir."""
    try:
        return transcribe_mp3_to_text_sync(mp3_path, recording_id)
    except Exception:
        logger.exception(
            "[non_diarize_transcribe][%s] non-diarize pipeline failed path=%s",
            recording_id,
            mp3_path,
        )
        raise
    finally:
        t0 = time.perf_counter()
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.info(
                "[non_diarize_transcribe][%s] cleanup took %.2f seconds",
                recording_id,
                time.perf_counter() - t0,
            )
        except Exception:
            logger.exception(
                "[non_diarize_transcribe][%s] cleanup failed work_dir=%s",
                recording_id,
                work_dir,
            )

"""Production pipeline: AssemblyAI → transliterate → host (same logic); orchestration supports async + callback."""

import asyncio
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.schemas.transcription.transcribe import RecordingTranscribeResponse, TranscriptionSegment
from src.services.transcription.host_detection import detect_and_relabel_host
from src.services.transcription.transcription_callback import post_transcription_result
from src.services.transcription.transliteration import transliterate_transcriptions

_ROOT = Path(__file__).resolve().parents[3]
_ASSEMBLY_LANGUAGE_CODES = ["en", "hi"]

logger = logging.getLogger(__name__)

# Whole word "clue" (any case) + whitespace + whole word "app" or "world" → "QLU" + same spacing + same tail casing.
_CLUE_BEFORE_APP_OR_WORLD = re.compile(r"(?i)\bclue(\s+)(app|world)\b")


def _replace_clue_with_qlu_before_app_or_world(text: str) -> str:
    return _CLUE_BEFORE_APP_OR_WORLD.sub(lambda m: f"QLU{m.group(1)}{m.group(2)}", text)


def _apply_clue_to_qlu_brand(segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
    """Product wording: Clue App / Clue World → QLU App / QLU World (case-insensitive match)."""
    return [
        TranscriptionSegment(
            speaker=s.speaker,
            text=_replace_clue_with_qlu_before_app_or_world(s.text),
            start=s.start,
            end=s.end,
        )
        for s in segments
    ]


def _speaker_label(raw: object) -> str:
    if raw is None:
        return "?"
    s = str(raw).strip()
    if not s:
        return "?"
    if s.isdigit():
        n = int(s)
        if 0 <= n <= 25:
            return chr(ord("A") + n)
        return "?"
    return s


def _segments_from_transcript(transcript: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for u in transcript.utterances or []:
        start_ms = int(getattr(u, "start", 0) or 0)
        end_ms = int(getattr(u, "end", 0) or 0)
        text = (getattr(u, "text", None) or "").strip()
        rows.append(
            {
                "speaker": _speaker_label(getattr(u, "speaker", None)),
                "text": text,
                "start": start_ms / 1000.0,
                "end": end_ms / 1000.0,
            }
        )
    if not rows and (getattr(transcript, "text", None) or "").strip():
        full = transcript.text.strip()
        dur = getattr(transcript, "audio_duration", None)
        end_s = float(dur) if isinstance(dur, (int, float)) and dur else 0.0
        rows.append({"speaker": "A", "text": full, "start": 0.0, "end": end_s})
    return rows


def _run_assembly_segments_from_mp3_sync(mp3_path: Path, recording_id: str) -> dict[str, Any]:
    """AssemblyAI transcribe + diarized segments only (same config as before). Caller owns temp dir cleanup."""
    load_dotenv(_ROOT / ".env")

    api_key = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY (or ASSEMBLY_API_KEY) is not set")

    import assemblyai as aai

    aai.settings.api_key = api_key
    aai.settings.http_timeout = 1800.0
    aai.settings.polling_interval = 3.0

    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        speaker_labels=True,
        language_codes=_ASSEMBLY_LANGUAGE_CODES,
    )
    t_asm0 = time.perf_counter()
    transcript = aai.Transcriber().transcribe(str(mp3_path), config=config)
    t_asm1 = time.perf_counter()
    logger.info(
        "[transcribe][%s] AssemblyAI transcribe + diarize took %.2f seconds",
        recording_id,
        t_asm1 - t_asm0,
    )

    if transcript.status == aai.TranscriptStatus.error:
        msg = transcript.error or "AssemblyAI transcription failed"
        raise RuntimeError(msg)

    segs = _segments_from_transcript(transcript)
    logger.info(
        "[transcribe][%s] AssemblyAI returned %s segment(s)",
        recording_id,
        len(segs),
    )
    return {"recording_id": recording_id, "transcriptions": segs}


async def send_empty_transcription_callback(recording_id: str, reason: str) -> None:
    """Close the loop on a recording that can never produce a transcript.

    The callback contract carries no field for *why*, so an empty ``transcriptions`` list is the
    whole message: this recording is finished and there was nothing in it to transcribe. The
    reason is logged instead of sent, which is what makes an empty callback explicable afterwards.

    Only ever for a permanent property of the recording itself. A transient failure — AssemblyAI
    down, a timeout — must NOT be reported through here: an empty list is indistinguishable from
    "nobody spoke", so an outage answered this way would silently discard real transcripts across
    every recording made while it lasted. Those keep failing loudly and send no callback at all.
    """
    logger.info("[transcribe][%s] no transcript possible: %s", recording_id, reason)
    try:
        await post_transcription_result(
            RecordingTranscribeResponse(recording_id=recording_id, transcriptions=[])
        )
        logger.info("[transcribe][%s] empty-result callback delivered", recording_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Empty-result callback failed recording_id=%s error_type=%s error=%s",
            recording_id,
            type(exc).__name__,
            exc,
        )


async def run_pipeline_after_prepare(mp3_path: Path, work_dir: Path, recording_id: str) -> None:
    """
    Assembly → transliterate → Clue→QLU brand fix → host → POST callback. Deletes work_dir when finished.
    Does not change inner behavior of transliteration or host_detection modules.
    """
    stage = "pipeline_start"
    try:
        logger.info("[transcribe][%s] background pipeline started", recording_id)

        stage = "assemblyai"
        logger.info("[transcribe][%s] stage=%s starting", recording_id, stage)
        data = await asyncio.to_thread(_run_assembly_segments_from_mp3_sync, mp3_path, recording_id)

        stage = "parse_segments"
        segments = [TranscriptionSegment(**x) for x in data["transcriptions"]]
        logger.info(
            "[transcribe][%s] stage=%s complete: %s segment(s) parsed",
            recording_id,
            stage,
            len(segments),
        )

        stage = "transliteration"
        logger.info(
            "[transcribe][%s] stage=%s starting (%s segment(s))",
            recording_id,
            stage,
            len(segments),
        )
        t_tr0 = time.perf_counter()
        try:
            transliterated = await transliterate_transcriptions(segments, recording_id=recording_id)
        except Exception as exc:  # noqa: BLE001
            # Presentation, not content: this rewrites Devanagari into Latin script. The words
            # themselves came from AssemblyAI a stage ago and are already correct, so failing here
            # costs polish. Letting it take the whole recording down means an OpenAI rate limit
            # loses a meeting that was successfully transcribed — the worse trade by far.
            logger.exception(
                "[transcribe][%s] stage=%s failed, continuing with untransliterated text "
                "error_type=%s error=%s",
                recording_id,
                stage,
                type(exc).__name__,
                exc,
            )
            transliterated = segments
        t_tr1 = time.perf_counter()
        logger.info(
            "[transcribe][%s] stage=%s complete in %.2f seconds",
            recording_id,
            stage,
            t_tr1 - t_tr0,
        )

        stage = "brand_fix"
        logger.info("[transcribe][%s] stage=%s starting", recording_id, stage)
        t_br0 = time.perf_counter()
        transliterated = _apply_clue_to_qlu_brand(transliterated)
        t_br1 = time.perf_counter()
        logger.info(
            "[transcribe][%s] stage=%s complete in %.2f seconds",
            recording_id,
            stage,
            t_br1 - t_br0,
        )

        stage = "host_detection"
        logger.info("[transcribe][%s] stage=%s starting", recording_id, stage)
        t_h0 = time.perf_counter()
        try:
            final_segments = await detect_and_relabel_host(
                transliterated, recording_id=recording_id
            )
        except Exception as exc:  # noqa: BLE001
            # Also presentation: it renames whichever speaker is the host from "A" to "Host".
            # Without it the callback still carries every word, attributed to A/B/C — diarized,
            # just not named.
            logger.exception(
                "[transcribe][%s] stage=%s failed, continuing with original speaker labels "
                "error_type=%s error=%s",
                recording_id,
                stage,
                type(exc).__name__,
                exc,
            )
            final_segments = transliterated
        t_h1 = time.perf_counter()
        logger.info(
            "[transcribe][%s] stage=%s complete in %.2f seconds",
            recording_id,
            stage,
            t_h1 - t_h0,
        )

        stage = "callback"
        logger.info(
            "[transcribe][%s] stage=%s starting (%s segment(s))",
            recording_id,
            stage,
            len(final_segments),
        )
        result = RecordingTranscribeResponse(
            recording_id=data["recording_id"],
            transcriptions=final_segments,
        )
        await post_transcription_result(result)
        logger.info("[transcribe][%s] stage=%s complete", recording_id, stage)

        logger.info("[transcribe][%s] background pipeline finished successfully", recording_id)
    except Exception as exc:
        logger.exception(
            "Background transcription pipeline failed recording_id=%s failed_stage=%s error_type=%s error=%s",
            recording_id,
            stage,
            type(exc).__name__,
            exc,
        )
    finally:
        t_cl0 = time.perf_counter()
        shutil.rmtree(work_dir, ignore_errors=True)
        t_cl1 = time.perf_counter()
        logger.info(
            "[transcribe][%s] cleanup complete in %.2f seconds",
            recording_id,
            t_cl1 - t_cl0,
        )

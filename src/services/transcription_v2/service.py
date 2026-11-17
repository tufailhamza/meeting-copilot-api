"""Transcription v2 pipeline: OpenAI diarization (English-only) -> transliterate -> brand fix -> host detect -> callback."""

import asyncio
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.schemas.transcription.transcribe import RecordingTranscribeResponse, TranscriptionSegment
from src.services.transcription.host_detection import detect_and_relabel_host
from src.services.transcription.transcription_callback import post_transcription_result
from src.services.transcription.transliteration import transliterate_transcriptions

_ROOT = Path(__file__).resolve().parents[3]
_OPENAI_DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"
_OPENAI_LANGUAGE = "en"

logger = logging.getLogger(__name__)

_CLUE_BEFORE_APP_OR_WORLD = re.compile(r"(?i)\bclue(\s+)(app|world)\b")


def _replace_clue_with_qlu_before_app_or_world(text: str) -> str:
    return _CLUE_BEFORE_APP_OR_WORLD.sub(lambda m: f"QLU{m.group(1)}{m.group(2)}", text)


def _apply_clue_to_qlu_brand(segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
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
    return s


def _segments_from_openai_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    segments = data.get("segments") or []
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start_s = float(seg.get("start", 0.0) or 0.0)
            end_s = float(seg.get("end", 0.0) or 0.0)
            text = str(seg.get("text", "") or "").strip()
            out.append(
                {
                    "speaker": _speaker_label(seg.get("speaker")),
                    "text": text,
                    "start": start_s,
                    "end": end_s,
                }
            )
    if not out:
        full = str(data.get("text", "") or "").strip()
        if full:
            out.append({"speaker": "A", "text": full, "start": 0.0, "end": 0.0})
    return out


def _run_openai_segments_from_mp3_sync(mp3_path: Path, recording_id: str) -> dict[str, Any]:
    load_dotenv(_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    t_up0 = time.perf_counter()
    with mp3_path.open("rb") as f:
        raw_response = client.audio.transcriptions.with_raw_response.create(
            model=_OPENAI_DIARIZATION_MODEL,
            file=f,
            response_format="diarized_json",
            chunking_strategy="auto",
            language=_OPENAI_LANGUAGE,
        )
    t_up1 = time.perf_counter()
    elapsed_openai = t_up1 - t_up0
    print(
        f"[transcribe_v2][{recording_id}] Upload to OpenAI took {elapsed_openai:.2f} seconds",
        flush=True,
    )
    print(
        f"[transcribe_v2][{recording_id}] OpenAI transcribe + diarize took {elapsed_openai:.2f} seconds "
        "(same wall clock as upload; single API request)",
        flush=True,
    )
    data: dict[str, Any] = raw_response.http_response.json()
    segs = _segments_from_openai_response(data)
    return {"recording_id": recording_id, "transcriptions": segs}


async def run_pipeline_after_prepare_v2(mp3_path: Path, work_dir: Path, recording_id: str) -> None:
    """
    OpenAI diarization (English-only) -> transliterate -> Clue->QLU brand fix -> host -> callback.
    Deletes work_dir when finished.
    """
    try:
        data = await asyncio.to_thread(_run_openai_segments_from_mp3_sync, mp3_path, recording_id)
        segments = [TranscriptionSegment(**x) for x in data["transcriptions"]]
        t_tr0 = time.perf_counter()
        transliterated = await transliterate_transcriptions(segments)
        t_tr1 = time.perf_counter()
        print(
            f"[transcribe_v2][{recording_id}] Transliteration took {t_tr1 - t_tr0:.2f} seconds",
            flush=True,
        )
        t_br0 = time.perf_counter()
        branded = _apply_clue_to_qlu_brand(transliterated)
        t_br1 = time.perf_counter()
        print(
            f"[transcribe_v2][{recording_id}] Post-process text (Clue→QLU) took {t_br1 - t_br0:.2f} seconds",
            flush=True,
        )
        t_h0 = time.perf_counter()
        final_segments = await detect_and_relabel_host(branded)
        t_h1 = time.perf_counter()
        print(
            f"[transcribe_v2][{recording_id}] Host detection took {t_h1 - t_h0:.2f} seconds",
            flush=True,
        )
        result = RecordingTranscribeResponse(
            recording_id=data["recording_id"],
            transcriptions=final_segments,
        )
        await post_transcription_result(result)
    except Exception:
        logger.exception(
            "Background transcription v2 pipeline failed recording_id=%s",
            recording_id,
        )
    finally:
        t_cl0 = time.perf_counter()
        shutil.rmtree(work_dir, ignore_errors=True)
        t_cl1 = time.perf_counter()
        print(
            f"[transcribe_v2][{recording_id}] Cleanup took {t_cl1 - t_cl0:.2f} seconds",
            flush=True,
        )


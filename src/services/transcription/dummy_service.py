"""Local recording test: OpenAI or AssemblyAI (English-only).

The FastAPI route ``POST .../transcribe/dummy`` is commented out in ``src/routes/transcription.py``.
You can still run this file as a script from the project root for CLI tests.
"""

import asyncio
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.services.transcription.audio_preprocess import (
    ALLOWED_MEDIA_FORMATS,
    normalize_media_format,
    transcode_to_mp3,
    validate_media_format,
)

STUB_MESSAGE = "this function transcribes :)"

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RECORDING = _ROOT / "recording.mp3"
_RESPONSE_JSON = _ROOT / "transcription_last_response.json"
_TRANSCRIPT_JSON = _ROOT / "transcript.json"

# AssemblyAI: bilingual list only (max 2 language_codes). No language_detection_options unless language_detection is True; that flag conflicts with language_codes.
_OPENAI_DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"
_ASSEMBLY_LANGUAGE_CODE = "en"


def _resolve_local_input_path(preferred_path: str | None = None) -> Path | None:
    """Resolve local test media path from env var or fallback files."""
    raw_path = (preferred_path or "").strip()
    if not raw_path:
        raw_path = (os.getenv("DUMMY_RECORDING_PATH") or "").strip()
    if raw_path:
        p = Path(raw_path).expanduser()
        if not p.is_absolute():
            p = (_ROOT / p).resolve()
        if not p.is_file():
            print(f"Input media path does not exist: {p}")
            return None
        ext = normalize_media_format(p.suffix)
        try:
            validate_media_format(ext)
        except ValueError:
            allowed = ", ".join(sorted(ALLOWED_MEDIA_FORMATS))
            print(f"Unsupported input extension {ext!r}. Supported: {allowed}")
            return None
        return p

    if _DEFAULT_RECORDING.is_file():
        return _DEFAULT_RECORDING

    for ext in sorted(ALLOWED_MEDIA_FORMATS):
        cand = _ROOT / f"recording{ext}"
        if cand.is_file():
            return cand
    print(
        f"No recording file found. Put one of recording{sorted(ALLOWED_MEDIA_FORMATS)} "
        f"in project root, or set DUMMY_RECORDING_PATH."
    )
    return None


def _prepare_local_mp3_input(input_path: Path) -> tuple[Path, Path | None]:
    """Return MP3 path for transcription. Caller must delete returned temp dir when not None."""
    ext = normalize_media_format(input_path.suffix)
    validate_media_format(ext)
    if ext == ".mp3":
        return input_path, None

    tmpdir = Path(tempfile.mkdtemp(prefix="qlu_dummy_transcribe_"))
    out_mp3 = tmpdir / "normalized.mp3"
    try:
        transcode_to_mp3(input_path, out_mp3)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    return out_mp3, tmpdir


def _run_openai_sync(input_path_arg: str | None = None) -> None:
    load_dotenv(_ROOT / ".env")
    run_started_at = time.perf_counter()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY missing")
        return

    input_path = _resolve_local_input_path(input_path_arg)
    if input_path is None:
        return

    mp3_path: Path
    temp_dir: Path | None = None
    try:
        mp3_path, temp_dir = _prepare_local_mp3_input(input_path)
        client = OpenAI(api_key=api_key)

        t0 = time.perf_counter()
        with mp3_path.open("rb") as f:
            raw_response = client.audio.transcriptions.with_raw_response.create(
                model=_OPENAI_DIARIZATION_MODEL,
                file=f,
                response_format="diarized_json",
                chunking_strategy="auto",
                language="en",
            )

        http = raw_response.http_response
        data: dict[str, Any] = http.json()
        elapsed_s = time.perf_counter() - t0
        total_elapsed_s = time.perf_counter() - run_started_at
        print(f"OpenAI transcription (send → full JSON received): {elapsed_s:.2f} seconds")
        print(f"OpenAI total runtime (start → transcript received): {total_elapsed_s:.2f} seconds")

        payload = {"provider": "openai", **data}
        _RESPONSE_JSON.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Wrote API response to:", _RESPONSE_JSON)
        _TRANSCRIPT_JSON.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Wrote transcript to:", _TRANSCRIPT_JSON)

        print("full text:", (data.get("text") or "").strip())
        segments = data.get("segments") or []
        if not segments:
            print("No segments in response.")
            return

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            speaker = seg.get("speaker", "?")
            text = (seg.get("text") or "").strip()
            print(f"[{start:.2f}-{end:.2f}] Speaker {speaker}: {text}")

    except Exception as exc:
        print("Transcription error:", exc)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _run_assembly_sync(input_path_arg: str | None = None) -> None:
    """Universal-2 + diarization, English-only via language_code=en."""
    load_dotenv(_ROOT / ".env")
    run_started_at = time.perf_counter()

    api_key = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        print("ASSEMBLYAI_API_KEY (or ASSEMBLY_API_KEY) missing")
        return

    input_path = _resolve_local_input_path(input_path_arg)
    if input_path is None:
        return

    try:
        import assemblyai as aai
    except ImportError:
        print("Install AssemblyAI: pip install -U assemblyai")
        return

    aai.settings.api_key = api_key
    aai.settings.http_timeout = 1800.0
    aai.settings.polling_interval = 3.0

    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        speaker_labels=True,
        language_code=_ASSEMBLY_LANGUAGE_CODE,
    )

    mp3_path: Path
    temp_dir: Path | None = None
    try:
        mp3_path, temp_dir = _prepare_local_mp3_input(input_path)
        t0 = time.perf_counter()
        transcript = aai.Transcriber().transcribe(str(mp3_path), config=config)
        elapsed_s = time.perf_counter() - t0
        total_elapsed_s = time.perf_counter() - run_started_at
        print(f"AssemblyAI Universal-2 (submit → completed): {elapsed_s:.2f} seconds")
        print(
            f"AssemblyAI total runtime (start → transcript received): {total_elapsed_s:.2f} seconds"
        )

        if transcript.status == aai.TranscriptStatus.error:
            print("AssemblyAI error:", transcript.error)
            return

        utterances_raw: list[dict[str, Any]] = []
        for u in transcript.utterances or []:
            utterances_raw.append(
                {
                    "speaker": getattr(u, "speaker", None),
                    "text": getattr(u, "text", None),
                    "start": getattr(u, "start", None),
                    "end": getattr(u, "end", None),
                    "confidence": getattr(u, "confidence", None),
                }
            )

        snapshot: dict[str, Any] = {
            "provider": "assembly",
            "request": {
                "language_code": _ASSEMBLY_LANGUAGE_CODE,
            },
            "id": transcript.id,
            "status": str(transcript.status),
            "text": transcript.text,
            "utterances": utterances_raw,
            "language_code": getattr(transcript, "language_code", None),
            "language_codes": getattr(transcript, "language_codes", None),
        }

        print("--- AssemblyAI snapshot (JSON) ---")
        print(json.dumps(snapshot, indent=2, ensure_ascii=False)[:8000])
        print("--- end snapshot (truncated if long) ---")

        _RESPONSE_JSON.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Wrote API response to:", _RESPONSE_JSON)
        _TRANSCRIPT_JSON.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Wrote transcript to:", _TRANSCRIPT_JSON)

        for u in transcript.utterances or []:
            start_ms = getattr(u, "start", 0) or 0
            end_ms = getattr(u, "end", 0) or 0
            sp = getattr(u, "speaker", "?")
            txt = (getattr(u, "text", None) or "").strip()
            print(f"[{start_ms}-{end_ms} ms] Speaker {sp}: {txt}")

    except Exception as exc:
        print("AssemblyAI transcription error:", exc)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


async def dummy_transcribe(provider: str = "openai", input_path: str | None = None) -> str:
    if provider == "openai":
        await asyncio.to_thread(_run_openai_sync, input_path)
    else:
        await asyncio.to_thread(_run_assembly_sync, input_path)
    return STUB_MESSAGE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy local transcription runner")
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Local media path. If set, overrides DUMMY_RECORDING_PATH and recording.<ext> fallback.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "assembly"],
        default="openai",
        help="Transcription provider (default: openai).",
    )
    args = parser.parse_args()

    async def _demo() -> None:
        await dummy_transcribe(provider=args.provider, input_path=args.input_path)

    asyncio.run(_demo())

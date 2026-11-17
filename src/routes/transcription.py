import asyncio
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.config import settings
from src.schemas.transcription.transcribe import (
    RecordingTranscribeRequest,
    TranscribeAcceptedResponse,
)
from src.services.transcription.audio_preprocess import (
    NoAudioStreamError,
    extension_from_content_type,
    prepare_assembly_audio_mp3_paths,
)
from src.services.transcription.service import (
    run_pipeline_after_prepare,
    send_empty_transcription_callback,
)

router = APIRouter()

# --- Dummy local test (`recording.mp3` + OpenAI/AssemblyAI) — disabled ---
# from fastapi import Query
# from src.schemas.transcription.transcribe import TranscribeResponse
# from src.services.transcription.dummy_service import dummy_transcribe
#
# def _normalize_model(model: str | None) -> str:
#     v = (model or "").strip().lower()
#     if v == "openai":
#         return "openai"
#     return "assembly"


@router.post("/transcribe", response_model=TranscribeAcceptedResponse)
async def transcribe(
    body: RecordingTranscribeRequest,
    background_tasks: BackgroundTasks,
) -> TranscribeAcceptedResponse:

    if not settings.transcriber_url.strip() or not settings.m2m_token.strip():
        raise HTTPException(
            status_code=503,
            detail="TRANSCRIBER_URL and M2M_TOKEN must be set for async transcription callback.",
        )

    t0 = time.perf_counter()
    try:
        media_format = extension_from_content_type(body.content_type)
        mp3_path, work_dir = await asyncio.to_thread(
            prepare_assembly_audio_mp3_paths,
            body.signed_url,
            media_format,
        )
    except NoAudioStreamError as exc:
        # A recording with no sound in it is a result, not a failure — and no retry will change
        # it. So the caller gets the ordinary accepted response and the empty transcript reaches
        # them on the callback, which keeps their success path the only path they need. Reported
        # as a 502 instead, it read as this service being broken and nobody looked at the file.
        background_tasks.add_task(
            send_empty_transcription_callback,
            body.recording_id,
            str(exc),
        )
        return TranscribeAcceptedResponse(recording_id=body.recording_id, status="accepted")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Audio download or normalization failed: {exc}",
        ) from exc
    finally:
        elapsed_s = time.perf_counter() - t0
        print(f"transcribe accept (through audio prep) took {elapsed_s:.2f} seconds")

    background_tasks.add_task(
        run_pipeline_after_prepare,
        mp3_path,
        work_dir,
        body.recording_id,
    )
    return TranscribeAcceptedResponse(recording_id=body.recording_id, status="accepted")


# @router.post("/transcribe/dummy", response_model=TranscribeResponse)
# async def transcribe_dummy(
#     model: str | None = Query(
#         default=None,
#         description='Transcription engine: "openai" or "assembly". Defaults to assembly.',
#     ),
# ) -> TranscribeResponse:
#     """Local `recording.mp3` test only (OpenAI or AssemblyAI)."""
#     provider = _normalize_model(model)
#     message = await dummy_transcribe(provider)
#     return TranscribeResponse(message=message, provider=provider)

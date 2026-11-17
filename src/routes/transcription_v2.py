import asyncio
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.config import settings
from src.schemas.transcription.transcribe import (
    RecordingTranscribeRequest,
    TranscribeAcceptedResponse,
)
from src.services.transcription.audio_preprocess import (
    extension_from_content_type,
    prepare_assembly_audio_mp3_paths,
)
from src.services.transcription_v2.service import run_pipeline_after_prepare_v2

router = APIRouter()


@router.post("/transcribe_v2", response_model=TranscribeAcceptedResponse)
async def transcribe_v2(
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
        print(f"transcribe_v2 accept (through audio prep) took {elapsed_s:.2f} seconds")

    background_tasks.add_task(
        run_pipeline_after_prepare_v2,
        mp3_path,
        work_dir,
        body.recording_id,
    )
    return TranscribeAcceptedResponse(recording_id=body.recording_id, status="accepted")

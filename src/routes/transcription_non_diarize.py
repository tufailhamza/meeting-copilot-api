import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException

from src.schemas.transcription.transcribe import (
    NonDiarizeTranscribeResponse,
    RecordingTranscribeRequest,
)
from src.services.transcription.audio_preprocess import (
    extension_from_content_type,
    prepare_assembly_audio_mp3_paths,
)
from src.services.transcription_non_diarize.service import run_non_diarize_transcribe

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/non_diarize_transcribe", response_model=NonDiarizeTranscribeResponse)
async def non_diarize_transcribe(body: RecordingTranscribeRequest) -> NonDiarizeTranscribeResponse:
    recording_id = body.recording_id
    t_total0 = time.perf_counter()
    logger.info(
        "[non_diarize_transcribe][%s] request started content_type=%s",
        recording_id,
        body.content_type,
    )

    try:
        try:
            media_format = extension_from_content_type(body.content_type)
        except ValueError as exc:
            logger.error(
                "[non_diarize_transcribe][%s] invalid content_type=%r error=%s",
                recording_id,
                body.content_type,
                exc,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        t_prep0 = time.perf_counter()
        try:
            mp3_path, work_dir = await asyncio.to_thread(
                prepare_assembly_audio_mp3_paths,
                body.signed_url,
                media_format,
            )
        except ValueError as exc:
            logger.error(
                "[non_diarize_transcribe][%s] audio prep validation failed: %s",
                recording_id,
                exc,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception(
                "[non_diarize_transcribe][%s] audio prep runtime failed",
                recording_id,
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "[non_diarize_transcribe][%s] audio download or normalization failed",
                recording_id,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Audio download or normalization failed: {exc}",
            ) from exc
        finally:
            logger.info(
                "[non_diarize_transcribe][%s] audio prep took %.2f seconds",
                recording_id,
                time.perf_counter() - t_prep0,
            )

        t_tx0 = time.perf_counter()
        try:
            transcript_text = await asyncio.to_thread(
                run_non_diarize_transcribe,
                mp3_path,
                work_dir,
                recording_id,
            )
        except ValueError as exc:
            logger.error(
                "[non_diarize_transcribe][%s] transcription config/error: %s",
                recording_id,
                exc,
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "[non_diarize_transcribe][%s] transcription failed",
                recording_id,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Transcription failed: {exc}",
            ) from exc
        finally:
            logger.info(
                "[non_diarize_transcribe][%s] transcribe+cleanup took %.2f seconds",
                recording_id,
                time.perf_counter() - t_tx0,
            )

        logger.info(
            "[non_diarize_transcribe][%s] TOTAL request took %.2f seconds chars=%s",
            recording_id,
            time.perf_counter() - t_total0,
            len(transcript_text),
        )
        return NonDiarizeTranscribeResponse(
            recording_id=recording_id,
            transcript_text=transcript_text,
        )
    except HTTPException:
        logger.error(
            "[non_diarize_transcribe][%s] TOTAL request failed after %.2f seconds",
            recording_id,
            time.perf_counter() - t_total0,
        )
        raise
    except Exception:
        logger.exception(
            "[non_diarize_transcribe][%s] TOTAL request unexpected failure after %.2f seconds",
            recording_id,
            time.perf_counter() - t_total0,
        )
        raise

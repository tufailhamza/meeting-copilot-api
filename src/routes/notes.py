import asyncio
import logging

from fastapi import APIRouter, HTTPException

from src.config import settings
from src.schemas.notes.generate_notes import (
    GenerateNotesRequest,
    GenerateNotesResponse,
)
from src.schemas.notes.generate_summary import (
    GenerateSummaryRequest,
    GenerateSummaryResponse,
)
from src.services.notes.notes_generation import generate_notes_sync
from src.services.notes.summary import generate_summary_sync

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/generate_summary",
    response_model=GenerateSummaryResponse
)
async def generate_summary(body: GenerateSummaryRequest) -> GenerateSummaryResponse:

    if not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY must be set to generate summaries.",
        )

    try:
        summary_text = await asyncio.to_thread(
            generate_summary_sync,
            body.recording_id,
            list(body.transcriptions),
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Summary generation failed for recording_id=%s", body.recording_id)
        raise HTTPException(
            status_code=502,
            detail=f"Summary generation failed: {exc}",
        ) from exc

    return GenerateSummaryResponse(recording_id=body.recording_id, summary=summary_text)


@router.post("/generate_notes", response_model=GenerateNotesResponse)
async def generate_notes(body: GenerateNotesRequest) -> GenerateNotesResponse:
    if not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY must be set to generate notes.",
        )

    try:
        return await asyncio.to_thread(
            generate_notes_sync,
            body.recording_id,
            list(body.transcriptions),
            body.template,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Notes generation failed for recording_id=%s", body.recording_id)
        raise HTTPException(
            status_code=502,
            detail=f"Notes generation failed: {exc}",
        ) from exc

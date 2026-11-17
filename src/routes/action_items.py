import asyncio
import logging

from fastapi import APIRouter, HTTPException

from src.config import settings
from src.schemas.action_items.extract_action_items import (
    ExtractActionItemsRequest,
    ExtractActionItemsResponse,
)
from src.services.action_items.extract import extract_action_items_sync

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/process_transcripts",
    response_model=ExtractActionItemsResponse,
    response_model_exclude_none=True,
)
async def process_transcripts(
    body: ExtractActionItemsRequest,
) -> ExtractActionItemsResponse:
    if body.user_metadata and not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY must be set to extract action items.",
        )

    try:
        return await asyncio.to_thread(extract_action_items_sync, body)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Action items extraction failed")
        raise HTTPException(
            status_code=502,
            detail=f"Action items extraction failed: {exc}",
        ) from exc

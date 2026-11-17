from fastapi import APIRouter, Request

from src.config import settings
from src.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        request_id=getattr(request.state, "request_id", None),
    )

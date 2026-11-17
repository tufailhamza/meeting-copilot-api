from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings, setup_logging
from src.live.routes_live import router as live_router
from src.middleware import (
    PreservePostBodyForValidationDebugMiddleware,
    RequestContextMiddleware,
    request_validation_exception_handler,
)
from src.routes import action_items as action_items_routes
from src.routes import health as health_routes
from src.routes import notes as notes_routes
from src.routes import retrieval as retrieval_routes
from src.routes import templates as templates_routes
from src.routes import transcription as transcription_routes
from src.routes import transcription_non_diarize as transcription_non_diarize_routes
from src.routes import transcription_v2 as transcription_v2_routes


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    yield


app = FastAPI(
    title=settings.app_name,
    description="QLU Transcriber API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_exception_handler(RequestValidationError, request_validation_exception_handler)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PreservePostBodyForValidationDebugMiddleware)

api = settings.api_prefix

app.include_router(health_routes.router, prefix=api, tags=["health"])
app.include_router(
    transcription_routes.router,
    prefix=f"{api}/transcription",
    tags=["transcription"],
)
app.include_router(
    transcription_v2_routes.router,
    prefix=f"{api}/transcription",
    tags=["transcription"],
)
app.include_router(
    transcription_non_diarize_routes.router,
    prefix=f"{api}/transcription",
    tags=["transcription"],
)
app.include_router(
    notes_routes.router,
    prefix=f"{api}/notes",
    tags=["notes"],
)
app.include_router(
    action_items_routes.router,
    prefix=f"{api}/meetings",
    tags=["meetings"],
)
app.include_router(
    templates_routes.router,
    prefix=f"{api}/templates",
    tags=["templates"],
)
app.include_router(
    retrieval_routes.router,
    prefix=f"{api}/retrieval",
    tags=["retrieval"],
)
app.include_router(
    live_router,
    prefix=f"{api}/live",
    tags=["live"],
)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "docs": "/docs"}


if __name__ == "__main__":
    # Set DEBUG=true in .env for auto-reload during local development.
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

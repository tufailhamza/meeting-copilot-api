import logging
import uuid

from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import settings

REQUEST_ID_HEADER = "X-Request-ID"
logger = logging.getLogger(__name__)

# Log at most this many UTF-8 bytes of the raw JSON body on 422 (signed URLs can be huge).
_BODY_PREVIEW_MAX_BYTES = 24_000


def _json_body_capture_paths() -> frozenset[str]:
    api = (settings.api_prefix or "").strip().rstrip("/")
    base = f"/{api}".replace("//", "/") if api else ""
    return frozenset(
        {
            f"{base}/transcription/transcribe".replace("//", "/"),
            f"{base}/transcription/transcribe_v2".replace("//", "/"),
            f"{base}/transcription/non_diarize_transcribe".replace("//", "/"),
            f"{base}/notes/generate_summary".replace("//", "/"),
            f"{base}/notes/generate_notes".replace("//", "/"),
        }
    )


class PreservePostBodyForValidationDebugMiddleware(BaseHTTPMiddleware):
    """For selected POST JSON routes, buffer the raw body and replay it so FastAPI can still parse it.

    ``request.state.raw_request_body`` is then available in the ``RequestValidationError`` handler
    (the default body stream is often already consumed when validation fails).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        paths = _json_body_capture_paths()
        if request.method == "POST" and request.url.path in paths:
            body = await request.body()

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request = Request(request.scope, receive)
            request.state.raw_request_body = body

        return await call_next(request)


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    raw: bytes = getattr(request.state, "raw_request_body", None) or b""
    preview = raw[:_BODY_PREVIEW_MAX_BYTES].decode("utf-8", errors="replace")
    if len(raw) > _BODY_PREVIEW_MAX_BYTES:
        preview += f"\n... (truncated, total bytes={len(raw)})"

    logger.warning(
        "422 RequestValidationError | request_id=%s | %s %s | errors=%s | body=%s",
        getattr(request.state, "request_id", None),
        request.method,
        request.url.path,
        exc.errors(),
        preview,
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

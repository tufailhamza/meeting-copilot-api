"""POST final transcription payload to the configured transcriber (Zoro) URL."""

import json
import logging
import time

import httpx

from src.config import settings
from src.schemas.transcription.transcribe import RecordingTranscribeResponse

logger = logging.getLogger(__name__)

_CALLBACK_PATH = "/m2m/zoro/recording-transcriptions-details"
_LOG_BODY_MAX_CHARS = 50_000


def _callback_url() -> str:
    base = (settings.transcriber_url or "").strip().rstrip("/")
    return f"{base}{_CALLBACK_PATH}"


async def post_transcription_result(payload: RecordingTranscribeResponse) -> None:
    url = _callback_url()
    token = (settings.m2m_token or "").strip()
    if not url or url == _CALLBACK_PATH:
        raise ValueError("TRANSCRIBER_URL is not set or empty")
    if not token:
        raise ValueError("M2M_TOKEN is not set or empty")

    logger.info(
        "[transcribe][%s] callback POST starting url=%s segments=%s",
        payload.recording_id,
        url,
        len(payload.transcriptions),
    )

    body = payload.model_dump()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    body_preview = json.dumps(body, indent=2, ensure_ascii=False)
    if len(body_preview) > _LOG_BODY_MAX_CHARS:
        body_preview = body_preview[:_LOG_BODY_MAX_CHARS] + "\n... (truncated for log)"
    logger.info(
        "Callback POST url=%s auth=Bearer(len=%s)\n%s",
        url,
        len(token),
        body_preview,
    )
    print(
        f"[callback] POST {url}\nAuthorization: Bearer <redacted len={len(token)}>\n{body_preview}",
        flush=True,
    )

    async with httpx.AsyncClient(timeout=600.0) as client:
        t_cb0 = time.perf_counter()
        r = await client.post(url, json=body, headers=headers)
        t_cb1 = time.perf_counter()
        logger.info(
            "[transcribe][%s] callback POST finished in %.2f seconds status=%s",
            payload.recording_id,
            t_cb1 - t_cb0,
            r.status_code,
        )
        r.raise_for_status()
    logger.info("Posted transcription result for recording_id=%s status=%s", payload.recording_id, r.status_code)

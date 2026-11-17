"""Self-documenting contract endpoint.

Exposes the live WebSocket message contract as JSON Schema over plain HTTP so it shows up in
``/docs`` and can be handed to the Node backend team as the integration spec. WebSocket
payloads are not part of FastAPI's auto-generated OpenAPI, so this endpoint makes the
inbound/outbound message shapes discoverable.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.live.schemas_live import inbound_json_schema, outbound_json_schema

router = APIRouter()


@router.get("/schema", tags=["live"], summary="Live WebSocket message contract (JSON Schema)")
async def live_schema() -> dict:
    return {
        "transport": "websocket",
        "endpoint": "/api/live/ws",
        "inbound": inbound_json_schema(),
        "outbound": outbound_json_schema(),
    }

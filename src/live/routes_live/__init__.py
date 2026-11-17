"""Live feature routers (WebSocket + contract). Exposes a single ``router`` mounted by main.py."""

from fastapi import APIRouter

from src.live.routes_live import call_prep_chat, call_prep_load, contract, ws

router = APIRouter()
router.include_router(ws.router)
router.include_router(contract.router)
router.include_router(call_prep_chat.router)
router.include_router(call_prep_load.router)

__all__ = ["router"]

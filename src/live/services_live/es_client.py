"""Elasticsearch client seam.

Thin ``httpx`` wrapper rather than the ``elasticsearch`` package: the only calls we need are a
document GET and a search, and httpx is already a dependency.

Auth is an API key sent as ``ApiKey base64("<ES_ID>:<ES_KEY>")`` — ``ES_ID`` is the key id and
``ES_KEY`` the key itself, matching the working ``curl`` in ``.env``.
"""

from __future__ import annotations

import base64
import logging
from typing import Any
from urllib.parse import quote

import httpx

from src.live.config_live import es_settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
PEOPLE_INDEX = "people"


class ESNotConfigured(RuntimeError):
    """ES_URL / ES_ID / ES_KEY are missing from the environment."""


def _auth_header() -> str:
    raw = f"{es_settings.es_id.strip()}:{es_settings.es_key.strip()}"
    return "ApiKey " + base64.b64encode(raw.encode()).decode()


def _headers() -> dict[str, str]:
    return {"Authorization": _auth_header(), "Content-Type": "application/json"}


def _base_url() -> str:
    return es_settings.es_url.strip().rstrip("/")


def _require_configured() -> None:
    if not es_settings.configured:
        raise ESNotConfigured(
            "ES_URL, ES_ID and ES_KEY must all be set to query Elasticsearch."
        )


async def ping() -> dict[str, Any]:
    """Hit ``GET /`` — cluster banner if the URL and API key are good."""
    _require_configured()
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        resp = await client.get(_base_url() + "/", headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def get_document(index: str, doc_id: str) -> dict[str, Any] | None:
    """``GET /<index>/_doc/<doc_id>``. ``None`` when the document does not exist.

    A 404 is a real answer here (no such doc), not a failure — every other non-2xx raises.
    """
    _require_configured()
    url = f"{_base_url()}/{quote(index)}/_doc/{quote(str(doc_id))}"
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        resp = await client.get(url, headers=_headers())

    if resp.status_code == 404:
        logger.info("[live] es get_document not_found index=%s doc_id=%s", index, doc_id)
        return None
    resp.raise_for_status()
    return resp.json()


async def search(index: str, body: dict[str, Any]) -> dict[str, Any]:
    """``POST /<index>/_search`` with a raw query body."""
    _require_configured()
    url = f"{_base_url()}/{quote(index)}/_search"
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        resp = await client.post(url, headers=_headers(), json=body)
    resp.raise_for_status()
    return resp.json()


async def mget(index: str, ids: list[str]) -> dict[str, str | None]:
    """``POST /<index>/_mget`` — one round trip for many ids.

    Returns ``{id: _source}`` for the ids that exist; missing ones are absent from the map
    rather than raising, since a stale id in a payload is normal.
    """
    _require_configured()
    wanted = [str(i).strip() for i in ids if str(i).strip()]
    if not wanted:
        return {}

    url = f"{_base_url()}/{quote(index)}/_mget"
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        resp = await client.post(url, headers=_headers(), json={"ids": wanted})
    resp.raise_for_status()

    out: dict[str, Any] = {}
    for doc in resp.json().get("docs") or []:
        if isinstance(doc, dict) and doc.get("found"):
            out[str(doc.get("_id"))] = doc.get("_source") or {}
    return out


async def search_by_terms(
    index: str,
    field: str,
    values: list[str],
    *,
    size: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch documents whose ``field`` matches any of ``values``; returns their ``_source``s.

    Used for lookups keyed on something other than the document id (e.g. companies arrive as
    ``li_universalname``, not an ES id).
    """
    _require_configured()
    wanted = [str(v).strip() for v in values if str(v).strip()]
    if not wanted:
        return []

    body = {"query": {"terms": {field: wanted}}, "size": size or len(wanted)}
    result = await search(index, body)
    hits = (result.get("hits") or {}).get("hits") or []
    return [h.get("_source") or {} for h in hits if isinstance(h, dict)]

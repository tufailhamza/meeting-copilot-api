from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from src.live.config_live import company_filter_settings

logger = logging.getLogger(__name__)

# Entity aliases applied before any lookup. Resolving every spoken form to the one name the
# service actually holds keeps a single company from appearing as two, and — more importantly —
# stops a lookup missing because of how someone happened to say or spell it.
#
# Keys are matched on a squashed form (lowercased, non-alphanumerics dropped), so "Clue.AI",
# "clue ai", "Clue-AI" and "CLUEAI" all resolve through one entry rather than needing a row each.
_COMPANY_ALIASES: dict[str, str] = {
    "clue": "qlu.ai",
    "clueai": "qlu.ai",
    "qlu": "qlu.ai",
    "qluai": "qlu.ai",
}

_ALIAS_KEY_RE = re.compile(r"[^a-z0-9]+")


def _alias_key(name: str) -> str:
    return _ALIAS_KEY_RE.sub("", (name or "").casefold())


def canonical_company_name(company_name: str) -> str:
    """Return the canonical name for ``company_name``, or the input unchanged.

    Idempotent: feeding it a canonical name returns that name, so it is safe to apply more than
    once along a path.
    """
    name = (company_name or "").strip()
    if not name:
        return name
    return _COMPANY_ALIASES.get(_alias_key(name), name)


@dataclass
class CompanyFilterResult:
    company_name: str
    es_id: str | None
    li_urn: str | None
    li_name: str | None
    li_universalname: str | None
    li_industries: list[str]


async def map_company_name(company_name: str) -> CompanyFilterResult | None:
    """Resolve ``company_name`` via the external company-filter service.

    Returns a normalized ``CompanyFilterResult`` or ``None`` if the service is not configured,
    the request fails, or the response is missing the expected shape.
    """
    if not company_filter_settings.configured:
        logger.info(
            "[live] company_filter skipped company_name=%r reason=not_configured",
            company_name,
        )
        return None

    name = canonical_company_name(company_name)
    if not name:
        return None

    base = company_filter_settings.company_filter_base_url.rstrip("/")
    url = f"{base}/map_company"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": company_filter_settings.company_filter_token,
    }
    payload: dict[str, Any] = {
        "company_name": name,
        "user_query": "",
        "relevant_industries": [],
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except Exception:  # noqa: BLE001
        logger.exception(
            "[live] company_filter request failed company_name=%r url=%s",
            name,
            url,
        )
        return None

    if resp.status_code // 100 != 2:
        logger.warning(
            "[live] company_filter non_2xx_status company_name=%r status=%s body=%s",
            name,
            resp.status_code,
            resp.text[:500],
        )
        return None

    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        logger.exception("[live] company_filter invalid_json company_name=%r", name)
        return None

    source = data.get("source") or {}

    result = CompanyFilterResult(
        company_name=data.get("company_name") or name,
        es_id=str(data.get("es_id") or "") or None,
        li_urn=source.get("li_urn"),
        li_name=source.get("li_name"),
        li_universalname=source.get("li_universalname"),
        li_industries=list(source.get("li_industries") or []),
    )

    # A company the service *found* is surfaced even when it came back without LinkedIn
    # identifiers. Dropping it here used to make a successful lookup look like a miss, which is
    # indistinguishable to the host from the company simply not existing.
    if not (result.li_urn or result.li_name or result.li_universalname):
        logger.info(
            "[live] company_filter resolved_without_identifiers company_name=%r raw=%r",
            result.company_name,
            name,
        )

    return result


async def generate_similar_companies(user_query: str) -> list[CompanyFilterResult]:
    """Call the user-query-based company generation endpoint.

    Returns a list of ``CompanyFilterResult`` rows parsed from ``data: {...}`` lines, or an
    empty list on error.
    """
    results: list[CompanyFilterResult] = []

    if not company_filter_settings.configured:
        logger.info(
            "[live] company_filter generation skipped reason=not_configured user_query=%r",
            user_query,
        )
        return results

    query = (user_query or "").strip()
    if not query:
        return results

    base = company_filter_settings.company_filter_base_url.rstrip("/")
    url = f"{base}/test_company_generation_from_userquery"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": company_filter_settings.company_filter_token,
    }
    payload: dict[str, Any] = {"userquery": query}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except Exception:  # noqa: BLE001
        logger.exception(
            "[live] company_filter generation request failed user_query=%r url=%s",
            query,
            url,
        )
        return results

    if resp.status_code // 100 != 2:
        logger.warning(
            "[live] company_filter generation non_2xx_status status=%s body=%s",
            resp.status_code,
            resp.text[:500],
        )
        return results

    text = resp.text or ""
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw:
            continue
        try:
            item = httpx.Response(200, text=raw).json()
        except Exception:  # noqa: BLE001
            logger.exception("[live] company_filter generation invalid_json line=%r", line)
            continue
        es = (item.get("es_data") or {}) if isinstance(item, dict) else {}
        if not es:
            continue
        name = es.get("name") or item.get("list")
        if not name:
            continue
        result = CompanyFilterResult(
            company_name=str(name),
            es_id=str(es.get("es_id") or "") or None,
            li_urn=es.get("urn"),
            li_name=es.get("name"),
            li_universalname=es.get("universalName"),
            li_industries=[es.get("industry")] if es.get("industry") else [],
        )
        results.append(result)

    return results


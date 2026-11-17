from __future__ import annotations

"""Web search helper for host chat using OpenAI's search-capable model."""

import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import APIError, AsyncOpenAI

from src.config import settings as app_settings
from src.live.services_live.projection import MeetingSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchResult:
    reply: str
    sources: list[dict[str, Any]]


async def run_web_search(query: str, session: MeetingSession) -> WebSearchResult:
    """Call OpenAI's search-capable chat model to answer a query with web results."""
    api_key = app_settings.openai_api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = AsyncOpenAI(api_key=api_key)

    # Keep prompt short and focused; routing agent already decided we need web search.
    # ``session.context`` is keyed by user id, so the meeting-level values have to be gathered
    # across users rather than read off the top level.
    meeting_context_parts: list[str] = []
    org_names = {
        str(ctx.get("org_name")).strip()
        for ctx in session.context.values()
        if ctx.get("org_name")
    }
    if org_names:
        meeting_context_parts.append(f"Org: {', '.join(sorted(org_names))}")
    meeting_types = {
        str(ctx.get("meeting_type")).strip()
        for ctx in session.context.values()
        if ctx.get("meeting_type")
    }
    if meeting_types:
        meeting_context_parts.append(f"Meeting type: {', '.join(sorted(meeting_types))}")

    meeting_context = " | ".join(meeting_context_parts) if meeting_context_parts else ""

    system_prompt = (
        "You are assisting a sales or recruiting host during a live call.\n"
        "Use your built-in web search to answer the user's query in 1–3 plain sentences.\n"
        "Focus only on the specific fact they asked for (for example, a company's revenue for a given year).\n"
        "Do NOT include stock price, ticker details, or general market information unless explicitly requested.\n"
        "Do NOT mention any website names, URLs, search engines, or model/provider names.\n"
        "Reply as simple plain text only, as if you already knew the information.\n"
    )

    user_content = f"Query: {query}"
    if meeting_context:
        user_content += f"\nMeeting context: {meeting_context}"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
    except APIError as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"OpenAI web search error: {exc}") from exc

    message = response.choices[0].message
    raw_reply = (message.content or "").strip()

    # Keep only the first paragraph to avoid trailing sections (e.g. stock quotes).
    first_block = raw_reply.split("\n\n", 1)[0].strip()

    # Basic cleanup: strip markdown artifacts and any stray links if present.
    reply_text = first_block
    # Remove markdown link syntax completely.
    reply_text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)[^)]*\)",
        r"\1",
        reply_text,
    )
    # Remove bare URLs.
    reply_text = re.sub(r"https?://\S+", "", reply_text)
    # Remove markdown headings and bullet markers if any slipped through.
    reply_text = re.sub(r"^#+\s*", "", reply_text, flags=re.MULTILINE)
    reply_text = re.sub(r"^\s*[-*]\s+", "", reply_text, flags=re.MULTILINE)
    # Collapse extra spaces created by removals.
    reply_text = re.sub(r"\s{2,}", " ", reply_text).strip()

    # Best-effort extraction of URLs mentioned in the reply for metadata.
    urls = re.findall(r"https?://\S+", raw_reply)
    sources: list[dict[str, Any]] = []
    for url in urls[:3]:
        cleaned = url.rstrip(").,]")
        sources.append({"title": None, "url": cleaned})

    return WebSearchResult(reply=reply_text, sources=sources)


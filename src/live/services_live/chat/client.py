"""ChatClient: structured chat turn inference, streamed as the model writes.

The reply is one field of a structured response, so raw streaming would leak JSON to the host.
Instead we follow the SDK's *parsed* deltas and emit only the growth of ``reply`` — real
token-by-token output with the structured fields still intact when the turn finishes.

Previously the whole completion was generated first and then replayed word by word, so the host
stared at nothing for the entire generation and the ``chat_token`` frames were decorative.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from openai import APIError, AsyncOpenAI
from pydantic_core import from_json

from src.config import settings as app_settings
from src.live.config_live import live_settings
from src.live.services_live.chat.context import ChatContext
from src.live.services_live.chat.prompts import CHAT_SYSTEM_PROMPT
from src.live.services_live.chat.schemas import ChatTurnResult

logger = logging.getLogger(__name__)

# Called with each new piece of reply text as it is generated.
DeltaHandler = Callable[[str], Awaitable[None]]
# Given the partially-parsed turn, decides whether its reply is worth streaming. Routing fields
# are emitted before ``reply``, so this is answerable before any reply text exists.
EmitPredicate = Callable[[dict], bool]


async def _ignore_delta(_: str) -> None:
    return None


class ChatClient(ABC):
    @abstractmethod
    async def stream_turn(
        self,
        ctx: ChatContext,
        user_text: str,
        on_delta: DeltaHandler,
        should_emit: EmitPredicate | None = None,
    ) -> ChatTurnResult:
        """Generate a turn, invoking ``on_delta`` with reply text as it arrives.

        ``should_emit`` is consulted against the partial parse before any reply text is streamed;
        returning False suppresses streaming for this turn (the result is still returned).
        """

    async def complete_turn(self, ctx: ChatContext, user_text: str) -> ChatTurnResult:
        """Non-streaming convenience wrapper."""
        return await self.stream_turn(ctx, user_text, _ignore_delta)


class OpenAIChatClient(ChatClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is not set")
        self._model = model.strip()
        self._client = client or AsyncOpenAI(api_key=api_key.strip())

    def _messages(self, ctx: ChatContext, user_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"SESSION_CONTEXT_JSON:\n{ctx.to_prompt_block()}\n\n"
                    f"HOST_MESSAGE:\n{user_text}"
                ),
            },
        ]

    async def stream_turn(
        self,
        ctx: ChatContext,
        user_text: str,
        on_delta: DeltaHandler,
        should_emit: EmitPredicate | None = None,
    ) -> ChatTurnResult:
        emitted = 0
        suppressed = False
        try:
            async with self._client.beta.chat.completions.stream(
                model=self._model,
                messages=self._messages(ctx, user_text),
                response_format=ChatTurnResult,
                temperature=0.35,
            ) as stream:
                async for event in stream:
                    if getattr(event, "type", None) != "content.delta":
                        continue
                    data = _partial_turn(event)
                    if should_emit is not None and not suppressed:
                        # Routing fields arrive before ``reply``, so this is decided before the
                        # host sees anything.
                        suppressed = not should_emit(data)
                    if suppressed:
                        continue
                    reply = data.get("reply") or ""
                    if isinstance(reply, str) and len(reply) > emitted:
                        await on_delta(reply[emitted:])
                        emitted = len(reply)
                completion = await stream.get_final_completion()
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no parsed chat result")

        # The model may finish the JSON after the reply field; flush anything not yet emitted so
        # the streamed text always matches the final message. Skipped when suppressed — a
        # superseded reply must never reach the host.
        if not suppressed and len(parsed.reply) > emitted:
            await on_delta(parsed.reply[emitted:])
        return parsed


def _partial_turn(event: object) -> dict:
    """Parse the accumulated JSON *including* a half-written ``reply``.

    The SDK's own ``event.parsed`` uses a partial mode that **discards incomplete trailing
    strings**, so ``reply`` does not appear at all until the model finishes writing it — which
    made "streaming" arrive as one lump at the end. Re-parsing the raw snapshot with
    ``trailing-strings`` exposes the field as it grows, which is the whole point.
    """
    content = None
    snapshot = getattr(event, "snapshot", None)
    try:
        content = snapshot.choices[0].message.content  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        content = None
    if content:
        try:
            data = from_json(content.encode("utf-8"), allow_partial="trailing-strings")
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            # Older pydantic_core without trailing-strings support: fall through.
            pass
    parsed = getattr(event, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if parsed is None:
        return {}
    return getattr(parsed, "__dict__", {}) or {}


class FakeChatClient(ChatClient):
    """Deterministic chat client for tests; streams its canned reply word by word."""

    def __init__(self, responses: list[ChatTurnResult] | None = None) -> None:
        self._responses = list(responses or [])
        self._idx = 0
        self.calls = 0

    async def stream_turn(
        self,
        ctx: ChatContext,
        user_text: str,
        on_delta: DeltaHandler,
        should_emit: EmitPredicate | None = None,
    ) -> ChatTurnResult:
        self.calls += 1
        if self._idx < len(self._responses):
            result = self._responses[self._idx]
            self._idx += 1
        else:
            result = ChatTurnResult(reply=f"Echo: {user_text[:120]}")
        if should_emit is None or should_emit(result.model_dump()):
            for delta in iter_reply_deltas(result.reply):
                await on_delta(delta)
        return result


def iter_reply_deltas(text: str):
    """Split a reply into word-sized deltas (used by the fake client and by tests)."""
    words = text.split()
    if not words:
        if text:
            yield text
        return
    for i, word in enumerate(words):
        yield word if i == 0 else f" {word}"


def build_chat_client() -> ChatClient | None:
    if not app_settings.openai_api_key.strip():
        return None
    return OpenAIChatClient(
        api_key=app_settings.openai_api_key,
        model=live_settings.openai_chat_model,
    )

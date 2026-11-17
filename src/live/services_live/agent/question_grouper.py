"""Group semantically-identical prep questions across users.

Each user brings their own prep, and two users' copies of the same question are rarely worded
identically: *"What is your ATS?"* and *"Which ATS do you use?"* are one question to a human and
two questions to a character comparison. Getting that wrong means the agent sees a duplicate and
has to mark each copy separately.

Deciding sameness is a language judgment, so a model makes it — once per session, in the
background, after the socket is already live. Until it resolves (or if it fails), the session falls
back to exact-text matching, which is conservative: it over-splits rather than wrongly merging.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings as app_settings
from src.live.config_live import live_settings
from src.live.services_live.events import QuestionGroupMember, QuestionGroupsResolved

logger = logging.getLogger(__name__)

GROUPING_SYSTEM_PROMPT = """You decide which prep questions mean the same thing as one target question.

You receive ONE target question and a list of candidate questions, each with a `key`. Different
meeting participants prepared their own lists, so the same question often appears several times in
different words.

Return the keys of the candidates that mean the **same** as the target — asking one would fully
satisfy the other: same subject, same information sought. Wording, length and politeness do not
matter; meaning does.

Do NOT return a candidate that is merely on the same topic. "What is your ATS?" and "How much does
your ATS cost?" concern the same system but seek different information. "How large is the finance
team?" and "How many direct reports will the CFO have?" are likewise different questions.

Return an empty list when nothing matches. That is the common answer and it is always safe — a
wrong match silently attaches one question's answer to a different question, which is far worse
than leaving a duplicate.
"""


class QuestionGrouping(BaseModel):
    """Groups of question keys that mean the same thing."""

    model_config = ConfigDict(extra="forbid")

    groups: list[list[str]] = Field(default_factory=list)


class QuestionMatches(BaseModel):
    """Which candidates mean the same as one target question."""

    model_config = ConfigDict(extra="forbid")

    matching_keys: list[str] = Field(default_factory=list)


def encode_key(user_id: str, question_id: str) -> str:
    return f"{user_id}::{question_id}"


def decode_key(key: str) -> tuple[str, str] | None:
    user_id, sep, question_id = key.partition("::")
    if not sep or not user_id or not question_id:
        return None
    return user_id, question_id


def verify_group_members(
    members: list[QuestionGroupMember],
    texts: dict[tuple[str, str], str],
) -> bool:
    """Cheap sanity check on a resolved group, used at mark time rather than at grouping time.

    Grouping is decided once, early, by a model. Everything downstream trusts it for the rest of
    the session, so a single bad pairing silently records one question's answer against another —
    which is exactly what a host notices and we do not. This is not a similarity score: it only
    rejects the shape that cannot be legitimate, one user appearing twice in the same group.
    """
    seen: set[str] = set()
    for member in members:
        if member.user_id in seen:
            return False
        seen.add(member.user_id)
    return True


def build_grouping_event(
    grouping: QuestionGrouping,
    known_keys: dict[str, tuple[str, str]],
) -> QuestionGroupsResolved:
    """Turn model output into an event, dropping anything it invented and adding what it missed."""
    groups: list[list[QuestionGroupMember]] = []
    seen: set[str] = set()
    for raw_group in grouping.groups:
        members: list[QuestionGroupMember] = []
        users_in_group: set[str] = set()
        for key in raw_group:
            resolved = known_keys.get(key)
            if resolved is None or key in seen:
                continue
            # One user never holds the same question twice, so two of their questions inside one
            # group is a bookkeeping error, not a real match. Keeping it would attach one of their
            # questions' answers to the other — the failure this validation exists to stop. The
            # first is kept; the rest fall through to standing alone.
            if resolved[0] in users_in_group:
                logger.warning(
                    "[live] grouping rejected duplicate user in group user_id=%s question_id=%s",
                    resolved[0],
                    resolved[1],
                )
                continue
            users_in_group.add(resolved[0])
            seen.add(key)
            members.append(QuestionGroupMember(user_id=resolved[0], question_id=resolved[1]))
        if members:
            groups.append(members)
    # Any question the model failed to mention stands alone rather than vanishing.
    for key, (user_id, question_id) in known_keys.items():
        if key not in seen:
            groups.append([QuestionGroupMember(user_id=user_id, question_id=question_id)])
    return QuestionGroupsResolved(groups=groups)


class QuestionGrouper(ABC):
    @abstractmethod
    async def group(self, questions: dict[str, str]) -> QuestionGrouping:
        """``questions`` maps key -> text."""


class OpenAIQuestionGrouper(QuestionGrouper):
    def __init__(self, *, api_key: str, model: str, client: AsyncOpenAI | None = None) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is not set")
        self._model = model.strip()
        self._client = client or AsyncOpenAI(api_key=api_key.strip())

    async def group(self, questions: dict[str, str]) -> QuestionGrouping:
        """Match each question against the others, then union the pairs into groups.

        One call per question rather than one call over all of them. Asking a model to re-emit
        every key exactly once across ~30 items is a bookkeeping task it quietly gets wrong, and a
        shuffled-but-valid key rewires which question a canonical points at for the whole session —
        one question's answer ends up recorded against another. Anchoring each call to a single
        question removes that surface: the worst a bad call can do is miss a match.
        """
        keys = list(questions)
        if len(keys) < 2:
            return QuestionGrouping(groups=[[k] for k in keys])

        async def matches_for(anchor: str) -> list[str]:
            candidates = [
                {"key": k, "text": questions[k]} for k in keys if k != anchor
            ]
            try:
                response = await self._client.beta.chat.completions.parse(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": GROUPING_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"target": questions[anchor], "candidates": candidates},
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    response_format=QuestionMatches,
                    temperature=0.0,
                )
            except APIError as exc:
                raise RuntimeError(f"OpenAI API error (question grouping): {exc}") from exc
            parsed = response.choices[0].message.parsed
            if parsed is None:
                return []
            return [k for k in parsed.matching_keys if k in questions and k != anchor]

        results = await asyncio.gather(
            *(matches_for(k) for k in keys), return_exceptions=True
        )

        # Union-find over the pairs. A match is only honoured when **both** questions named each
        # other, so one model slip cannot merge two unrelated questions.
        claimed: dict[str, set[str]] = {}
        for key, result in zip(keys, results):
            claimed[key] = set(result) if isinstance(result, list) else set()

        parent = {k: k for k in keys}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for key in keys:
            for other in claimed[key]:
                if key in claimed.get(other, set()):
                    parent[find(key)] = find(other)

        merged: dict[str, list[str]] = {}
        for key in keys:
            merged.setdefault(find(key), []).append(key)
        return QuestionGrouping(groups=list(merged.values()))


class FakeQuestionGrouper(QuestionGrouper):
    """Deterministic grouper for tests: returns whatever grouping it was constructed with."""

    def __init__(self, grouping: QuestionGrouping | None = None) -> None:
        self._grouping = grouping
        self.calls = 0

    async def group(self, questions: dict[str, str]) -> QuestionGrouping:
        self.calls += 1
        if self._grouping is not None:
            return self._grouping
        return QuestionGrouping(groups=[[key] for key in questions])


def build_question_grouper() -> QuestionGrouper | None:
    if not app_settings.openai_api_key.strip():
        return None
    return OpenAIQuestionGrouper(
        api_key=app_settings.openai_api_key,
        model=live_settings.openai_question_grouping_model,
    )

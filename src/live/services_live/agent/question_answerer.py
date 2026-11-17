"""Answer factual questions asked out loud in the meeting.

Fact-checking has two halves. Verifying a claim someone *asserted* is the half that existed. This
is the other one: someone *asks* something factual, nobody in the room knows, and the host wants an
answer while it is still the live topic.

It runs as its own lane rather than through the coaching gate, for one reason: **latency**. The
gate is restraint-biased and fires every few turns — fine for a suggestion, useless for an answer,
because by the time it lands the room has moved on. This runs on every segment against a tiny
prompt, so an answer comes back in about a second.

The judgment that matters most here is what *not* to answer. A discovery question — "what does your
hiring timeline look like?" — is aimed at the counterparty, and answering it for them would wreck
the meeting. That call is the model's; there are no pattern rules on question shape.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings as app_settings
from src.live.config_live import live_settings

logger = logging.getLogger(__name__)

ANSWERER_SYSTEM_PROMPT = """You watch a live meeting transcript for factual questions the room
cannot answer itself, and you answer them.

## Answer a question only when ALL of these hold
1. It asks about the **world** — history, geography, science, public facts about a company or
   person. Not about anyone in this meeting, their product, their process or their needs.
   A question can be phrased generally and still be about them: "what dialects does a
   transcriber need to support?" is about *their* requirements, and only they can answer it.
1b. **Or it is a self-contained calculation** someone asked out loud — "what's 30% of 5000?",
   "what's 22 times 800?", "what's 650 plus 40k?". Give the result plainly and nothing else:
   "That's 1,500." No workings, no commentary.
   This covers arithmetic **only**, and only when the numbers are in the question itself. If the
   sum depends on their figures, their pipeline or their pricing, it is about them — stay silent.
   And never interpret the result: give the number, never what it means for their deal.
2. **Nobody in the transcript has answered it.** If someone already did, stay silent.
3. It is a real question, not rhetorical ("shall we get started?", "make sense?", "right?").
4. It is not in ``already_answered`` — you have handled it before.

## ``is_general_world_knowledge`` — set this honestly, it is the release valve
Set it **true** only when the question is about the world — or is a self-contained calculation —
and you are actually going to answer it.
Set it **false** whenever the question is about this meeting, these people, their company, their
process or their situation — and then leave ``answerable_question`` null.

Never put a refusal, a hedge, or an explanation of why you cannot answer into ``answer``. There is
no way to show that to the host as anything other than an answer, so it reaches them as a card
asserting your own inability. If you cannot answer, say nothing: ``answerable_question`` null plus
a ``skip_reason``.

## The host's own prep questions — the hardest rule, and the one that matters most
You receive ``prep_questions``: the list the host planned to ask on this call. If the question in
the transcript is one of them, or means the same as one of them, **you must stay silent**, no
matter how answerable it looks.

The host asks those questions to hear *this prospect's* answer. Answering one yourself replaces the
answer they needed with generic knowledge, and it lands on their screen while they are still
waiting for the room to respond. "Which dialects does the transcriber need to support?" has a
textbook answer and is still catastrophic to answer here — the point is which dialects *these
people* need, and only they can say.

A question that appears in ``prep_questions`` is never world knowledge for this purpose, however
general it sounds on its own.

## NEVER answer these
- **Questions aimed at a participant about their own situation.** "What does your hiring timeline
  look like?", "How does your ATS work?", "What's your budget?", "Who signs off on this?" These are
  the host's discovery questions. The person in the meeting is meant to answer them. Answering for
  them destroys the conversation the host is trying to have. This is the most important rule here.
- Questions about the meeting itself — scheduling, agenda, next steps.
- Anything you would have to guess at. Silence is better than a confident wrong answer.

## Live data
If answering well needs **current** information — this year's revenue, headcount, recent funding,
current pricing, recent news — set ``needs_live_data`` true and put a short lookup string in
``search_query``, and leave ``answer`` null. Someone else will fetch it and come back to you.

For **stable** facts — founding dates, geography, history, what a company broadly does — answer
directly from your own knowledge. Do not ask for a lookup you do not need.

## Style
One or two sentences. Plain, factual, immediately useful to someone mid-call. No preamble, no
"great question", no hedging about your knowledge.

## Output
- Nothing to answer → ``answerable_question`` null, and a short ``skip_reason``.
- Something to answer → ``answerable_question`` set to the question in plain words, plus either
  ``answer`` or (``needs_live_data`` + ``search_query``).
"""

COMPOSE_SYSTEM_PROMPT = """You are answering a factual question asked during a live meeting, using
search findings that were just fetched for you.

Give the answer in one or two plain sentences. Use only the findings — do not add your own
recollection, and do not mention searching, sources, or where the information came from. Write it
as a direct answer to the question.
"""


class QuestionAnswer(BaseModel):
    """What the answerer decided about the latest stretch of conversation."""

    model_config = ConfigDict(extra="forbid")

    # Null means there is nothing to answer; ``skip_reason`` says why (logged, never shown).
    answerable_question: str | None = None
    skip_reason: str | None = None
    # Structural gate, not a phrasing rule. The prompt already forbids answering a question aimed
    # at someone in the room, and the model still did it — then wrote its refusal into ``answer``,
    # which shipped to the host as a card saying "this is not answerable by me". Requiring an
    # explicit claim that the question is about the world means a refusal cannot be emitted at
    # all: no claim, no frame.
    # True for a world fact you are about to state, or a self-contained calculation you are about
    # to perform. False for anything that depends on this meeting, these people or their numbers.
    is_general_world_knowledge: bool = False
    needs_live_data: bool = False
    search_query: str | None = None
    answer: str | None = None


class ComposedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)


class QuestionAnswerer(ABC):
    @abstractmethod
    async def detect_and_answer(
        self,
        recent_turns: list[dict[str, str]],
        already_answered: list[str],
        prep_questions: list[str] | None = None,
    ) -> QuestionAnswer:
        """Look at the tail of the conversation and decide whether to answer something.

        ``prep_questions`` is what the host planned to ask. Anything matching one of them is the
        host's own script and must never be answered — see the prompt.
        """

    @abstractmethod
    async def compose_from_findings(self, question: str, findings: str) -> str:
        """Turn search findings into the answer the host reads."""


class OpenAIQuestionAnswerer(QuestionAnswerer):
    def __init__(self, *, api_key: str, model: str, client: AsyncOpenAI | None = None) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is not set")
        self._model = model.strip()
        self._client = client or AsyncOpenAI(api_key=api_key.strip())

    async def detect_and_answer(
        self,
        recent_turns: list[dict[str, str]],
        already_answered: list[str],
        prep_questions: list[str] | None = None,
    ) -> QuestionAnswer:
        payload = {
            "recent_transcript": recent_turns,
            "already_answered": already_answered,
            # The host's script. Anything matching it is theirs to ask, not yours to answer.
            "prep_questions": prep_questions or [],
        }
        try:
            response = await self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": ANSWERER_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format=QuestionAnswer,
                temperature=0.0,
            )
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error (question answerer): {exc}") from exc
        parsed = response.choices[0].message.parsed
        return parsed or QuestionAnswer(skip_reason="empty parse")

    async def compose_from_findings(self, question: str, findings: str) -> str:
        try:
            response = await self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": COMPOSE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": question, "findings": findings}, ensure_ascii=False
                        ),
                    },
                ],
                response_format=ComposedAnswer,
                temperature=0.0,
            )
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error (answer compose): {exc}") from exc
        parsed = response.choices[0].message.parsed
        return parsed.answer if parsed else ""


class FakeQuestionAnswerer(QuestionAnswerer):
    """Deterministic answerer for tests."""

    def __init__(
        self,
        responses: list[QuestionAnswer] | None = None,
        *,
        composed: str = "Composed from findings.",
    ) -> None:
        self._responses = list(responses or [])
        self._idx = 0
        self._composed = composed
        self.calls = 0
        self.compose_calls = 0
        self.seen_already_answered: list[list[str]] = []

    async def detect_and_answer(self, recent_turns, already_answered, prep_questions=None):
        self.calls += 1
        self.seen_already_answered.append(list(already_answered))
        if self._idx < len(self._responses):
            result = self._responses[self._idx]
            self._idx += 1
            return result
        return QuestionAnswer(skip_reason="fake: nothing to answer")

    async def compose_from_findings(self, question: str, findings: str) -> str:
        self.compose_calls += 1
        return self._composed


def build_question_answerer() -> QuestionAnswerer | None:
    if not app_settings.openai_api_key.strip():
        return None
    return OpenAIQuestionAnswerer(
        api_key=app_settings.openai_api_key,
        model=live_settings.openai_answerer_model,
    )

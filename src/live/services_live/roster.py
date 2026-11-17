"""Map transcript speakers to coached users.

A session's transcript contains everyone in the room. Only some of those speakers are *coached
users* (they have prep questions, a chat thread, and receive addressed events); the rest are
meeting participants — the counterparty, colleagues who aren't running the coach, anyone the
transcription pipeline picked up.

Matching is best-effort by design. ``participant_uuid`` is authoritative when the pipeline supplies
it on both sides; otherwise we fall back to a normalized name comparison, which is inherently
fragile (diarization may emit "Priya", "Priya Raghavan", or "Speaker 1" for the same person).

**Nothing about correctness may depend on a match succeeding.** An unmatched speaker is simply
treated as an external participant: they are described to the agent as context and never become
the recipient of an event. Addressing always comes from explicit ``user_id`` fields on the wire,
never from speaker matching.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.live.services_live.projection import MeetingSession, SessionUserState, TranscriptTurn
from src.live.services_live.text_norm import normalize_text


@dataclass(frozen=True)
class SpeakerRole:
    """How one transcript speaker relates to the coached roster."""

    speaker: str
    user_id: str | None = None
    matched_by: str | None = None  # "participant_uuid" | "name" | None

    @property
    def is_coached_user(self) -> bool:
        return self.user_id is not None


def _name_key(name: str | None) -> str:
    return normalize_text(name or "")


def match_speaker(
    users: dict[str, SessionUserState],
    *,
    speaker: str,
    participant_uuid: str | None = None,
) -> SpeakerRole:
    """Resolve one speaker to a coached user, or to an external participant."""
    if participant_uuid:
        for user in users.values():
            if user.participant_uuid and user.participant_uuid == participant_uuid:
                return SpeakerRole(
                    speaker=speaker,
                    user_id=user.user_id,
                    matched_by="participant_uuid",
                )

    key = _name_key(speaker)
    if key:
        for user in users.values():
            if _name_key(user.name) == key:
                return SpeakerRole(speaker=speaker, user_id=user.user_id, matched_by="name")

    return SpeakerRole(speaker=speaker)


def match_turn(session: MeetingSession, turn: TranscriptTurn) -> SpeakerRole:
    return match_speaker(
        session.users,
        speaker=turn.speaker,
        participant_uuid=turn.participant_uuid,
    )


def speaker_roles(session: MeetingSession) -> list[SpeakerRole]:
    """One entry per distinct speaker seen in the transcript, in first-appearance order."""
    seen: dict[str, SpeakerRole] = {}
    for turn in session.transcript:
        key = f"{turn.participant_uuid or ''}|{_name_key(turn.speaker)}"
        if key in seen:
            continue
        seen[key] = match_turn(session, turn)
    return list(seen.values())


def external_participants(session: MeetingSession) -> list[str]:
    """Speaker names heard in the meeting that are not coached users."""
    out: list[str] = []
    for role in speaker_roles(session):
        if not role.is_coached_user and role.speaker not in out:
            out.append(role.speaker)
    return out


def roster_summary(session: MeetingSession) -> dict[str, list[dict[str, str | None]]]:
    """Roster description for agent prompts: who is coached, who is just in the room."""
    heard = {role.user_id for role in speaker_roles(session) if role.user_id}
    return {
        "coached_users": [
            {
                "user_id": user.user_id,
                "name": user.name or None,
                "heard_speaking": "yes" if user.user_id in heard else "no",
            }
            for user in session.users.values()
        ],
        "other_participants": [{"name": name} for name in external_participants(session)],
    }

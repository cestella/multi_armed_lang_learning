"""Domain event models for the language tutor event log."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["it", "es"]

EventType = Literal[
    "session_started",
    "user_submitted",
    "evaluation_completed",
    "reward_computed",
    "engagement_rated",
    "bandit_updated",
    "arm_selected",
    "assistant_responded",
    "tool_failed",
    "session_ended",
]


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Event(BaseModel):
    """A single domain event stored in events.jsonl."""

    event_id: str = Field(default_factory=_new_uuid)
    turn_id: str | None = None
    ts: str = Field(default_factory=_now_iso)
    language: Language
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)


# --- Typed payload helpers for constructing events ---


def session_started(language: Language, **extra: Any) -> Event:
    return Event(language=language, type="session_started", payload=extra)


def user_submitted(language: Language, turn_id: str, text: str) -> Event:
    return Event(
        language=language,
        type="user_submitted",
        turn_id=turn_id,
        payload={"text": text},
    )


def evaluation_completed(language: Language, turn_id: str, result: dict[str, Any]) -> Event:
    return Event(
        language=language,
        type="evaluation_completed",
        turn_id=turn_id,
        payload=result,
    )


def reward_computed(
    language: Language, turn_id: str, reward: float, components: dict[str, Any]
) -> Event:
    return Event(
        language=language,
        type="reward_computed",
        turn_id=turn_id,
        payload={"reward": reward, "components": components},
    )


def bandit_updated(language: Language, turn_id: str, arm_id: str, reward: float) -> Event:
    return Event(
        language=language,
        type="bandit_updated",
        turn_id=turn_id,
        payload={"arm_id": arm_id, "reward": reward},
    )


def arm_selected(
    language: Language, turn_id: str, arm_id: str, score: float, scores: dict[str, float]
) -> Event:
    return Event(
        language=language,
        type="arm_selected",
        turn_id=turn_id,
        payload={"arm_id": arm_id, "score": score, "scores": scores},
    )


def assistant_responded(language: Language, turn_id: str, text: str) -> Event:
    return Event(
        language=language,
        type="assistant_responded",
        turn_id=turn_id,
        payload={"text": text},
    )


def tool_failed(language: Language, turn_id: str | None, tool: str, error: str) -> Event:
    return Event(
        language=language,
        type="tool_failed",
        turn_id=turn_id,
        payload={"tool": tool, "error": error},
    )


def engagement_rated(
    language: Language, turn_id: str, stars: int, blended_reward: float
) -> Event:
    return Event(
        language=language,
        type="engagement_rated",
        turn_id=turn_id,
        payload={"stars": stars, "blended_reward": blended_reward},
    )


def session_ended(language: Language) -> Event:
    return Event(language=language, type="session_ended")

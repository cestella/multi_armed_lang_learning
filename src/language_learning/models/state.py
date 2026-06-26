"""State models for learner profile, bandit, app state, and UI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Learner Profile ---


class RecurringItem(BaseModel):
    label: str
    count: int = 0


class RecentFocus(BaseModel):
    label: str
    since: str  # ISO8601


class LearnerProfile(BaseModel):
    schema_version: int = 1
    language: str = "it"
    created_at: str = ""
    updated_at: str = ""
    preferences: dict[str, Any] = Field(default_factory=dict)
    recurring_wins: list[RecurringItem] = Field(default_factory=list)
    recurring_fixes: list[RecurringItem] = Field(default_factory=list)
    recent_focus: RecentFocus | None = None


# --- Bandit State ---


class ArmStats(BaseModel):
    n: int = 0
    mean: float = 0.0


class BanditState(BaseModel):
    schema_version: int = 1
    algo: str = "ucb1"
    c: float = 0.7
    cooldown_max_repeat: int = 2
    cooldown_penalty: float = 0.15
    total_pulls: int = 0
    arms: dict[str, ArmStats] = Field(default_factory=dict)
    recent_arms: list[str] = Field(default_factory=list)
    processed_turn_ids: list[str] = Field(default_factory=list)


# --- UI State ---


class FeedbackPanel(BaseModel):
    praise: str = ""
    fix_one: str = ""
    micro_rule: str | None = None
    next_nudge: str = ""
    hint_phrase: str | None = None
    retry_prompt: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str
    turn_id: str | None = None


class ConversationFocus(BaseModel):
    focus_skill: str  # e.g. "past_narration"
    source_arm: str  # arm_id that was active when focus began
    turns_remaining: int = 3  # decrement each turn
    expansion_level: int = 1  # 1=basic, 2=expanded, 3=complex
    created_at: str = ""  # ISO timestamp


class DebugState(BaseModel):
    current_arm: str = ""
    last_reward: float | None = None
    reward_components: dict[str, Any] = Field(default_factory=dict)
    arm_scores: dict[str, float] = Field(default_factory=dict)
    focus_skill: str | None = None
    focus_turns: int | None = None


class AppState(BaseModel):
    language: str = "it"
    session_active: bool = False
    chat_messages: list[ChatMessage] = Field(default_factory=list)
    feedback_panel: FeedbackPanel = Field(default_factory=FeedbackPanel)
    status_line: str = ""
    debug_visible: bool = False
    debug_state: DebugState = Field(default_factory=DebugState)
    current_arm: str = ""
    pending_rating_turn_id: str | None = None
    codex_status: str = "disconnected"
    turn_count: int = 0
    conversation_focus: ConversationFocus | None = None

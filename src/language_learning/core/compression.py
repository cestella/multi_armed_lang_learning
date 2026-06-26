"""Conversation compression for Codex restart context."""

from __future__ import annotations

from language_learning.models.skill_state import SkillState
from language_learning.models.state import ChatMessage, LearnerProfile


def compress_conversation(
    messages: list[ChatMessage],
    learner_profile: LearnerProfile,
    skill_state: SkillState,
    max_recent: int = 6,
) -> dict:
    """Compress conversation state for Codex restart prompt.

    Simple extractive summary (no LLM needed):
    - message_count: total messages exchanged
    - learner_struggles: from learner_profile.recurring_fixes
    - learner_wins: from learner_profile.recurring_wins
    - recent_focus: from learner_profile.recent_focus
    - skill_summary: skills with mastery < 0.5
    - recent_messages: last max_recent messages verbatim
    """
    # Count only user and assistant messages (skip system)
    conversation_messages = [
        m for m in messages if m.role in ("user", "assistant")
    ]

    learner_struggles = [
        {"label": item.label, "count": item.count}
        for item in learner_profile.recurring_fixes
    ]

    learner_wins = [
        {"label": item.label, "count": item.count}
        for item in learner_profile.recurring_wins
    ]

    recent_focus = None
    if learner_profile.recent_focus:
        recent_focus = learner_profile.recent_focus.label

    # Skills with mastery < 0.5 — areas that need work
    skill_summary = [
        {"skill": name, "mastery": round(stats.mastery, 2)}
        for name, stats in skill_state.skills.items()
        if stats.mastery < 0.5
    ]

    # Last N messages verbatim
    recent = conversation_messages[-max_recent:]
    recent_messages = [
        {"role": m.role, "text": m.text}
        for m in recent
    ]

    return {
        "message_count": len(conversation_messages),
        "learner_struggles": learner_struggles,
        "learner_wins": learner_wins,
        "recent_focus": recent_focus,
        "skill_summary": skill_summary,
        "recent_messages": recent_messages,
    }

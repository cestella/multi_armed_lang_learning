"""Conversation focus management — locks tutor into a scene for repeated practice."""

from __future__ import annotations

from datetime import datetime, timezone

from language_learning.models.arms import Arm
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.skill_state import SkillState
from language_learning.models.state import ConversationFocus


def maybe_create_focus(
    evaluation: EvaluationResult,
    arm: Arm,
    existing_focus: ConversationFocus | None,
    skill_state: SkillState,
    max_turns: int = 3,
) -> ConversationFocus | None:
    """Create a conversation focus when the learner struggles.

    Returns None if a focus already exists, no weakness is detected, or
    all target skills are already mastered (>= 0.6).
    """
    if existing_focus is not None:
        return None

    # Detect struggle
    struggled = (
        (evaluation.target_attempted and evaluation.target_success in ("no", "partial"))
        or evaluation.avoidance in ("weak", "strong")
    )
    if not struggled:
        return None

    # Pick the highest-weighted target skill with mastery < 0.6
    focus_skill = _pick_weak_skill(arm, skill_state)
    if focus_skill is None:
        return None

    return ConversationFocus(
        focus_skill=focus_skill,
        source_arm=arm.arm_id,
        turns_remaining=max_turns,
        expansion_level=1,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def advance_focus(
    focus: ConversationFocus,
    evaluation: EvaluationResult,
) -> ConversationFocus | None:
    """Advance focus by one turn. Returns None when expired.

    On success: increments expansion_level and decrements turns_remaining
    an extra time (success accelerates exit).
    """
    turns = focus.turns_remaining - 1
    expansion = focus.expansion_level

    if evaluation.target_success == "yes":
        expansion = min(expansion + 1, 3)
        turns -= 1  # success accelerates exit

    if turns <= 0:
        return None

    return ConversationFocus(
        focus_skill=focus.focus_skill,
        source_arm=focus.source_arm,
        turns_remaining=turns,
        expansion_level=expansion,
        created_at=focus.created_at,
    )


def _pick_weak_skill(arm: Arm, skill_state: SkillState) -> str | None:
    """Pick the highest-weighted target skill with mastery < 0.6."""
    candidates: list[tuple[float, str]] = []
    for skill_name, weight in arm.target_skills.items():
        stats = skill_state.skills.get(skill_name)
        mastery = stats.mastery if stats else 0.5  # default mastery for unseen skills
        if mastery < 0.6:
            candidates.append((weight, skill_name))
    if not candidates:
        return None
    # Sort by weight descending, return highest
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

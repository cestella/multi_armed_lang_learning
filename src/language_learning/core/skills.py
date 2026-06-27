"""Skill tracking: update mastery, profile display, and bandit priors."""

from __future__ import annotations

from datetime import datetime, timezone

from language_learning.models.arms import Arm
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.skill_state import SkillState, SkillStats
from language_learning.models.state import LearnerProfile, RecurringItem

ERROR_SKILL_MAP: dict[str, list[str]] = {
    "past_tense": ["past_narration"],
    "tense": ["past_narration"],
    "verb_conjugation": ["past_narration", "conditional_future"],
    "conditional": ["conditional_future"],
    "future": ["conditional_future"],
    "agreement": ["agreement"],
    "gender": ["agreement"],
    "word_choice": ["vocabulary_description"],
    "vocabulary": ["vocabulary_description"],
    "spelling": ["spelling_accuracy"],
    "preposition": ["preposition_usage"],
    "subjunctive": ["subjunctive_mood"],
}

LEARNING_RATE = 0.15


def update_skills(
    skill_state: SkillState,
    evaluation: EvaluationResult,
    arm: Arm,
) -> SkillState:
    """Update skill mastery and confidence after an evaluation.

    Returns a new SkillState (does not mutate input).
    """
    new_state = skill_state.model_copy(deep=True)

    # Collect all skills touched this turn
    touched_skills: set[str] = set()

    # Track which skills had errors
    error_skills: set[str] = set()

    # Decrease mastery for error-related skills
    for error in evaluation.errors:
        mapped = ERROR_SKILL_MAP.get(error.type, [])
        for skill_name in mapped:
            touched_skills.add(skill_name)
            error_skills.add(skill_name)
            stats = new_state.skills.setdefault(skill_name, SkillStats())
            stats.mastery -= LEARNING_RATE * stats.mastery * 0.5
            stats.mastery = max(0.0, stats.mastery)

    # Track target skills when attempted (for scaffold_need updates)
    if evaluation.target_attempted:
        for skill_name in arm.target_skills:
            touched_skills.add(skill_name)
            new_state.skills.setdefault(skill_name, SkillStats())

    # Increase mastery for target skills on success
    if evaluation.target_attempted and evaluation.target_success in ("yes", "partial"):
        success_weight = 1.0 if evaluation.target_success == "yes" else 0.5
        for skill_name, weight in arm.target_skills.items():
            stats = new_state.skills[skill_name]
            stats.mastery += LEARNING_RATE * success_weight * weight * (1.0 - stats.mastery)
            stats.mastery = min(1.0, stats.mastery)

    # Increase confidence for all touched skills
    for skill_name in touched_skills:
        stats = new_state.skills[skill_name]
        stats.confidence += (1.0 - stats.confidence) * 0.1
        stats.confidence = min(1.0, stats.confidence)

    # Update scaffold_need and last_seen for all touched skills
    now = datetime.now(timezone.utc).isoformat()
    for skill_name in touched_skills:
        stats = new_state.skills[skill_name]
        delta = 0.0

        # Avoidance
        if evaluation.avoidance == "strong":
            delta += 0.15
        elif evaluation.avoidance == "weak":
            delta += 0.05

        # Error mapped to this skill
        if skill_name in error_skills:
            delta += 0.10

        # Target attempted outcomes (only for target skills)
        if evaluation.target_attempted and skill_name in arm.target_skills:
            if evaluation.target_success == "yes":
                delta -= 0.15
            elif evaluation.target_success == "partial":
                delta -= 0.05
            else:
                delta += 0.05

        stats.scaffold_need = max(0.0, min(1.0, stats.scaffold_need + delta))
        stats.last_seen = now

    new_state.updated_at = datetime.now(timezone.utc).isoformat()
    return new_state


def update_profile_display(
    profile: LearnerProfile,
    evaluation: EvaluationResult,
    arm: Arm,
) -> LearnerProfile:
    """Update recurring_fixes and recurring_wins for UI display.

    Returns a new LearnerProfile (does not mutate input).
    """
    new_profile = profile.model_copy(deep=True)

    # Update recurring_fixes from errors
    for error in evaluation.errors:
        label = error.type
        found = False
        for item in new_profile.recurring_fixes:
            if item.label == label:
                item.count += 1
                found = True
                break
        if not found:
            new_profile.recurring_fixes.append(RecurringItem(label=label, count=1))

    # Update recurring_wins on target success
    if evaluation.target_success == "yes":
        tag = arm.tags[0] if arm.tags else arm.arm_id
        found = False
        for item in new_profile.recurring_wins:
            if item.label == tag:
                item.count += 1
                found = True
                break
        if not found:
            new_profile.recurring_wins.append(RecurringItem(label=tag, count=1))

    # Set recent_focus if any fix count >= 3
    for item in new_profile.recurring_fixes:
        if item.count >= 3:
            from language_learning.models.state import RecentFocus

            new_profile.recent_focus = RecentFocus(
                label=item.label,
                since=datetime.now(timezone.utc).isoformat(),
            )
            break

    new_profile.updated_at = datetime.now(timezone.utc).isoformat()
    return new_profile


def derive_support_level(
    skill_state: SkillState,
    arm: Arm,
    high_threshold: float = 0.65,
    medium_threshold: float = 0.35,
) -> str:
    """Derive scaffolding support level from skill state and selected arm.

    Returns "low", "medium", or "high".
    """
    if not arm.target_skills or not skill_state.skills:
        return "low"

    total_weighted_need = 0.0
    total_weight = 0.0
    for skill_name, weight in arm.target_skills.items():
        stats = skill_state.skills.get(skill_name, SkillStats())
        total_weighted_need += weight * stats.scaffold_need
        total_weight += weight

        # Force high if any targeted skill has very low mastery AND confidence
        if stats.mastery < 0.3 and stats.confidence < 0.3:
            return "high"

    if total_weight == 0.0:
        return "low"

    weighted_need = total_weighted_need / total_weight

    if weighted_need >= high_threshold:
        return "high"
    if weighted_need >= medium_threshold:
        return "medium"
    return "low"


def compute_skill_prior(
    skill_state: SkillState,
    arms: list[Arm],
    alpha: float = 0.1,
) -> dict[str, float]:
    """Compute additive prior for bandit arm selection based on skill deficits.

    Returns {arm_id: prior_bonus}.
    """
    priors: dict[str, float] = {}
    for arm in arms:
        if not arm.target_skills:
            priors[arm.arm_id] = 0.0
            continue
        total = 0.0
        for skill_name, target_weight in arm.target_skills.items():
            stats = skill_state.skills.get(skill_name, SkillStats())
            need = 1.0 - stats.mastery
            uncertainty = 1.0 - stats.confidence
            total += target_weight * (0.8 * need + 0.2 * uncertainty)
        priors[arm.arm_id] = alpha * total
    return priors

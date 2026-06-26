"""Tests for conversation focus lifecycle."""

from __future__ import annotations

from language_learning.core.focus import advance_focus, maybe_create_focus
from language_learning.models.arms import Arm
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.skill_state import SkillState, SkillStats
from language_learning.models.state import ConversationFocus


def _arm(
    arm_id: str = "narrative_yesterday",
    target_skills: dict[str, float] | None = None,
) -> Arm:
    if target_skills is None:
        target_skills = {"past_narration": 0.7, "verb_agreement": 0.3}
    return Arm(arm_id=arm_id, intent="test", target_skills=target_skills)


def _skill_state(overrides: dict[str, float] | None = None) -> SkillState:
    skills = {}
    defaults = {"past_narration": 0.4, "verb_agreement": 0.5}
    if overrides:
        defaults.update(overrides)
    for name, mastery in defaults.items():
        skills[name] = SkillStats(mastery=mastery)
    return SkillState(language="it", skills=skills)


def _eval(
    target_attempted: bool = True,
    target_success: str = "no",
    avoidance: str = "none",
) -> EvaluationResult:
    return EvaluationResult(
        target_attempted=target_attempted,
        target_success=target_success,
        avoidance=avoidance,
    )


class TestMaybeCreateFocus:
    def test_creates_on_struggle(self):
        """target_attempted=True + target_success='no' → creates focus."""
        focus = maybe_create_focus(
            _eval(target_attempted=True, target_success="no"),
            _arm(),
            None,
            _skill_state(),
        )
        assert focus is not None
        assert focus.focus_skill == "past_narration"
        assert focus.source_arm == "narrative_yesterday"
        assert focus.turns_remaining == 3
        assert focus.expansion_level == 1

    def test_creates_on_partial(self):
        """target_success='partial' → creates focus."""
        focus = maybe_create_focus(
            _eval(target_success="partial"),
            _arm(),
            None,
            _skill_state(),
        )
        assert focus is not None

    def test_creates_on_avoidance(self):
        """avoidance='strong' → creates focus."""
        focus = maybe_create_focus(
            _eval(target_attempted=False, target_success="no", avoidance="strong"),
            _arm(),
            None,
            _skill_state(),
        )
        assert focus is not None

    def test_no_focus_on_full_success(self):
        """target_success='yes' with no avoidance → no focus."""
        focus = maybe_create_focus(
            _eval(target_attempted=True, target_success="yes", avoidance="none"),
            _arm(),
            None,
            _skill_state(),
        )
        assert focus is None

    def test_no_focus_when_one_exists(self):
        """Existing focus → returns None."""
        existing = ConversationFocus(
            focus_skill="past_narration", source_arm="narrative_yesterday"
        )
        focus = maybe_create_focus(
            _eval(target_success="no"),
            _arm(),
            existing,
            _skill_state(),
        )
        assert focus is None

    def test_no_focus_when_all_skills_mastered(self):
        """All target skills mastery >= 0.6 → no focus."""
        focus = maybe_create_focus(
            _eval(target_success="no"),
            _arm(),
            None,
            _skill_state({"past_narration": 0.8, "verb_agreement": 0.7}),
        )
        assert focus is None

    def test_picks_highest_weighted_weak_skill(self):
        """Among multiple target skills, chooses highest weight with mastery < 0.6."""
        arm = _arm(target_skills={"past_narration": 0.3, "verb_agreement": 0.7})
        # verb_agreement has higher weight (0.7) and mastery < 0.6
        focus = maybe_create_focus(
            _eval(target_success="no"),
            arm,
            None,
            _skill_state({"past_narration": 0.4, "verb_agreement": 0.4}),
        )
        assert focus is not None
        assert focus.focus_skill == "verb_agreement"

    def test_skips_mastered_skill_for_lower_weighted(self):
        """Highest-weighted skill is mastered, picks next highest weak skill."""
        arm = _arm(target_skills={"past_narration": 0.8, "verb_agreement": 0.2})
        focus = maybe_create_focus(
            _eval(target_success="no"),
            arm,
            None,
            _skill_state({"past_narration": 0.9, "verb_agreement": 0.3}),
        )
        assert focus is not None
        assert focus.focus_skill == "verb_agreement"


class TestAdvanceFocus:
    def test_decrements_turns(self):
        """Normal turn decrements turns_remaining by 1."""
        focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm="narrative_yesterday",
            turns_remaining=3,
            expansion_level=1,
        )
        result = advance_focus(focus, _eval(target_success="no"))
        assert result is not None
        assert result.turns_remaining == 2
        assert result.expansion_level == 1

    def test_expiration(self):
        """turns_remaining decrements to 0 → returns None."""
        focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm="narrative_yesterday",
            turns_remaining=1,
        )
        result = advance_focus(focus, _eval(target_success="no"))
        assert result is None

    def test_success_accelerates_exit(self):
        """target_success='yes' decrements twice + increments expansion."""
        focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm="narrative_yesterday",
            turns_remaining=3,
            expansion_level=1,
        )
        result = advance_focus(focus, _eval(target_success="yes"))
        assert result is not None
        assert result.turns_remaining == 1  # 3 - 1 - 1
        assert result.expansion_level == 2

    def test_success_can_expire(self):
        """Success with 2 turns remaining → 0 → expired."""
        focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm="narrative_yesterday",
            turns_remaining=2,
            expansion_level=1,
        )
        result = advance_focus(focus, _eval(target_success="yes"))
        assert result is None

    def test_expansion_level_caps_at_3(self):
        """Expansion level does not exceed 3."""
        focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm="narrative_yesterday",
            turns_remaining=5,
            expansion_level=3,
        )
        result = advance_focus(focus, _eval(target_success="yes"))
        assert result is not None
        assert result.expansion_level == 3

    def test_does_not_mutate_input(self):
        """advance_focus returns a new copy, does not modify the input."""
        focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm="narrative_yesterday",
            turns_remaining=3,
            expansion_level=1,
        )
        result = advance_focus(focus, _eval(target_success="no"))
        assert result is not None
        assert focus.turns_remaining == 3  # unchanged
        assert result.turns_remaining == 2

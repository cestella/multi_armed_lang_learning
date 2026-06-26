"""Tests for skill tracking, profile display updates, and bandit priors."""

from language_learning.core.skills import (
    ERROR_SKILL_MAP,
    compute_skill_prior,
    derive_support_level,
    update_profile_display,
    update_skills,
)
from language_learning.models.arms import Arm
from language_learning.models.evaluation import ErrorItem, EvaluationResult
from language_learning.models.skill_state import SkillState, SkillStats
from language_learning.models.state import LearnerProfile, RecurringItem


def _arm(**overrides) -> Arm:
    defaults = {
        "arm_id": "narrative_yesterday",
        "intent": "past tense narration",
        "tags": ["narrative", "past"],
        "target_skills": {"past_narration": 0.8, "vocabulary_description": 0.2},
    }
    defaults.update(overrides)
    return Arm(**defaults)


def _evaluation(**overrides) -> EvaluationResult:
    defaults = {
        "praise": "Good job",
        "fix_one": "Watch the verb ending",
        "target_attempted": True,
        "target_success": "yes",
        "errors": [],
        "fluency_proxy": 0.7,
        "novelty_proxy": 0.6,
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


class TestUpdateSkills:
    def test_success_increases_mastery(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.0)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="yes")

        new_state = update_skills(state, ev, arm)

        assert new_state.skills["past_narration"].mastery > 0.5
        assert new_state.skills["past_narration"].confidence > 0.0

    def test_partial_success_increases_less(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.0)}
        )
        arm = _arm()
        ev_full = _evaluation(target_success="yes")
        ev_partial = _evaluation(target_success="partial")

        new_full = update_skills(state, ev_full, arm)
        new_partial = update_skills(state, ev_partial, arm)

        assert new_full.skills["past_narration"].mastery > new_partial.skills["past_narration"].mastery

    def test_errors_decrease_mastery(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.7, confidence=0.3)}
        )
        arm = _arm()
        ev = _evaluation(
            target_attempted=False,
            target_success="no",
            errors=[ErrorItem(type="past_tense", note="wrong tense")],
        )

        new_state = update_skills(state, ev, arm)

        assert new_state.skills["past_narration"].mastery < 0.7
        assert new_state.skills["past_narration"].confidence > 0.3

    def test_unknown_error_type_ignored(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.2)}
        )
        arm = _arm()
        ev = _evaluation(
            target_attempted=False,
            target_success="no",
            errors=[ErrorItem(type="unknown_error", note="something")],
        )

        new_state = update_skills(state, ev, arm)

        # Mastery unchanged since error type not in map
        assert new_state.skills["past_narration"].mastery == 0.5

    def test_new_skill_created_on_error(self):
        state = SkillState(skills={})
        arm = _arm()
        ev = _evaluation(
            target_attempted=False,
            target_success="no",
            errors=[ErrorItem(type="spelling", note="typo")],
        )

        new_state = update_skills(state, ev, arm)

        assert "spelling_accuracy" in new_state.skills
        assert new_state.skills["spelling_accuracy"].mastery < 0.5  # decreased from default

    def test_new_skill_created_on_success(self):
        state = SkillState(skills={})
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="yes")

        new_state = update_skills(state, ev, arm)

        assert "past_narration" in new_state.skills
        assert new_state.skills["past_narration"].mastery > 0.5  # increased from default

    def test_mastery_clamped_to_bounds(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.99, confidence=0.9)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="yes")

        new_state = update_skills(state, ev, arm)

        assert 0.0 <= new_state.skills["past_narration"].mastery <= 1.0

    def test_does_not_mutate_input(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.2)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="yes")

        update_skills(state, ev, arm)

        assert state.skills["past_narration"].mastery == 0.5
        assert state.skills["past_narration"].confidence == 0.2

    def test_updated_at_set(self):
        state = SkillState(skills={}, updated_at="")
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="yes")

        new_state = update_skills(state, ev, arm)

        assert new_state.updated_at != ""


class TestUpdateProfileDisplay:
    def test_error_adds_to_recurring_fixes(self):
        profile = LearnerProfile(language="it")
        arm = _arm()
        ev = _evaluation(
            errors=[ErrorItem(type="spelling", note="typo")],
            target_success="no",
        )

        new_profile = update_profile_display(profile, ev, arm)

        assert len(new_profile.recurring_fixes) == 1
        assert new_profile.recurring_fixes[0].label == "spelling"
        assert new_profile.recurring_fixes[0].count == 1

    def test_repeated_error_increments_count(self):
        profile = LearnerProfile(
            language="it",
            recurring_fixes=[RecurringItem(label="spelling", count=2)],
        )
        arm = _arm()
        ev = _evaluation(
            errors=[ErrorItem(type="spelling", note="another typo")],
            target_success="no",
        )

        new_profile = update_profile_display(profile, ev, arm)

        assert new_profile.recurring_fixes[0].count == 3

    def test_success_adds_to_recurring_wins(self):
        profile = LearnerProfile(language="it")
        arm = _arm()
        ev = _evaluation(target_success="yes")

        new_profile = update_profile_display(profile, ev, arm)

        assert len(new_profile.recurring_wins) == 1
        assert new_profile.recurring_wins[0].label == "narrative"  # first tag

    def test_partial_success_does_not_add_win(self):
        profile = LearnerProfile(language="it")
        arm = _arm()
        ev = _evaluation(target_success="partial")

        new_profile = update_profile_display(profile, ev, arm)

        assert len(new_profile.recurring_wins) == 0

    def test_recent_focus_set_on_count_3(self):
        profile = LearnerProfile(
            language="it",
            recurring_fixes=[RecurringItem(label="spelling", count=2)],
        )
        arm = _arm()
        ev = _evaluation(
            errors=[ErrorItem(type="spelling", note="typo")],
            target_success="no",
        )

        new_profile = update_profile_display(profile, ev, arm)

        assert new_profile.recent_focus is not None
        assert new_profile.recent_focus.label == "spelling"

    def test_does_not_mutate_input(self):
        profile = LearnerProfile(language="it")
        arm = _arm()
        ev = _evaluation(
            errors=[ErrorItem(type="spelling", note="typo")],
            target_success="yes",
        )

        update_profile_display(profile, ev, arm)

        assert len(profile.recurring_fixes) == 0
        assert len(profile.recurring_wins) == 0


class TestComputeSkillPrior:
    def test_low_mastery_gives_higher_prior(self):
        state = SkillState(
            skills={
                "past_narration": SkillStats(mastery=0.2, confidence=0.5),
                "vocabulary_description": SkillStats(mastery=0.8, confidence=0.5),
            }
        )
        arms = [
            _arm(arm_id="narrative_yesterday", target_skills={"past_narration": 1.0}),
            _arm(arm_id="description_scene", target_skills={"vocabulary_description": 1.0}),
        ]

        priors = compute_skill_prior(state, arms)

        assert priors["narrative_yesterday"] > priors["description_scene"]

    def test_unknown_skill_uses_defaults(self):
        state = SkillState(skills={})
        arms = [_arm(arm_id="test_arm", target_skills={"new_skill": 1.0})]

        priors = compute_skill_prior(state, arms)

        # Default mastery=0.5, confidence=0.0 → need=0.5, uncertainty=1.0
        expected = 0.1 * 1.0 * (0.8 * 0.5 + 0.2 * 1.0)
        assert abs(priors["test_arm"] - expected) < 1e-10

    def test_arm_without_target_skills_gets_zero(self):
        state = SkillState(skills={})
        arms = [_arm(arm_id="no_skills", target_skills={})]

        priors = compute_skill_prior(state, arms)

        assert priors["no_skills"] == 0.0

    def test_high_mastery_high_confidence_low_prior(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.95, confidence=0.95)}
        )
        arms = [_arm(arm_id="test", target_skills={"past_narration": 1.0})]

        priors = compute_skill_prior(state, arms)

        assert priors["test"] < 0.01  # very small prior

    def test_alpha_scales_prior(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.3, confidence=0.5)}
        )
        arms = [_arm(arm_id="test", target_skills={"past_narration": 1.0})]

        p1 = compute_skill_prior(state, arms, alpha=0.1)
        p2 = compute_skill_prior(state, arms, alpha=0.2)

        assert abs(p2["test"] - 2 * p1["test"]) < 1e-10


class TestScaffoldNeed:
    def test_error_increases_scaffold_need(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3, scaffold_need=0.3)}
        )
        arm = _arm()
        ev = _evaluation(
            target_attempted=False,
            target_success="no",
            errors=[ErrorItem(type="past_tense", note="wrong tense")],
        )

        new_state = update_skills(state, ev, arm)

        # Error adds +0.10
        assert new_state.skills["past_narration"].scaffold_need > 0.3

    def test_strong_avoidance_increases_scaffold_need(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3, scaffold_need=0.3)}
        )
        arm = _arm()
        ev = _evaluation(
            target_attempted=True,
            target_success="yes",
            avoidance="strong",
        )

        new_state = update_skills(state, ev, arm)

        # Strong avoidance +0.15, but full success -0.15 → net 0 for target skills
        # But past_narration is a target skill, so both apply
        assert new_state.skills["past_narration"].scaffold_need == 0.3

    def test_full_success_decreases_scaffold_need(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3, scaffold_need=0.5)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="yes", avoidance="none")

        new_state = update_skills(state, ev, arm)

        # Full success -0.15
        assert new_state.skills["past_narration"].scaffold_need == 0.35

    def test_partial_success_decreases_less(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3, scaffold_need=0.5)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="partial", avoidance="none")

        new_state = update_skills(state, ev, arm)

        # Partial success -0.05
        assert new_state.skills["past_narration"].scaffold_need == 0.45

    def test_no_success_increases_scaffold_need(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3, scaffold_need=0.3)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="no", avoidance="none")

        new_state = update_skills(state, ev, arm)

        # No success +0.05
        assert new_state.skills["past_narration"].scaffold_need == 0.35

    def test_scaffold_need_clamped_to_bounds(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3, scaffold_need=0.95)}
        )
        arm = _arm()
        ev = _evaluation(
            target_attempted=True,
            target_success="no",
            avoidance="strong",
            errors=[ErrorItem(type="past_tense", note="wrong")],
        )

        new_state = update_skills(state, ev, arm)

        assert new_state.skills["past_narration"].scaffold_need <= 1.0

    def test_scaffold_need_clamped_at_zero(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3, scaffold_need=0.05)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="yes", avoidance="none")

        new_state = update_skills(state, ev, arm)

        assert new_state.skills["past_narration"].scaffold_need >= 0.0

    def test_multiple_conditions_additive(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3, scaffold_need=0.3)}
        )
        arm = _arm()
        # Error (+0.10) + weak avoidance (+0.05) + target no success (+0.05) = +0.20
        ev = _evaluation(
            target_attempted=True,
            target_success="no",
            avoidance="weak",
            errors=[ErrorItem(type="past_tense", note="wrong")],
        )

        new_state = update_skills(state, ev, arm)

        assert abs(new_state.skills["past_narration"].scaffold_need - 0.5) < 1e-10

    def test_last_seen_updated(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.3)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=True, target_success="yes")

        new_state = update_skills(state, ev, arm)

        assert new_state.skills["past_narration"].last_seen is not None

    def test_last_seen_not_set_for_untouched_skills(self):
        state = SkillState(
            skills={"spelling_accuracy": SkillStats(mastery=0.5, confidence=0.3)}
        )
        arm = _arm()
        ev = _evaluation(target_attempted=False, target_success="no")

        new_state = update_skills(state, ev, arm)

        assert new_state.skills["spelling_accuracy"].last_seen is None


class TestDeriveSupportLevel:
    def test_low_scaffold_need_returns_low(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.7, confidence=0.5, scaffold_need=0.2)}
        )
        arm = _arm(target_skills={"past_narration": 1.0})

        assert derive_support_level(state, arm) == "low"

    def test_medium_scaffold_need_returns_medium(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.5, scaffold_need=0.5)}
        )
        arm = _arm(target_skills={"past_narration": 1.0})

        assert derive_support_level(state, arm) == "medium"

    def test_high_scaffold_need_returns_high(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.5, scaffold_need=0.8)}
        )
        arm = _arm(target_skills={"past_narration": 1.0})

        assert derive_support_level(state, arm) == "high"

    def test_low_mastery_low_confidence_forces_high(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.2, confidence=0.2, scaffold_need=0.1)}
        )
        arm = _arm(target_skills={"past_narration": 1.0})

        # Even with low scaffold_need, very low mastery+confidence forces high
        assert derive_support_level(state, arm) == "high"

    def test_no_target_skills_returns_low(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.2, confidence=0.2, scaffold_need=0.9)}
        )
        arm = _arm(target_skills={})

        assert derive_support_level(state, arm) == "low"

    def test_empty_skill_state_returns_low(self):
        state = SkillState(skills={})
        arm = _arm(target_skills={"past_narration": 1.0})

        assert derive_support_level(state, arm) == "low"

    def test_weighted_average_across_skills(self):
        state = SkillState(
            skills={
                "past_narration": SkillStats(mastery=0.5, confidence=0.5, scaffold_need=0.8),
                "vocabulary_description": SkillStats(mastery=0.5, confidence=0.5, scaffold_need=0.2),
            }
        )
        # weight 0.8 for past_narration, 0.2 for vocabulary
        arm = _arm(target_skills={"past_narration": 0.8, "vocabulary_description": 0.2})

        # weighted = (0.8*0.8 + 0.2*0.2) / (0.8+0.2) = (0.64+0.04)/1.0 = 0.68
        assert derive_support_level(state, arm) == "high"

    def test_unknown_skill_uses_default_scaffold_need(self):
        state = SkillState(skills={})
        arm = _arm(target_skills={"unknown_skill": 1.0})

        # Default scaffold_need=0.3, which is < 0.35 → low
        # But empty skills dict → returns "low" early
        assert derive_support_level(state, arm) == "low"

    def test_custom_thresholds(self):
        state = SkillState(
            skills={"past_narration": SkillStats(mastery=0.5, confidence=0.5, scaffold_need=0.4)}
        )
        arm = _arm(target_skills={"past_narration": 1.0})

        # With default thresholds (0.35, 0.65): 0.4 → medium
        assert derive_support_level(state, arm) == "medium"

        # With higher medium threshold: 0.4 < 0.5 → low
        assert derive_support_level(state, arm, medium_threshold=0.5) == "low"

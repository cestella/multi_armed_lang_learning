"""Tests for the CEFR estimation layer."""

from __future__ import annotations

import copy

import pytest

from language_learning.core.cefr import (
    CEFR_DOMAINS,
    _estimate_domain_score,
    _score_to_label,
    format_cefr_summary,
    update_cefr_state,
)
from language_learning.models.arms import Arm
from language_learning.models.cefr_state import CefrState, DomainEvidence
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.skill_state import SkillState, SkillStats


def _make_arm(**kwargs) -> Arm:
    defaults = {
        "arm_id": "narrative_yesterday",
        "intent": "Past tense narration",
        "cefr_domains": ["narration", "conversation"],
        "target_skills": {"past_narration": 0.8, "vocabulary_description": 0.2},
    }
    defaults.update(kwargs)
    return Arm(**defaults)


def _make_eval(**kwargs) -> EvaluationResult:
    defaults = {
        "target_attempted": True,
        "target_success": "partial",
        "avoidance": "none",
        "fluency_proxy": 0.7,
        "novelty_proxy": 0.6,
    }
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


def _make_skill_state(**kwargs) -> SkillState:
    return SkillState(
        language="it",
        skills={
            "vocabulary_description": SkillStats(mastery=0.6),
            "agreement": SkillStats(mastery=0.5),
            "past_narration": SkillStats(mastery=0.4),
            "conditional_future": SkillStats(mastery=0.3),
        },
        **kwargs,
    )


class TestFirstTurn:
    def test_first_turn_creates_evidence(self):
        """Single update populates domain evidence with decayed values."""
        state = CefrState(language="it")
        arm = _make_arm()
        evaluation = _make_eval()
        skill_state = _make_skill_state()

        new_state = update_cefr_state(state, evaluation, arm, skill_state)

        # narration and conversation domains should have evidence
        assert "narration" in new_state.evidence
        assert "conversation" in new_state.evidence
        assert new_state.evidence["narration"].turn_count == 1
        assert new_state.evidence["conversation"].turn_count == 1

        # Decayed values should be non-zero
        ev = new_state.evidence["narration"]
        assert ev.attempt_rate > 0
        assert ev.success_rate > 0

    def test_comprehension_on_engagement(self):
        """Comprehension updated when target_attempted=True and avoidance='none'."""
        state = CefrState(language="it")
        arm = _make_arm()
        evaluation = _make_eval(target_attempted=True, avoidance="none")
        skill_state = _make_skill_state()

        new_state = update_cefr_state(state, evaluation, arm, skill_state)
        assert "comprehension" in new_state.evidence
        assert new_state.evidence["comprehension"].turn_count == 1

    def test_comprehension_not_on_avoidance(self):
        """Comprehension NOT updated when avoidance is present."""
        state = CefrState(language="it")
        arm = _make_arm()
        evaluation = _make_eval(target_attempted=True, avoidance="weak")
        skill_state = _make_skill_state()

        new_state = update_cefr_state(state, evaluation, arm, skill_state)
        assert "comprehension" not in new_state.evidence

    def test_comprehension_not_when_not_attempted(self):
        """Comprehension NOT updated when target not attempted."""
        state = CefrState(language="it")
        arm = _make_arm()
        evaluation = _make_eval(target_attempted=False, avoidance="none")
        skill_state = _make_skill_state()

        new_state = update_cefr_state(state, evaluation, arm, skill_state)
        assert "comprehension" not in new_state.evidence


class TestDecay:
    def test_decay_reflects_recent_performance(self):
        """After many high-success turns then a failure, rates drop."""
        state = CefrState(language="it")
        arm = _make_arm()
        skill_state = _make_skill_state()

        # 10 successful turns
        for _ in range(10):
            state = update_cefr_state(
                state,
                _make_eval(target_attempted=True, target_success="yes"),
                arm,
                skill_state,
            )

        high_success = state.evidence["narration"].success_rate

        # Now a failure
        state = update_cefr_state(
            state,
            _make_eval(target_attempted=True, target_success="no"),
            arm,
            skill_state,
        )

        assert state.evidence["narration"].success_rate < high_success


class TestComprehensionConfidence:
    def test_comprehension_confidence_capped(self):
        """Comprehension confidence never exceeds 0.5."""
        state = CefrState(language="it")
        arm = _make_arm()
        skill_state = _make_skill_state()

        # Many successful turns
        for _ in range(30):
            state = update_cefr_state(
                state,
                _make_eval(target_attempted=True, target_success="yes", avoidance="none"),
                arm,
                skill_state,
            )

        assert state.confidence.get("comprehension", 0.0) <= 0.5


class TestAvoidancePenalty:
    def test_avoidance_penalizes_score_and_confidence(self):
        """High avoidance lowers both score and confidence."""
        state = CefrState(language="it")
        arm = _make_arm()
        skill_state = _make_skill_state()

        # Turns with no avoidance
        state_clean = CefrState(language="it")
        for _ in range(10):
            state_clean = update_cefr_state(
                state_clean,
                _make_eval(avoidance="none"),
                arm,
                skill_state,
            )

        # Turns with strong avoidance
        state_avoid = CefrState(language="it")
        for _ in range(10):
            state_avoid = update_cefr_state(
                state_avoid,
                _make_eval(avoidance="strong"),
                arm,
                skill_state,
            )

        ev_clean = state_clean.evidence["narration"]
        ev_avoid = state_avoid.evidence["narration"]

        score_clean, conf_clean = _estimate_domain_score("narration", ev_clean, skill_state)
        score_avoid, conf_avoid = _estimate_domain_score("narration", ev_avoid, skill_state)

        assert score_avoid < score_clean
        assert conf_avoid < conf_clean


class TestImmutability:
    def test_does_not_mutate_input(self):
        """update_cefr_state returns new copy, original unchanged."""
        state = CefrState(language="it")
        original = state.model_copy(deep=True)
        arm = _make_arm()
        skill_state = _make_skill_state()

        new_state = update_cefr_state(state, _make_eval(), arm, skill_state)

        assert state.evidence == original.evidence
        assert state.domains == original.domains
        assert new_state is not state
        assert new_state.evidence != original.evidence


class TestTimestamp:
    def test_last_updated_set(self):
        """Timestamp populated after update."""
        state = CefrState(language="it")
        arm = _make_arm()
        skill_state = _make_skill_state()

        new_state = update_cefr_state(state, _make_eval(), arm, skill_state)
        assert new_state.last_updated != ""


class TestScoring:
    def test_high_success_higher_score(self):
        """Sustained success produces score > 60."""
        state = CefrState(language="it")
        arm = _make_arm()
        skill_state = _make_skill_state()

        for _ in range(20):
            state = update_cefr_state(
                state,
                _make_eval(
                    target_attempted=True,
                    target_success="yes",
                    avoidance="none",
                    fluency_proxy=0.8,
                    novelty_proxy=0.7,
                ),
                arm,
                skill_state,
            )

        ev = state.evidence["narration"]
        score, _ = _estimate_domain_score("narration", ev, skill_state)
        assert score > 60

    def test_skill_mastery_bonus_is_modest(self):
        """Skill bonus contributes at most ~5 points."""
        ev = DomainEvidence(
            attempt_rate=0.5,
            success_rate=0.5,
            avoidance_rate=0.0,
            fluency_avg=0.5,
            novelty_avg=0.5,
            turn_count=10,
        )

        # High mastery
        high_skills = SkillState(
            language="it",
            skills={
                "past_narration": SkillStats(mastery=1.0),
                "vocabulary_description": SkillStats(mastery=1.0),
            },
        )
        score_high, _ = _estimate_domain_score("narration", ev, high_skills)

        # Low mastery
        low_skills = SkillState(
            language="it",
            skills={
                "past_narration": SkillStats(mastery=0.0),
                "vocabulary_description": SkillStats(mastery=0.0),
            },
        )
        score_low, _ = _estimate_domain_score("narration", ev, low_skills)

        bonus_range = score_high - score_low
        assert bonus_range <= 10.1  # ±5 max, so range ≤ ~10


class TestZeroTurns:
    def test_zero_turns_a1_zero_confidence(self):
        """No evidence: label A1, confidence 0.0."""
        state = CefrState(language="it")
        assert state.overall_estimate == "A1"
        assert state.confidence.get("overall", 0.0) == 0.0


class TestLabels:
    def test_low_confidence_coarse_labels(self):
        """Confidence < 0.5 uses 5 coarse labels."""
        coarse_labels = {"A1", "A2", "A2+", "B1", "B1+"}
        for score in range(0, 101, 5):
            label = _score_to_label(score, 0.3)
            assert label in coarse_labels, f"score={score} got {label}"

    def test_high_confidence_fine_labels(self):
        """Confidence >= 0.5 uses full 10 labels."""
        fine_labels = {
            "A1", "A1+", "A2", "A2+", "A2+/B1-", "B1-", "B1", "B1+", "B1+/B2-", "B2",
        }
        labels_seen = set()
        for score in range(0, 101):
            label = _score_to_label(score, 0.7)
            assert label in fine_labels, f"score={score} got {label}"
            labels_seen.add(label)
        # Should see all 10 labels
        assert labels_seen == fine_labels

    def test_boundary_zones_produce_overlap_labels(self):
        """Score ~60 with high confidence produces A2+/B1-."""
        label = _score_to_label(60.0, 0.7)
        assert label == "A2+/B1-"


class TestFormat:
    def test_format_includes_all_domains_with_evidence(self):
        """Format output includes label, conf, turns, attempts."""
        state = CefrState(language="it")
        arm = _make_arm()
        skill_state = _make_skill_state()

        for _ in range(5):
            state = update_cefr_state(state, _make_eval(), arm, skill_state)

        summary = format_cefr_summary(state)

        assert "CEFR Estimate (IT)" in summary
        assert "Overall:" in summary
        assert "narration" in summary
        assert "conversation" in summary
        assert "comprehension" in summary
        assert "conf:" in summary
        assert "turns" in summary


class TestOverallWeighting:
    def test_overall_is_confidence_weighted(self):
        """Low-confidence domains contribute less to overall."""
        state = CefrState(language="it")
        skill_state = _make_skill_state()

        # Build up conversation with many turns (high confidence)
        arm_conv = _make_arm(cefr_domains=["conversation"])
        for _ in range(20):
            state = update_cefr_state(
                state,
                _make_eval(target_attempted=True, target_success="yes"),
                arm_conv,
                skill_state,
            )

        # Build up opinion with few turns (low confidence)
        arm_op = _make_arm(cefr_domains=["opinion"])
        for _ in range(2):
            state = update_cefr_state(
                state,
                _make_eval(target_attempted=True, target_success="no"),
                arm_op,
                skill_state,
            )

        # Conversation has more weight due to higher confidence
        conv_conf = state.confidence.get("conversation", 0.0)
        op_conf = state.confidence.get("opinion", 0.0)
        assert conv_conf > op_conf


class TestArmNoCefrDomains:
    def test_arm_with_no_cefr_domains(self):
        """Arm without cefr_domains only updates comprehension if criteria met."""
        state = CefrState(language="it")
        arm = _make_arm(cefr_domains=[])
        skill_state = _make_skill_state()

        # With engagement criteria met → comprehension only
        new_state = update_cefr_state(
            state,
            _make_eval(target_attempted=True, avoidance="none"),
            arm,
            skill_state,
        )
        assert "comprehension" in new_state.evidence
        # No other domains
        for domain in ["conversation", "description", "narration", "opinion"]:
            assert domain not in new_state.evidence

        # Without engagement criteria → nothing
        state2 = CefrState(language="it")
        new_state2 = update_cefr_state(
            state2,
            _make_eval(target_attempted=False, avoidance="none"),
            arm,
            skill_state,
        )
        assert len(new_state2.evidence) == 0

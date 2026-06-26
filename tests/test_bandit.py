"""Tests for UCB1 bandit logic."""

import math

from language_learning.core.bandit import ucb1_select, ucb1_update
from language_learning.models.state import ArmStats, BanditState


def _make_state(**overrides) -> BanditState:
    defaults = {
        "arms": {
            "arm_a": ArmStats(n=0, mean=0.0),
            "arm_b": ArmStats(n=0, mean=0.0),
            "arm_c": ArmStats(n=0, mean=0.0),
        },
        "total_pulls": 0,
        "recent_arms": [],
        "processed_turn_ids": [],
    }
    defaults.update(overrides)
    return BanditState(**defaults)


class TestWarmStart:
    def test_unexplored_arms_selected_lexically(self):
        state = _make_state()
        arm_id, score, scores = ucb1_select(state)
        assert arm_id == "arm_a"  # first lexically among all-zero arms
        assert score == 0.0

    def test_partially_explored_selects_unexplored(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=5, mean=0.6),
                "arm_b": ArmStats(n=0, mean=0.0),
                "arm_c": ArmStats(n=3, mean=0.4),
            },
            total_pulls=8,
        )
        arm_id, _, _ = ucb1_select(state)
        assert arm_id == "arm_b"


class TestUCB1Scores:
    def test_higher_mean_wins(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=10, mean=0.8),
                "arm_b": ArmStats(n=10, mean=0.3),
            },
            total_pulls=20,
        )
        arm_id, _, scores = ucb1_select(state)
        assert arm_id == "arm_a"
        assert scores["arm_a"] > scores["arm_b"]

    def test_exploration_bonus_for_underexplored(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=100, mean=0.5),
                "arm_b": ArmStats(n=1, mean=0.4),
            },
            total_pulls=101,
        )
        _, _, scores = ucb1_select(state)
        # arm_b should have a big exploration bonus
        expected_b = 0.4 + 0.7 * math.sqrt(math.log(101) / 1)
        assert abs(scores["arm_b"] - expected_b) < 1e-10


class TestCooldown:
    def test_cooldown_penalty_reduces_score(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=10, mean=0.6),
                "arm_b": ArmStats(n=10, mean=0.6),
            },
            total_pulls=20,
            recent_arms=["arm_a", "arm_a"],
        )
        _, _, scores = ucb1_select(state)
        # arm_a should be penalized by 0.15 * 2 = 0.30
        base = 0.6 + 0.7 * math.sqrt(math.log(20) / 10)
        assert abs(scores["arm_a"] - (base - 0.30)) < 1e-10
        assert abs(scores["arm_b"] - base) < 1e-10


class TestRepeatLimit:
    def test_force_different_arm_on_repeat(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=10, mean=0.9),
                "arm_b": ArmStats(n=10, mean=0.1),
            },
            total_pulls=20,
            recent_arms=["arm_a", "arm_a"],  # cooldown_max_repeat=2, all same
        )
        arm_id, _, _ = ucb1_select(state)
        assert arm_id == "arm_b"  # forced away from arm_a


class TestUpdate:
    def test_basic_update(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=0, mean=0.0),
                "arm_b": ArmStats(n=0, mean=0.0),
            },
        )
        new_state = ucb1_update(state, "arm_a", 0.8, "turn-1")
        assert new_state.total_pulls == 1
        assert new_state.arms["arm_a"].n == 1
        assert new_state.arms["arm_a"].mean == 0.8
        assert new_state.recent_arms == ["arm_a"]
        assert "turn-1" in new_state.processed_turn_ids

    def test_incremental_mean(self):
        state = _make_state(
            arms={"arm_a": ArmStats(n=1, mean=0.8)},
            total_pulls=1,
        )
        new_state = ucb1_update(state, "arm_a", 0.4, "turn-2")
        assert new_state.arms["arm_a"].n == 2
        assert abs(new_state.arms["arm_a"].mean - 0.6) < 1e-10

    def test_recent_arms_trimmed(self):
        state = _make_state(
            arms={"arm_a": ArmStats(n=2, mean=0.5), "arm_b": ArmStats(n=1, mean=0.3)},
            total_pulls=3,
            recent_arms=["arm_a", "arm_b"],
        )
        new_state = ucb1_update(state, "arm_a", 0.6, "turn-3")
        assert len(new_state.recent_arms) == 2  # trimmed to cooldown_max_repeat
        assert new_state.recent_arms == ["arm_b", "arm_a"]


class TestIdempotency:
    def test_duplicate_turn_id_no_op(self):
        state = _make_state(
            arms={"arm_a": ArmStats(n=1, mean=0.5)},
            total_pulls=1,
            processed_turn_ids=["turn-1"],
        )
        new_state = ucb1_update(state, "arm_a", 0.9, "turn-1")
        assert new_state.arms["arm_a"].n == 1
        assert new_state.arms["arm_a"].mean == 0.5
        assert new_state.total_pulls == 1


class TestSkillPrior:
    def test_skill_prior_biases_selection(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=10, mean=0.5),
                "arm_b": ArmStats(n=10, mean=0.5),
            },
            total_pulls=20,
        )
        # arm_b gets a large prior, should be selected
        prior = {"arm_a": 0.0, "arm_b": 0.5}
        arm_id, _, scores = ucb1_select(state, skill_prior=prior)
        assert arm_id == "arm_b"
        assert scores["arm_b"] > scores["arm_a"]

    def test_no_prior_same_as_none(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=10, mean=0.5),
                "arm_b": ArmStats(n=10, mean=0.5),
            },
            total_pulls=20,
        )
        _, _, scores_none = ucb1_select(state, skill_prior=None)
        _, _, scores_empty = ucb1_select(state, skill_prior={})
        assert scores_none == scores_empty

    def test_prior_additive_to_ucb(self):
        state = _make_state(
            arms={
                "arm_a": ArmStats(n=10, mean=0.5),
            },
            total_pulls=10,
        )
        _, _, scores_no_prior = ucb1_select(state)
        _, _, scores_with_prior = ucb1_select(state, skill_prior={"arm_a": 0.1})
        assert abs(scores_with_prior["arm_a"] - scores_no_prior["arm_a"] - 0.1) < 1e-10


class TestTieBreaking:
    def test_lexical_tie_breaking(self):
        state = _make_state(
            arms={
                "arm_b": ArmStats(n=10, mean=0.5),
                "arm_a": ArmStats(n=10, mean=0.5),
            },
            total_pulls=20,
        )
        arm_id, _, _ = ucb1_select(state)
        assert arm_id == "arm_a"  # lexically first wins ties

    def test_does_not_mutate_input(self):
        state = _make_state(
            arms={"arm_a": ArmStats(n=1, mean=0.5)},
            total_pulls=1,
        )
        ucb1_update(state, "arm_a", 0.9, "turn-x")
        assert state.arms["arm_a"].mean == 0.5
        assert state.total_pulls == 1

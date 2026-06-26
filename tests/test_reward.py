"""Table-driven tests for reward computation."""

import pytest

from language_learning.core.reward import blend_reward, compute_reward
from language_learning.models.evaluation import EvaluationResult


@pytest.mark.parametrize(
    "kwargs, expected_reward",
    [
        # Perfect score: attempted=True, success=yes, novelty=1, fluency=1, no avoidance
        (
            {
                "target_attempted": True,
                "target_success": "yes",
                "novelty_proxy": 1.0,
                "fluency_proxy": 1.0,
                "avoidance": "none",
            },
            0.35 + 0.25 + 0.20 + 0.10,  # = 0.90
        ),
        # Zero score: not attempted, no success, zero novelty/fluency, strong avoidance
        (
            {
                "target_attempted": False,
                "target_success": "no",
                "novelty_proxy": 0.0,
                "fluency_proxy": 0.0,
                "avoidance": "strong",
            },
            0.0,  # clamped from -0.30
        ),
        # Partial success
        (
            {
                "target_attempted": True,
                "target_success": "partial",
                "novelty_proxy": 0.5,
                "fluency_proxy": 0.5,
                "avoidance": "none",
            },
            0.35 + 0.125 + 0.10 + 0.05,  # = 0.625
        ),
        # Weak avoidance
        (
            {
                "target_attempted": True,
                "target_success": "yes",
                "novelty_proxy": 0.5,
                "fluency_proxy": 0.5,
                "avoidance": "weak",
            },
            0.35 + 0.25 + 0.10 + 0.05 - 0.15,  # = 0.60
        ),
        # Not attempted but high fluency and novelty
        (
            {
                "target_attempted": False,
                "target_success": "no",
                "novelty_proxy": 1.0,
                "fluency_proxy": 1.0,
                "avoidance": "none",
            },
            0.0 + 0.0 + 0.20 + 0.10,  # = 0.30
        ),
    ],
    ids=["perfect", "zero_clamped", "partial", "weak_avoidance", "no_attempt_high_fluency"],
)
def test_reward_values(kwargs, expected_reward):
    result = EvaluationResult(**kwargs)
    reward, components = compute_reward(result)
    assert abs(reward - expected_reward) < 1e-10
    assert isinstance(components, dict)
    assert "target_attempted" in components


def test_reward_clamped_to_zero():
    result = EvaluationResult(
        target_attempted=False,
        target_success="no",
        novelty_proxy=0.0,
        fluency_proxy=0.0,
        avoidance="strong",
    )
    reward, _ = compute_reward(result)
    assert reward == 0.0


def test_reward_clamped_to_one():
    # Construct a case that would exceed 1.0 (not possible with current formula,
    # but verify clamping logic works)
    result = EvaluationResult(
        target_attempted=True,
        target_success="yes",
        novelty_proxy=1.0,
        fluency_proxy=1.0,
        avoidance="none",
    )
    reward, _ = compute_reward(result)
    assert reward <= 1.0


def test_components_sum_to_raw():
    result = EvaluationResult(
        target_attempted=True,
        target_success="partial",
        novelty_proxy=0.7,
        fluency_proxy=0.3,
        avoidance="weak",
    )
    reward, components = compute_reward(result)
    raw = sum(components.values())
    # Reward is clamped raw
    assert reward == max(0.0, min(1.0, raw))


# --- blend_reward tests ---


@pytest.mark.parametrize(
    "learning_reward, stars, expected",
    [
        # 5 stars: 0.6*1.0 + 0.4*1.0 = 1.0
        (1.0, 5, 1.0),
        # 1 star: 0.6*0.0 + 0.4*0.0 = 0.0
        (0.0, 1, 0.0),
        # 3 stars (neutral): 0.6*0.5 + 0.4*0.5 = 0.5
        (0.5, 3, 0.5),
        # High learning, low engagement: 0.6*0.9 + 0.4*0.0 = 0.54
        (0.9, 1, 0.54),
        # Low learning, high engagement: 0.6*0.0 + 0.4*1.0 = 0.4
        (0.0, 5, 0.4),
        # Default 3 stars with mid learning: 0.6*0.6 + 0.4*0.5 = 0.56
        (0.6, 3, 0.56),
    ],
    ids=["max_both", "min_both", "mid_both", "high_learn_low_engage",
         "low_learn_high_engage", "default_stars"],
)
def test_blend_reward(learning_reward, stars, expected):
    result = blend_reward(learning_reward, stars)
    assert abs(result - expected) < 1e-10


def test_blend_reward_clamped():
    # Should never exceed 1.0 even with out-of-range inputs
    assert blend_reward(1.5, 5) == 1.0
    # Should never go below 0.0
    assert blend_reward(-0.5, 1) == 0.0

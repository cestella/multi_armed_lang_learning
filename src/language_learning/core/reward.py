"""Pure reward computation. No side effects."""

from __future__ import annotations

from language_learning.models.evaluation import EvaluationResult


def compute_reward(evaluation: EvaluationResult) -> tuple[float, dict[str, float]]:
    """Compute reward from evaluation result.

    Formula from Architecture.md section 13:
      reward =
        0.35*(target_attempted?1:0) +
        0.25*(target_success=="yes"?1:target_success=="partial"?0.5:0) +
        0.20*novelty_proxy +
        0.10*fluency_proxy -
        0.30*(avoidance=="strong"?1:avoidance=="weak"?0.5:0)

    Returns (clamped_reward, components_dict).
    """
    target_attempted_val = 1.0 if evaluation.target_attempted else 0.0

    target_success_map = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    target_success_val = target_success_map[evaluation.target_success]

    avoidance_map = {"strong": 1.0, "weak": 0.5, "none": 0.0}
    avoidance_val = avoidance_map[evaluation.avoidance]

    components = {
        "target_attempted": 0.35 * target_attempted_val,
        "target_success": 0.25 * target_success_val,
        "novelty": 0.20 * evaluation.novelty_proxy,
        "fluency": 0.10 * evaluation.fluency_proxy,
        "avoidance_penalty": -0.30 * avoidance_val,
    }

    raw = sum(components.values())
    clamped = max(0.0, min(1.0, raw))

    return clamped, components


def blend_reward(learning_reward: float, engagement_stars: int) -> float:
    """Blend learning reward with user engagement rating.

    engagement_stars: 1-5 star rating from user.
    Returns blended reward clamped to [0, 1].
    """
    engagement = (engagement_stars - 1) / 4.0
    return max(0.0, min(1.0, 0.6 * learning_reward + 0.4 * engagement))

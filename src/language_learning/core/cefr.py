"""CEFR estimation logic — derived layer on top of the internal skill model."""

from __future__ import annotations

from datetime import datetime, timezone

from language_learning.models.arms import Arm
from language_learning.models.cefr_state import CefrState, DomainEvidence
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.skill_state import SkillState

CEFR_DOMAINS = ["conversation", "description", "narration", "opinion", "comprehension"]

# Ordered CEFR levels from lowest to highest
CEFR_ORDER = ["A1", "A1+", "A2", "A2+", "A2+/B1-", "B1-", "B1", "B1+", "B1+/B2-", "B2"]


def cefr_rank(level: str) -> int:
    """Return numeric rank of a CEFR level (0 = A1, 9 = B2). Unknown → 0."""
    try:
        return CEFR_ORDER.index(level)
    except ValueError:
        return 0


_DECAY = 0.15  # exponential decay factor

# Domain → relevant skills for skill_bonus calculation
_DOMAIN_SKILLS: dict[str, list[str]] = {
    "conversation": ["vocabulary_description", "agreement"],
    "description": ["vocabulary_description", "agreement"],
    "narration": ["past_narration", "vocabulary_description"],
    "opinion": ["conditional_future", "vocabulary_description", "agreement", "subjunctive_mood"],
    "comprehension": ["vocabulary_description"],
}

# Base importance weights for overall estimate
_DOMAIN_IMPORTANCE: dict[str, float] = {
    "conversation": 0.25,
    "description": 0.20,
    "narration": 0.20,
    "opinion": 0.20,
    "comprehension": 0.15,
}


def update_cefr_state(
    cefr_state: CefrState,
    evaluation: EvaluationResult,
    arm: Arm,
    skill_state: SkillState,
) -> CefrState:
    """Update CEFR state with evidence from a completed evaluation.

    Returns a new CefrState (no mutation of input).
    """
    new_state = cefr_state.model_copy(deep=True)

    # Accumulate evidence for each domain in the arm's cefr_domains
    for domain in arm.cefr_domains:
        if domain in CEFR_DOMAINS:
            _accumulate_evidence(new_state, domain, evaluation)

    # Accumulate comprehension only on engagement (understood + didn't avoid)
    if evaluation.target_attempted and evaluation.avoidance == "none":
        _accumulate_evidence(new_state, "comprehension", evaluation)

    # Recompute all estimates
    _recompute_estimates(new_state, skill_state)

    new_state.last_updated = datetime.now(timezone.utc).isoformat()
    return new_state


def _accumulate_evidence(
    state: CefrState,
    domain: str,
    evaluation: EvaluationResult,
) -> None:
    """Update decayed evidence for a single domain."""
    if domain not in state.evidence:
        state.evidence[domain] = DomainEvidence()

    ev = state.evidence[domain]
    alpha = _DECAY

    attempted = 1.0 if evaluation.target_attempted else 0.0
    success_map = {"no": 0.0, "partial": 0.5, "yes": 1.0}
    success_value = success_map.get(evaluation.target_success, 0.0)
    avoidance_map = {"none": 0.0, "weak": 0.5, "strong": 1.0}
    avoided = avoidance_map.get(evaluation.avoidance, 0.0)

    ev.attempt_rate = ev.attempt_rate * (1 - alpha) + attempted * alpha
    ev.success_rate = ev.success_rate * (1 - alpha) + success_value * alpha
    ev.avoidance_rate = ev.avoidance_rate * (1 - alpha) + avoided * alpha
    ev.fluency_avg = ev.fluency_avg * (1 - alpha) + evaluation.fluency_proxy * alpha
    ev.novelty_avg = ev.novelty_avg * (1 - alpha) + evaluation.novelty_proxy * alpha
    ev.turn_count += 1


def _recompute_estimates(state: CefrState, skill_state: SkillState) -> None:
    """Recompute all domain labels and overall estimate."""
    domain_scores: dict[str, tuple[float, float]] = {}

    for domain in CEFR_DOMAINS:
        if domain in state.evidence:
            score, confidence = _estimate_domain_score(
                domain, state.evidence[domain], skill_state
            )
            # Cap comprehension confidence at 0.5
            if domain == "comprehension":
                confidence = min(confidence, 0.5)
            domain_scores[domain] = (score, confidence)
            state.domains[domain] = _score_to_label(score, confidence)
            state.confidence[domain] = round(confidence, 2)

    # Overall estimate: confidence-weighted average
    if domain_scores:
        weighted_sum = 0.0
        weight_sum = 0.0
        importance_sum = 0.0
        for domain, (score, confidence) in domain_scores.items():
            importance = _DOMAIN_IMPORTANCE.get(domain, 0.15)
            effective_weight = importance * confidence
            weighted_sum += effective_weight * score
            weight_sum += effective_weight
            importance_sum += importance

        if weight_sum > 0:
            overall_score = weighted_sum / weight_sum
            overall_confidence = weight_sum / importance_sum

            # Early-promotion guard: clamp score based on total observations
            total_domain_turns = sum(ev.turn_count for ev in state.evidence.values())
            if total_domain_turns < 5:
                overall_score = min(overall_score, 19.9)   # cap at A1
            elif total_domain_turns < 10:
                overall_score = min(overall_score, 39.9)   # cap at A2

            state.overall_estimate = _score_to_label(overall_score, overall_confidence)
            state.confidence["overall"] = round(overall_confidence, 2)
        else:
            state.overall_estimate = "A1"
            state.confidence["overall"] = 0.0
    else:
        state.overall_estimate = "A1"
        state.confidence["overall"] = 0.0


def _estimate_domain_score(
    domain: str,
    evidence: DomainEvidence,
    skill_state: SkillState,
) -> tuple[float, float]:
    """Estimate a domain score (0-100) and confidence (0-1) from evidence."""
    ev = evidence

    # Heuristic formula from decayed evidence
    base_score = (
        30 * ev.attempt_rate
        + 40 * ev.success_rate
        + 10 * ev.fluency_avg
        + 10 * ev.novelty_avg
        - 15 * ev.avoidance_rate
    )

    # Skill bonus: modest anchor from relevant skill mastery (±5)
    relevant_skills = _DOMAIN_SKILLS.get(domain, [])
    if relevant_skills and skill_state.skills:
        masteries = []
        for skill_name in relevant_skills:
            if skill_name in skill_state.skills:
                masteries.append(skill_state.skills[skill_name].mastery)
        if masteries:
            avg_mastery = sum(masteries) / len(masteries)
            skill_bonus = (avg_mastery - 0.5) * 10
        else:
            skill_bonus = 0.0
    else:
        skill_bonus = 0.0

    score = max(0.0, min(100.0, base_score + skill_bonus))

    # Confidence
    confidence = min(1.0, ev.turn_count / 20) * 0.7
    if ev.avoidance_rate > 0.3:
        confidence *= 0.7

    return score, confidence


def _score_to_label(score: float, confidence: float) -> str:
    """Convert a numeric score (0-100) and confidence to a CEFR label."""
    if confidence < 0.5:
        # Low confidence: coarser labels
        if score < 20:
            return "A1"
        elif score < 40:
            return "A2"
        elif score < 60:
            return "A2+"
        elif score < 80:
            return "B1"
        else:
            return "B1+"
    else:
        # Normal confidence: full scale
        if score < 15:
            return "A1"
        elif score < 25:
            return "A1+"
        elif score < 40:
            return "A2"
        elif score < 55:
            return "A2+"
        elif score < 65:
            return "A2+/B1-"
        elif score < 72:
            return "B1-"
        elif score < 82:
            return "B1"
        elif score < 90:
            return "B1+"
        elif score < 95:
            return "B1+/B2-"
        else:
            return "B2"


def format_cefr_summary(cefr_state: CefrState) -> str:
    """Produce a human-readable CEFR summary."""
    lang = cefr_state.language.upper()
    overall = cefr_state.overall_estimate
    overall_conf = cefr_state.confidence.get("overall", 0.0)

    lines = [
        f"CEFR Estimate ({lang})",
        "=" * 40,
        f"Overall: {overall} (confidence: {overall_conf:.2f})",
        "",
        "By Domain:",
    ]

    total_turns = 0
    for domain in CEFR_DOMAINS:
        if domain in cefr_state.evidence:
            ev = cefr_state.evidence[domain]
            label = cefr_state.domains.get(domain, "A1")
            conf = cefr_state.confidence.get(domain, 0.0)
            turns = ev.turn_count
            total_turns = max(total_turns, turns)

            # Compute approximate attempts from decayed rate
            approx_attempts = int(ev.attempt_rate * turns) if turns > 0 else 0

            if domain == "comprehension":
                lines.append(
                    f"  {domain:<15s}: {label:<14s} (conf: {conf:.2f}, {turns} turns)"
                )
            else:
                lines.append(
                    f"  {domain:<15s}: {label:<14s} "
                    f"(conf: {conf:.2f}, {turns} turns, {approx_attempts} attempts)"
                )
        else:
            lines.append(f"  {domain:<15s}: --             (no data)")

    # Total turns across all evidence
    all_turns = sum(
        ev.turn_count for ev in cefr_state.evidence.values()
    )
    if all_turns > 0:
        lines.append("")
        lines.append(f"Based on {all_turns} domain-turn observations")

    return "\n".join(lines)

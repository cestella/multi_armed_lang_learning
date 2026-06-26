"""Pure UCB1 bandit functions. No side effects, no I/O."""

from __future__ import annotations

import math

from language_learning.models.state import ArmStats, BanditState


def ucb1_select(
    state: BanditState,
    skill_prior: dict[str, float] | None = None,
    eligible_arms: set[str] | None = None,
) -> tuple[str, float, dict[str, float]]:
    """Select an arm using UCB1 with cooldown and optional skill prior.

    eligible_arms: if provided, only arms in this set are candidates.
    Falls back to all arms if the eligible set is empty.

    Returns (arm_id, score, all_scores).
    """
    if not state.arms:
        raise ValueError("No arms configured in bandit state")

    # Resolve candidate set (eligible filter with safety fallback)
    candidates: set[str] = (
        (eligible_arms & state.arms.keys()) if eligible_arms else set(state.arms.keys())
    )
    if not candidates:
        candidates = set(state.arms.keys())

    # 1. Warm-start: if any candidate arm has n==0, pick first by lexical order
    unexplored = sorted(aid for aid in candidates if state.arms[aid].n == 0)
    if unexplored:
        chosen = unexplored[0]
        scores = {aid: 0.0 for aid in state.arms}
        return chosen, 0.0, scores

    # 2. Compute UCB1 scores with cooldown and skill prior (for candidates only)
    scores: dict[str, float] = {}
    for aid, stats in state.arms.items():
        if aid not in candidates:
            scores[aid] = -999.0  # ineligible: excluded from selection
            continue
        base_ucb = stats.mean + state.c * math.sqrt(
            math.log(state.total_pulls) / stats.n
        )
        cooldown_count = state.recent_arms.count(aid)
        prior = skill_prior.get(aid, 0.0) if skill_prior else 0.0
        score = base_ucb + prior - state.cooldown_penalty * cooldown_count
        scores[aid] = score

    # 3. Pick best candidate with lexical tie-breaking
    candidate_scores = {aid: s for aid, s in scores.items() if aid in candidates}
    chosen = _best_arm(candidate_scores)

    # 4. Repeat limit: if last cooldown_max_repeat arms are identical, force different
    if len(state.recent_arms) >= state.cooldown_max_repeat:
        tail = state.recent_arms[-state.cooldown_max_repeat :]
        if len(set(tail)) == 1 and tail[0] == chosen:
            alt_scores = {aid: s for aid, s in candidate_scores.items() if aid != chosen}
            if alt_scores:
                chosen = _best_arm(alt_scores)

    return chosen, scores[chosen], scores


def _best_arm(scores: dict[str, float]) -> str:
    """Pick arm with highest score, tie-break by lexical arm_id."""
    return max(sorted(scores.keys()), key=lambda aid: scores[aid])


def ucb1_update(
    state: BanditState, arm_id: str, reward: float, turn_id: str
) -> BanditState:
    """Update bandit state after observing a reward. Idempotent by turn_id.

    Returns a new BanditState (does not mutate input).
    """
    # Idempotency check
    if turn_id in state.processed_turn_ids:
        return state

    if arm_id not in state.arms:
        raise ValueError(f"Unknown arm: {arm_id}")

    new_state = state.model_copy(deep=True)
    stats = new_state.arms[arm_id]

    new_state.total_pulls += 1
    stats.n += 1
    stats.mean = stats.mean + (reward - stats.mean) / stats.n

    # Update recent_arms, trim to cooldown_max_repeat length
    new_state.recent_arms.append(arm_id)
    if len(new_state.recent_arms) > new_state.cooldown_max_repeat:
        new_state.recent_arms = new_state.recent_arms[-new_state.cooldown_max_repeat :]

    new_state.processed_turn_ids.append(turn_id)

    return new_state

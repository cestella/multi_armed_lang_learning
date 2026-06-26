"""Event-driven controller — pure Python, no HTTP server."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from language_learning.config import TutorConfig
from language_learning.controller.llm_client import LLMClient, LLMError
from language_learning.core.cefr import cefr_rank, format_cefr_summary, update_cefr_state
from language_learning.core.focus import advance_focus, maybe_create_focus
from language_learning.core.reward import blend_reward, compute_reward
from language_learning.core.skills import (
    compute_skill_prior,
    update_profile_display,
    update_skills,
)
from language_learning.models.arms import Arm, load_arms
from language_learning.models.cefr_state import CefrState
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.events import (
    arm_selected,
    assistant_responded,
    bandit_updated,
    engagement_rated,
    evaluation_completed,
    reward_computed,
    session_ended,
    session_started,
    tool_failed,
    user_submitted,
)
from language_learning.models.session_context import PERSONALITIES, SessionContext
from language_learning.models.skill_state import SkillState
from language_learning.models.state import (
    AppState,
    ArmStats,
    BanditState,
    ChatMessage,
    ConversationFocus,
    DebugState,
    FeedbackPanel,
    LearnerProfile,
)
from language_learning.storage.base import StorageBackend
from language_learning.storage.filesystem import FilesystemStorage

logger = logging.getLogger(__name__)

_LANGUAGE_NAMES = {"it": "Italian", "es": "Spanish"}
_INITIATE_SENTINEL = "[INITIATE]"


class Controller:
    """Orchestrates the full learning loop.

    The controller is UI-agnostic: it owns all domain state and exposes a
    clean async API (process_turn, submit_engagement_rating, ...).
    Callers (TUI, HTTP server, tests) pass an on_state_change callback to
    receive AppState snapshots after each mutation.
    """

    def __init__(
        self,
        language: str,
        config: TutorConfig,
        data_dir: str | None = None,
        storage: StorageBackend | None = None,
        on_state_change: Callable[[AppState], Any] | None = None,
    ) -> None:
        if storage is None:
            if data_dir is None:
                raise ValueError("Provide either data_dir or a StorageBackend instance.")
            storage = FilesystemStorage(data_dir)

        self.language = language
        self.storage = storage
        self.llm_client = LLMClient(config)
        self.on_state_change = on_state_change
        self.data_dir = data_dir  # kept for report generation

        self.app_state = AppState(language=language)
        self.bandit_state: BanditState | None = None
        self.learner_profile: LearnerProfile | None = None
        self.skill_state: SkillState | None = None
        self.arms: list[Arm] = []
        self.current_arm: Arm | None = None
        self.cefr_state: CefrState | None = None
        self.conversation_focus: ConversationFocus | None = None
        self.session_context: SessionContext | None = None

        self._focus_consecutive_successes: int = 0
        self._pending_reward: dict[str, Any] | None = None
        self.cefr_override: str | None = None
        self._warmup_pending: bool = False  # True after start_session, consumed on first user turn

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load state from storage and replay last session's events."""
        try:
            arms_data = self.storage.read_arms()
            self.arms = [Arm(**a) for a in arms_data]
        except FileNotFoundError:
            self.arms = []

        bandit_data = self.storage.read_snapshot(self.language, "bandit_state")
        self.bandit_state = (
            BanditState.model_validate(bandit_data)
            if bandit_data
            else self._default_bandit_state()
        )

        profile_data = self.storage.read_snapshot(self.language, "learner_profile")
        self.learner_profile = (
            LearnerProfile.model_validate(profile_data)
            if profile_data
            else LearnerProfile(
                language=self.language,
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        )

        skill_data = self.storage.read_snapshot(self.language, "skill_state")
        self.skill_state = (
            SkillState.model_validate(skill_data)
            if skill_data
            else SkillState(language=self.language)
        )

        cefr_data = self.storage.read_snapshot(self.language, "cefr_state")
        self.cefr_state = (
            CefrState.model_validate(cefr_data)
            if cefr_data
            else CefrState(language=self.language)
        )

        focus_data = self.storage.read_snapshot(self.language, "conversation_focus")
        if focus_data:
            self.conversation_focus = ConversationFocus.model_validate(focus_data)
            self.app_state.conversation_focus = self.conversation_focus

        await self._replay_events()

        if self.bandit_state and self.bandit_state.arms:
            from language_learning.core.bandit import ucb1_select
            skill_prior = self._compute_prior()
            arm_id, score, scores = ucb1_select(
                self.bandit_state, skill_prior, self._eligible_arm_ids()
            )
            self.current_arm = self._find_arm(arm_id)
            self.app_state.current_arm = arm_id
            self.app_state.debug_state.current_arm = arm_id
            self.app_state.debug_state.arm_scores = scores

        event = session_started(self.language)
        self.storage.append_event(self.language, event)
        self.app_state.session_active = True
        self._notify()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def start_session(
        self, topic: str, personality: str, cefr_override: str | None = None
    ) -> None:
        """Set session context and have the tutor open the conversation."""
        self.cefr_override = cefr_override or None
        self.session_context = SessionContext.create(
            language=self.language,
            topic=topic,
            personality=personality,
        )
        self.app_state.turn_count = 0
        self._warmup_pending = True  # next user turn is no-pressure warmup

        # Re-select arm now that we know the CEFR override, so the opening
        # message uses a level-appropriate arm rather than whatever was last active.
        if self.bandit_state and self.bandit_state.arms:
            from language_learning.core.bandit import ucb1_select
            skill_prior = self._compute_prior()
            arm_id, score, scores = ucb1_select(
                self.bandit_state, skill_prior, self._eligible_arm_ids()
            )
            self.current_arm = self._find_arm(arm_id)
            self.app_state.current_arm = arm_id

        await self.process_turn(_INITIATE_SENTINEL)

    def end_session(self) -> None:
        event = session_ended(self.language)
        self.storage.append_event(self.language, event)
        self.app_state.session_active = False
        self._notify()

    async def switch_language(self, new_language: str) -> None:
        self.end_session()
        self.language = new_language
        self.app_state = AppState(language=new_language)
        self.cefr_state = None
        self.conversation_focus = None
        self._focus_consecutive_successes = 0
        self._pending_reward = None
        await self.initialize()

    # ------------------------------------------------------------------
    # Core turn loop
    # ------------------------------------------------------------------

    async def process_turn(self, text: str) -> None:
        """Process one full turn: evaluate (if user message), then respond.

        Pass _INITIATE_SENTINEL to have the tutor open the conversation
        without first evaluating a user message.
        """
        is_initiation = (text == _INITIATE_SENTINEL)
        turn_id = str(uuid.uuid4())

        # Auto-rate any un-rated previous turn before starting a new one
        if self._pending_reward:
            self.submit_engagement_rating(self._pending_reward["turn_id"], stars=3)

        evaluation = EvaluationResult()

        # First user turn after start_session is a warmup: evaluate for CEFR
        # signal and show the feedback card, but skip reward/rating so the
        # learner feels no pressure on their very first response.
        is_warmup_turn = not is_initiation and self._warmup_pending
        if is_warmup_turn:
            self._warmup_pending = False

        if not is_initiation:
            event = user_submitted(self.language, turn_id, text)
            self.storage.append_event(self.language, event)
            self.app_state.chat_messages.append(
                ChatMessage(role="user", text=text, turn_id=turn_id)
            )
            self.app_state.status_line = "Evaluating..."
            self._notify()

            evaluation = await self._do_evaluate(turn_id, text)
            self._apply_evaluation(turn_id, evaluation)

            if not is_warmup_turn:
                reward, components = compute_reward(evaluation)
                self.storage.append_event(
                    self.language, reward_computed(self.language, turn_id, reward, components)
                )
                self._pending_reward = {
                    "turn_id": turn_id,
                    "learning_reward": reward,
                    "components": components,
                    "arm_id": self.current_arm.arm_id if self.current_arm else None,
                }
                self.app_state.pending_rating_turn_id = turn_id
            self._notify()

        self.app_state.status_line = "Responding..."
        self._notify()

        response_text = await self._do_generate(turn_id, text, evaluation, is_initiation)

        self.storage.append_event(
            self.language, assistant_responded(self.language, turn_id, response_text)
        )
        self.app_state.chat_messages.append(
            ChatMessage(role="assistant", text=response_text, turn_id=turn_id)
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if is_initiation:
            self.storage.append_transcript(
                self.language, today, f"\n**Tutor:** {response_text}\n\n---\n"
            )
        else:
            self.storage.append_transcript(
                self.language,
                today,
                f"\n**User:** {text}\n\n**Tutor:** {response_text}\n\n---\n",
            )

        self.app_state.turn_count += 1
        self.app_state.status_line = ""
        self._notify()

    # ------------------------------------------------------------------
    # Engagement rating
    # ------------------------------------------------------------------

    def submit_engagement_rating(self, turn_id: str, stars: int) -> None:
        """Complete the deferred bandit update with the user's engagement rating."""
        if not self._pending_reward or self._pending_reward["turn_id"] != turn_id:
            return

        learning_reward = self._pending_reward["learning_reward"]
        components = self._pending_reward["components"]
        arm_id = self._pending_reward["arm_id"]
        blended = blend_reward(learning_reward, stars)

        self.storage.append_event(
            self.language, engagement_rated(self.language, turn_id, stars, blended)
        )

        if self.bandit_state and arm_id:
            from language_learning.core.bandit import ucb1_update
            self.bandit_state = ucb1_update(self.bandit_state, arm_id, blended, turn_id)
            self.storage.append_event(
                self.language, bandit_updated(self.language, turn_id, arm_id, blended)
            )
            self.storage.write_snapshot(
                self.language, "bandit_state", self.bandit_state.model_dump()
            )

        if self.conversation_focus:
            locked = self._find_arm(self.conversation_focus.source_arm)
            if locked:
                self.current_arm = locked
                self.app_state.current_arm = locked.arm_id
                self.app_state.debug_state = DebugState(
                    current_arm=locked.arm_id,
                    last_reward=blended,
                    reward_components=components,
                    arm_scores=self.app_state.debug_state.arm_scores,
                    focus_skill=self.conversation_focus.focus_skill,
                    focus_turns=self.conversation_focus.turns_remaining,
                )
        elif self.bandit_state:
            from language_learning.core.bandit import ucb1_select
            skill_prior = self._compute_prior()
            new_arm_id, score, scores = ucb1_select(
                self.bandit_state, skill_prior, self._eligible_arm_ids()
            )
            self.current_arm = self._find_arm(new_arm_id)
            self.storage.append_event(
                self.language,
                arm_selected(self.language, turn_id, new_arm_id, score, scores),
            )
            self.app_state.current_arm = new_arm_id
            self.app_state.debug_state = DebugState(
                current_arm=new_arm_id,
                last_reward=blended,
                reward_components=components,
                arm_scores=scores,
            )

        self._pending_reward = None
        self.app_state.pending_rating_turn_id = None
        self._notify()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_stats(self) -> str:
        if not self.bandit_state:
            return "No bandit state loaded."
        lines = [
            f"Language: {self.language}",
            f"Algorithm: {self.bandit_state.algo} (c={self.bandit_state.c})",
            f"Total pulls: {self.bandit_state.total_pulls}",
            f"Recent arms: {self.bandit_state.recent_arms}",
            f"Current arm: {self.app_state.current_arm}",
            "",
            "Arm stats:",
        ]
        for arm_id in sorted(self.bandit_state.arms.keys()):
            s = self.bandit_state.arms[arm_id]
            lines.append(f"  {arm_id}: n={s.n}, mean={s.mean:.4f}")
        return "\n".join(lines)

    def get_topic_info(self) -> str:
        if not self.session_context:
            return "No session active. Use /start <topic> [as <personality>] to begin."
        desc = PERSONALITIES.get(self.session_context.personality, self.session_context.personality)
        return (
            f"Topic: {self.session_context.topic}\n"
            f"Personality: {self.session_context.personality} ({desc})\n"
            f"Started: {self.session_context.started_at}"
        )

    def get_cefr_summary(self) -> str:
        if not self.cefr_state:
            return "No CEFR data available yet."
        return format_cefr_summary(self.cefr_state)

    def get_why(self) -> str:
        ds = self.app_state.debug_state
        lines = [f"Current arm: {ds.current_arm}"]
        if ds.last_reward is not None:
            lines.append(f"Last reward: {ds.last_reward:.4f}")
            lines.append("Components:")
            for k, v in ds.reward_components.items():
                lines.append(f"  {k}: {v:.4f}")
        if ds.arm_scores:
            lines.append("\nArm scores:")
            for aid in sorted(ds.arm_scores.keys()):
                lines.append(f"  {aid}: {ds.arm_scores[aid]:.4f}")
        return "\n".join(lines)

    def generate_reports(self) -> None:
        try:
            from language_learning.reporting.progress_reports import generate_progress_reports
            generate_progress_reports(self.data_dir, self.language)
        except Exception:
            logger.warning("Report generation failed", exc_info=True)

    async def shutdown(self) -> None:
        """No-op — kept for API compatibility with callers."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _do_evaluate(self, turn_id: str, text: str) -> EvaluationResult:
        try:
            return await self.llm_client.evaluate(
                user_text=text,
                chat_history=self.app_state.chat_messages,
                arm=self.current_arm,
                language=self.language,
                session_context=self.session_context,
                learner_profile=self.learner_profile,
                skill_state=self.skill_state,
                cefr_state=self.cefr_state,
                cefr_level_override=self.cefr_override,
            )
        except LLMError as exc:
            logger.warning("Evaluation failed: %s", exc)
            self.storage.append_event(
                self.language, tool_failed(self.language, turn_id, "evaluate", str(exc))
            )
            return EvaluationResult()

    async def _do_generate(
        self,
        turn_id: str,
        user_text: str,
        evaluation: EvaluationResult,
        is_initiation: bool,
    ) -> str:
        fallback = (
            "Mi dispiace, ho avuto un problema tecnico. Puoi ripetere?"
            if self.language == "it"
            else "Lo siento, tuve un problema técnico. ¿Puedes repetir?"
        )
        try:
            if is_initiation:
                return await self.llm_client.initiate(
                    language=self.language,
                    session_context=self.session_context,
                    arm=self.current_arm,
                    cefr_state=self.cefr_state,
                    cefr_level_override=self.cefr_override,
                )
            return await self.llm_client.generate_response(
                user_text=user_text,
                evaluation=evaluation,
                chat_history=self.app_state.chat_messages,
                arm=self.current_arm,
                language=self.language,
                session_context=self.session_context,
                learner_profile=self.learner_profile,
                skill_state=self.skill_state,
                cefr_state=self.cefr_state,
                cefr_level_override=self.cefr_override,
            )
        except LLMError as exc:
            logger.warning("Response generation failed: %s", exc)
            self.storage.append_event(
                self.language,
                tool_failed(self.language, turn_id, "generate_response", str(exc)),
            )
            return fallback

    def _apply_evaluation(self, turn_id: str, evaluation: EvaluationResult) -> None:
        self.storage.append_event(
            self.language,
            evaluation_completed(self.language, turn_id, evaluation.model_dump()),
        )
        self.app_state.feedback_panel = FeedbackPanel(
            praise=evaluation.praise,
            fix_one=evaluation.fix_one,
            micro_rule=evaluation.micro_rule,
            next_nudge=evaluation.next_nudge,
            hint_phrase=evaluation.hint_phrase,
            retry_prompt=evaluation.retry_prompt,
        )

        if self.skill_state and self.current_arm:
            self.skill_state = update_skills(self.skill_state, evaluation, self.current_arm)
            self.storage.write_snapshot(self.language, "skill_state", self.skill_state.model_dump())

        if self.cefr_state and self.current_arm and self.skill_state:
            self.cefr_state = update_cefr_state(
                self.cefr_state, evaluation, self.current_arm, self.skill_state
            )
            self.storage.write_snapshot(self.language, "cefr_state", self.cefr_state.model_dump())

        if self.learner_profile and self.current_arm:
            self.learner_profile = update_profile_display(
                self.learner_profile, evaluation, self.current_arm
            )
            self.storage.write_snapshot(
                self.language, "learner_profile", self.learner_profile.model_dump()
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        if self.skill_state:
            self.storage.append_history(self.language, "skill_history", {
                "ts": now_iso,
                "skills": {
                    name: {
                        "mastery": round(s.mastery, 4),
                        "scaffold_need": round(s.scaffold_need, 4),
                    }
                    for name, s in self.skill_state.skills.items()
                },
            })
        if self.cefr_state:
            self.storage.append_history(self.language, "cefr_history", {
                "ts": now_iso,
                "overall": self.cefr_state.overall_estimate,
                "domains": dict(self.cefr_state.domains),
                "confidence": dict(self.cefr_state.confidence),
            })
        self.storage.append_history(self.language, "session_metrics", {
            "ts": now_iso,
            "target_attempted": evaluation.target_attempted,
            "target_success": evaluation.target_success,
            "avoidance": evaluation.avoidance,
            "fluency_proxy": evaluation.fluency_proxy,
            "novelty_proxy": evaluation.novelty_proxy,
            "arm_id": self.current_arm.arm_id if self.current_arm else None,
        })

        if self.conversation_focus:
            if evaluation.target_success == "yes":
                self._focus_consecutive_successes += 1
            else:
                self._focus_consecutive_successes = 0
            if self._focus_consecutive_successes >= 2:
                self.conversation_focus = None
                self._focus_consecutive_successes = 0
            else:
                self.conversation_focus = advance_focus(self.conversation_focus, evaluation)
        else:
            self.conversation_focus = maybe_create_focus(
                evaluation, self.current_arm, None, self.skill_state
            )
            self._focus_consecutive_successes = 0

        self._save_focus()
        self.app_state.conversation_focus = self.conversation_focus

    def _save_focus(self) -> None:
        if self.conversation_focus:
            self.storage.write_snapshot(
                self.language, "conversation_focus", self.conversation_focus.model_dump()
            )
        else:
            existing = self.storage.read_snapshot(self.language, "conversation_focus")
            if existing is not None:
                self.storage.write_snapshot(self.language, "conversation_focus", {})

    async def _replay_events(self) -> None:
        all_events = self.storage.read_events(self.language)
        last_idx = 0
        for i, e in enumerate(all_events):
            if e.get("type") == "session_started":
                last_idx = i
        for e in all_events[last_idx:]:
            etype = e.get("type")
            payload = e.get("payload", {})
            if etype == "user_submitted":
                self.app_state.chat_messages.append(
                    ChatMessage(
                        role="user",
                        text=payload.get("text", ""),
                        turn_id=e.get("turn_id"),
                    )
                )
            elif etype == "assistant_responded":
                self.app_state.chat_messages.append(
                    ChatMessage(
                        role="assistant",
                        text=payload.get("text", ""),
                        turn_id=e.get("turn_id"),
                    )
                )
            elif etype == "evaluation_completed":
                self.app_state.feedback_panel = FeedbackPanel(
                    praise=payload.get("praise", ""),
                    fix_one=payload.get("fix_one", ""),
                    micro_rule=payload.get("micro_rule"),
                    next_nudge=payload.get("next_nudge", ""),
                    hint_phrase=payload.get("hint_phrase"),
                    retry_prompt=payload.get("retry_prompt"),
                )

    def _default_bandit_state(self) -> BanditState:
        return BanditState(arms={arm.arm_id: ArmStats(n=0, mean=0.0) for arm in self.arms})

    def _find_arm(self, arm_id: str) -> Arm | None:
        return next((a for a in self.arms if a.arm_id == arm_id), None)

    def _compute_prior(self) -> dict[str, float] | None:
        if self.skill_state and self.arms:
            return compute_skill_prior(self.skill_state, self.arms)
        return None

    def _eligible_arm_ids(self) -> set[str] | None:
        """Return arm IDs eligible at the current CEFR level.

        Defaults to A1 when no CEFR data exists (conservative cold start).
        Returns None only if every arm is filtered out (safety valve).
        """
        level = self.cefr_override or (
            getattr(self.cefr_state, "overall_estimate", None) if self.cefr_state else None
        )
        # Default to A1 on first session — don't throw beginners into B1 arms
        if level is None:
            level = "A1"
        rank = cefr_rank(level)
        eligible = {
            a.arm_id for a in self.arms
            if a.cefr_min is None or cefr_rank(a.cefr_min) <= rank
        }
        return eligible if eligible else None

    def _notify(self) -> None:
        if self.on_state_change:
            self.on_state_change(self.app_state)

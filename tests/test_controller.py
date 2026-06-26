"""Comprehensive tests for Controller with mocked LLMClient and InMemoryStorage."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from language_learning.config import TutorConfig
from language_learning.controller.controller import Controller
from language_learning.controller.llm_client import LLMClient, LLMError
from language_learning.core.storage import read_jsonl
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.events import user_submitted, assistant_responded
from language_learning.models.session_context import SessionContext
from language_learning.models.state import (
    ArmStats,
    BanditState,
    ConversationFocus,
)
from language_learning.storage.memory import InMemoryStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARMS_DIR = Path(__file__).parent.parent / "arms"


def _arms_list() -> list[dict]:
    import yaml
    with open(ARMS_DIR / "arms.yaml") as f:
        return yaml.safe_load(f)["arms"]


def _mock_evaluation(**kw) -> EvaluationResult:
    defaults = dict(
        praise="Great use of the past tense!",
        fix_one="Watch verb ending",
        micro_rule="Past participle of -are verbs ends in -ato",
        recast="Sono andato al ristorante ieri sera.",
        next_nudge="What did you order?",
        target_attempted=True,
        target_success="partial",
        errors=[],
        avoidance="none",
        fluency_proxy=0.7,
        novelty_proxy=0.6,
    )
    defaults.update(kw)
    return EvaluationResult.model_validate(defaults)


@pytest.fixture
def storage():
    return InMemoryStorage(arms=_arms_list())


@pytest.fixture
def config():
    return TutorConfig(model="openai/gpt-4o", api_key="test-key")


@pytest.fixture
def controller(storage, config):
    ctrl = Controller(language="it", config=config, storage=storage)
    # Mock the LLM client so no real API calls are made
    ctrl.llm_client = MagicMock(spec=LLMClient)
    ctrl.llm_client.evaluate = AsyncMock(return_value=_mock_evaluation())
    ctrl.llm_client.generate_response = AsyncMock(return_value="Ciao! Come stai oggi?")
    ctrl.llm_client.initiate = AsyncMock(return_value="Benvenuto! Parliamo di viaggi.")
    return ctrl


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialize:
    async def test_creates_default_bandit_state(self, controller, storage):
        await controller.initialize()
        assert controller.bandit_state is not None
        assert len(controller.bandit_state.arms) == 7

    async def test_selects_initial_arm(self, controller):
        await controller.initialize()
        assert controller.current_arm is not None
        assert controller.app_state.current_arm != ""

    async def test_session_marked_active(self, controller, storage):
        await controller.initialize()
        assert controller.app_state.session_active is True

    async def test_session_started_event_logged(self, controller, storage):
        await controller.initialize()
        events = storage.read_events("it")
        assert any(e["type"] == "session_started" for e in events)

    async def test_loads_existing_bandit_state(self, controller, storage):
        state = BanditState(
            arms={"narrative_yesterday": ArmStats(n=5, mean=0.7)},
            total_pulls=5,
        )
        storage.write_snapshot("it", "bandit_state", state.model_dump())
        await controller.initialize()
        assert controller.bandit_state.total_pulls == 5
        assert controller.bandit_state.arms["narrative_yesterday"].mean == 0.7

    async def test_loads_existing_learner_profile(self, controller, storage):
        storage.write_snapshot("it", "learner_profile", {
            "schema_version": 1, "language": "it",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "preferences": {}, "recurring_wins": [], "recurring_fixes": [],
        })
        await controller.initialize()
        assert controller.learner_profile is not None
        assert controller.learner_profile.language == "it"

    async def test_loads_existing_skill_state(self, controller, storage):
        from language_learning.models.skill_state import SkillState
        ss = SkillState(language="it")
        storage.write_snapshot("it", "skill_state", ss.model_dump())
        await controller.initialize()
        assert controller.skill_state is not None

    async def test_loads_existing_cefr_state(self, controller, storage):
        from language_learning.models.cefr_state import CefrState
        cs = CefrState(language="it", overall_estimate="A2")
        storage.write_snapshot("it", "cefr_state", cs.model_dump())
        await controller.initialize()
        assert controller.cefr_state.overall_estimate == "A2"

    async def test_loads_conversation_focus(self, controller, storage):
        focus_data = {
            "focus_skill": "past_narration",
            "source_arm": "narrative_yesterday",
            "turns_remaining": 2,
            "expansion_level": 1,
            "created_at": "2026-03-01T00:00:00+00:00",
        }
        storage.write_snapshot("it", "conversation_focus", focus_data)
        await controller.initialize()
        assert controller.conversation_focus is not None
        assert controller.conversation_focus.focus_skill == "past_narration"

    async def test_replays_chat_history(self, controller, storage):
        e1 = user_submitted("it", "t1", "Ciao!")
        e2 = assistant_responded("it", "t1", "Ciao! Come stai?")
        storage.append_event("it", e1)
        storage.append_event("it", e2)
        await controller.initialize()
        assert len(controller.app_state.chat_messages) == 2
        assert controller.app_state.chat_messages[0].text == "Ciao!"
        assert controller.app_state.chat_messages[1].text == "Ciao! Come stai?"

    async def test_replays_only_from_last_session_started(self, controller, storage):
        from language_learning.models.events import session_started
        # Old session messages
        storage.append_event("it", user_submitted("it", "old", "old message"))
        # New session marker
        storage.append_event("it", session_started("it"))
        # New session message
        storage.append_event("it", user_submitted("it", "new", "new message"))
        await controller.initialize()
        texts = [m.text for m in controller.app_state.chat_messages]
        assert "old message" not in texts
        assert "new message" in texts

    async def test_no_pending_input_on_init(self, controller):
        await controller.initialize()
        assert controller._pending_reward is None

    async def test_notifies_state_change(self, controller):
        notifications = []
        controller.on_state_change = lambda s: notifications.append(s.session_active)
        await controller.initialize()
        assert True in notifications


# ---------------------------------------------------------------------------
# process_turn()
# ---------------------------------------------------------------------------

class TestProcessTurn:
    async def test_happy_path_full_turn(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Sono andato al ristorante ieri")

        events = storage.read_events("it")
        event_types = [e["type"] for e in events]
        assert "user_submitted" in event_types
        assert "evaluation_completed" in event_types
        assert "reward_computed" in event_types
        assert "assistant_responded" in event_types

    async def test_user_message_added_to_chat(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Ciao bella!")
        user_msgs = [m for m in controller.app_state.chat_messages if m.role == "user"]
        assert any(m.text == "Ciao bella!" for m in user_msgs)

    async def test_assistant_message_added_to_chat(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Ciao!")
        assistant_msgs = [m for m in controller.app_state.chat_messages if m.role == "assistant"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[-1].text == "Ciao! Come stai oggi?"

    async def test_feedback_panel_updated(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        panel = controller.app_state.feedback_panel
        assert panel.praise == "Great use of the past tense!"
        assert "verb ending" in panel.fix_one

    async def test_turn_count_increments(self, controller, storage):
        await controller.initialize()
        assert controller.app_state.turn_count == 0
        await controller.process_turn("Test")
        assert controller.app_state.turn_count == 1

    async def test_pending_reward_set_after_turn(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        assert controller._pending_reward is not None
        assert controller.app_state.pending_rating_turn_id is not None

    async def test_status_cleared_after_turn(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        assert controller.app_state.status_line == ""

    async def test_llm_evaluate_called_once(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test message")
        controller.llm_client.evaluate.assert_called_once()

    async def test_llm_generate_called_once(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test message")
        controller.llm_client.generate_response.assert_called_once()

    async def test_evaluation_failure_graceful_degradation(self, controller, storage):
        await controller.initialize()
        controller.llm_client.evaluate = AsyncMock(side_effect=LLMError("API down"))

        # Should not raise; response should still be generated
        await controller.process_turn("Test")

        events = storage.read_events("it")
        event_types = [e["type"] for e in events]
        assert "tool_failed" in event_types
        assert "assistant_responded" in event_types

    async def test_response_failure_logs_event_and_uses_fallback(self, controller, storage):
        await controller.initialize()
        controller.llm_client.generate_response = AsyncMock(side_effect=LLMError("timeout"))

        await controller.process_turn("Test")

        events = storage.read_events("it")
        event_types = [e["type"] for e in events]
        assert "tool_failed" in event_types
        # Fallback message still added to chat
        assistant_msgs = [m for m in controller.app_state.chat_messages if m.role == "assistant"]
        assert len(assistant_msgs) >= 1

    async def test_auto_rates_previous_pending_reward(self, controller, storage):
        await controller.initialize()
        original_arm = controller.current_arm

        # First turn creates pending reward
        await controller.process_turn("First message")
        first_turn_id = controller._pending_reward["turn_id"]
        assert controller._pending_reward is not None

        old_pulls = controller.bandit_state.total_pulls

        # Second turn auto-rates first turn (3 stars) and processes normally
        await controller.process_turn("Second message")

        # Pending reward should be consumed (replaced by new one for second turn)
        assert controller._pending_reward is not None
        assert controller._pending_reward["turn_id"] != first_turn_id
        # Bandit should have been updated for first turn
        assert controller.bandit_state.total_pulls > old_pulls

    async def test_transcript_appended(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Ciao!")
        # Transcript should exist (InMemory has read_transcript)
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        content = storage.read_transcript("it", today)
        assert "Ciao!" in content

    async def test_skills_updated_after_evaluation(self, controller, storage):
        await controller.initialize()
        initial_state = storage.read_snapshot("it", "skill_state")
        await controller.process_turn("Test")
        new_state = storage.read_snapshot("it", "skill_state")
        # skill_state snapshot should be written
        assert new_state is not None

    async def test_cefr_updated_after_evaluation(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        new_state = storage.read_snapshot("it", "cefr_state")
        assert new_state is not None

    async def test_learner_profile_updated_after_evaluation(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        new_profile = storage.read_snapshot("it", "learner_profile")
        assert new_profile is not None

    async def test_spanish_fallback_message(self, config, storage):
        ctrl = Controller(language="es", config=config, storage=storage)
        ctrl.llm_client = MagicMock(spec=LLMClient)
        ctrl.llm_client.evaluate = AsyncMock(return_value=_mock_evaluation())
        ctrl.llm_client.generate_response = AsyncMock(side_effect=LLMError("fail"))
        ctrl.llm_client.initiate = AsyncMock(return_value="¡Hola!")
        await ctrl.initialize()
        ctrl.session_context = SessionContext.create("es", "travel")
        await ctrl.process_turn("Hola")
        assistant_msgs = [m for m in ctrl.app_state.chat_messages if m.role == "assistant"]
        assert any("Lo siento" in m.text for m in assistant_msgs)


# ---------------------------------------------------------------------------
# Initiation turn
# ---------------------------------------------------------------------------

class TestInitiationTurn:
    async def test_initiation_calls_llm_initiate(self, controller, storage):
        await controller.initialize()
        controller.session_context = SessionContext.create("it", "travel")
        await controller.process_turn("[INITIATE]")
        controller.llm_client.initiate.assert_called_once()

    async def test_initiation_does_not_call_evaluate(self, controller, storage):
        await controller.initialize()
        controller.session_context = SessionContext.create("it", "travel")
        await controller.process_turn("[INITIATE]")
        controller.llm_client.evaluate.assert_not_called()

    async def test_initiation_adds_assistant_message(self, controller, storage):
        await controller.initialize()
        controller.session_context = SessionContext.create("it", "travel")
        await controller.process_turn("[INITIATE]")
        assistant_msgs = [m for m in controller.app_state.chat_messages if m.role == "assistant"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[-1].text == "Benvenuto! Parliamo di viaggi."

    async def test_start_session_triggers_initiation(self, controller, storage):
        await controller.initialize()
        await controller.start_session("travel", "encouraging")
        controller.llm_client.initiate.assert_called_once()
        assert controller.session_context.topic == "travel"

    async def test_initiation_no_pending_reward(self, controller, storage):
        await controller.initialize()
        controller.session_context = SessionContext.create("it", "culture")
        await controller.process_turn("[INITIATE]")
        # No evaluation → no pending reward
        assert controller._pending_reward is None


# ---------------------------------------------------------------------------
# Engagement rating
# ---------------------------------------------------------------------------

class TestEngagementRating:
    async def test_rating_updates_bandit(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        turn_id = controller._pending_reward["turn_id"]
        old_pulls = controller.bandit_state.total_pulls

        controller.submit_engagement_rating(turn_id, stars=5)

        assert controller.bandit_state.total_pulls == old_pulls + 1

    async def test_rating_clears_pending_reward(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        turn_id = controller._pending_reward["turn_id"]

        controller.submit_engagement_rating(turn_id, stars=4)

        assert controller._pending_reward is None
        assert controller.app_state.pending_rating_turn_id is None

    async def test_rating_logs_engagement_rated_event(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        turn_id = controller._pending_reward["turn_id"]
        controller.submit_engagement_rating(turn_id, stars=3)

        events = storage.read_events("it")
        assert any(e["type"] == "engagement_rated" for e in events)

    async def test_rating_logs_bandit_updated_event(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        turn_id = controller._pending_reward["turn_id"]
        controller.submit_engagement_rating(turn_id, stars=4)

        events = storage.read_events("it")
        assert any(e["type"] == "bandit_updated" for e in events)

    async def test_wrong_turn_id_is_ignored(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        old_pulls = controller.bandit_state.total_pulls

        controller.submit_engagement_rating("wrong-turn-id", stars=5)

        assert controller.bandit_state.total_pulls == old_pulls
        assert controller._pending_reward is not None

    async def test_rating_selects_next_arm(self, controller, storage):
        await controller.initialize()
        first_arm = controller.current_arm.arm_id
        await controller.process_turn("Test")
        turn_id = controller._pending_reward["turn_id"]
        controller.submit_engagement_rating(turn_id, stars=5)
        # Arm might change (deterministic but depends on bandit state)
        assert controller.current_arm is not None

    async def test_rating_persists_bandit_state(self, controller, storage):
        await controller.initialize()
        await controller.process_turn("Test")
        turn_id = controller._pending_reward["turn_id"]
        controller.submit_engagement_rating(turn_id, stars=5)

        persisted = storage.read_snapshot("it", "bandit_state")
        assert persisted is not None
        assert persisted["total_pulls"] == controller.bandit_state.total_pulls

    async def test_arm_locked_during_focus(self, controller, storage):
        await controller.initialize()
        original_arm = controller.current_arm
        controller.conversation_focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm=original_arm.arm_id,
            turns_remaining=2,
        )
        await controller.process_turn("Test")
        turn_id = controller._pending_reward["turn_id"]
        controller.submit_engagement_rating(turn_id, stars=5)
        assert controller.current_arm.arm_id == original_arm.arm_id


# ---------------------------------------------------------------------------
# Focus management
# ---------------------------------------------------------------------------

class TestConversationFocus:
    async def test_focus_cleared_after_two_consecutive_successes(self, controller, storage):
        await controller.initialize()
        controller.conversation_focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm=controller.current_arm.arm_id,
            turns_remaining=3,
        )
        controller._focus_consecutive_successes = 0

        controller.llm_client.evaluate = AsyncMock(
            return_value=_mock_evaluation(target_success="yes")
        )

        await controller.process_turn("First success")
        assert controller._focus_consecutive_successes == 1

        await controller.process_turn("Second success")
        assert controller.conversation_focus is None
        assert controller._focus_consecutive_successes == 0

    async def test_focus_not_cleared_on_failure(self, controller, storage):
        await controller.initialize()
        controller.conversation_focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm=controller.current_arm.arm_id,
            turns_remaining=3,
        )
        controller._focus_consecutive_successes = 1

        controller.llm_client.evaluate = AsyncMock(
            return_value=_mock_evaluation(target_success="no")
        )

        await controller.process_turn("Failure")
        assert controller._focus_consecutive_successes == 0
        assert controller.conversation_focus is not None

    async def test_focus_persisted_to_snapshot(self, controller, storage):
        await controller.initialize()
        controller.conversation_focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm=controller.current_arm.arm_id,
            turns_remaining=3,
        )
        controller._save_focus()
        snap = storage.read_snapshot("it", "conversation_focus")
        assert snap is not None
        assert snap.get("focus_skill") == "past_narration"

    async def test_empty_focus_snapshot_written_when_cleared(self, controller, storage):
        await controller.initialize()
        # Write a focus snapshot first
        storage.write_snapshot("it", "conversation_focus", {"focus_skill": "x"})
        controller.conversation_focus = None
        controller._save_focus()
        snap = storage.read_snapshot("it", "conversation_focus")
        # Empty dict signals cleared
        assert snap == {}


# ---------------------------------------------------------------------------
# Language switching
# ---------------------------------------------------------------------------

class TestSwitchLanguage:
    async def test_switch_language_changes_language(self, controller, storage):
        await controller.initialize()
        await controller.switch_language("es")
        assert controller.language == "es"
        assert controller.app_state.language == "es"

    async def test_switch_language_ends_old_session(self, controller, storage):
        await controller.initialize()
        await controller.switch_language("es")
        events_it = storage.read_events("it")
        assert any(e["type"] == "session_ended" for e in events_it)

    async def test_switch_language_starts_new_session(self, controller, storage):
        await controller.initialize()
        await controller.switch_language("es")
        events_es = storage.read_events("es")
        assert any(e["type"] == "session_started" for e in events_es)

    async def test_switch_language_resets_focus(self, controller, storage):
        await controller.initialize()
        controller.conversation_focus = ConversationFocus(
            focus_skill="past_narration",
            source_arm=controller.current_arm.arm_id,
            turns_remaining=3,
        )
        await controller.switch_language("es")
        assert controller.conversation_focus is None


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSessionManagement:
    async def test_end_session_marks_inactive(self, controller, storage):
        await controller.initialize()
        controller.end_session()
        assert controller.app_state.session_active is False

    async def test_end_session_logs_event(self, controller, storage):
        await controller.initialize()
        controller.end_session()
        events = storage.read_events("it")
        assert any(e["type"] == "session_ended" for e in events)

    async def test_no_session_context_initially(self, controller):
        await controller.initialize()
        assert controller.session_context is None


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestQueryHelpers:
    async def test_get_stats_returns_string(self, controller):
        await controller.initialize()
        stats = controller.get_stats()
        assert "Language: it" in stats
        assert "Total pulls:" in stats

    async def test_get_why_returns_string(self, controller):
        await controller.initialize()
        why = controller.get_why()
        assert "Current arm:" in why

    async def test_get_topic_info_no_session(self, controller):
        await controller.initialize()
        info = controller.get_topic_info()
        assert "/start" in info

    async def test_get_topic_info_with_session(self, controller):
        await controller.initialize()
        controller.session_context = SessionContext.create("it", "travel", "encouraging")
        info = controller.get_topic_info()
        assert "travel" in info

    async def test_get_cefr_summary_no_data(self, controller):
        await controller.initialize()
        # Fresh CEFR state should still return formatted string
        summary = controller.get_cefr_summary()
        assert len(summary) > 0

    async def test_get_stats_no_bandit(self, controller):
        # Before initialize
        stats = controller.get_stats()
        assert "No bandit state" in stats


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_no_data_dir_no_storage_raises(self, config):
        with pytest.raises(ValueError, match="data_dir"):
            Controller(language="it", config=config)

    def test_storage_provided_no_data_dir_ok(self, config, storage):
        ctrl = Controller(language="it", config=config, storage=storage)
        assert ctrl.storage is storage

    def test_data_dir_creates_filesystem_storage(self, config, tmp_path):
        # Create arms directory
        arms_dst = tmp_path / "arms" / "arms.yaml"
        arms_dst.parent.mkdir(parents=True)
        shutil.copy(ARMS_DIR / "arms.yaml", arms_dst)
        ctrl = Controller(language="it", config=config, data_dir=str(tmp_path))
        from language_learning.storage.filesystem import FilesystemStorage
        assert isinstance(ctrl.storage, FilesystemStorage)

"""Serialization round-trip tests for domain models."""

import json
from pathlib import Path

from language_learning.models.arms import Arm, load_arms
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.events import Event, user_submitted
from language_learning.models.state import ArmStats, BanditState, LearnerProfile


class TestEventRoundTrip:
    def test_serialize_deserialize(self):
        event = user_submitted("it", "turn-1", "Ciao mondo")
        data = event.model_dump()
        assert data["type"] == "user_submitted"
        assert data["payload"]["text"] == "Ciao mondo"

        restored = Event.model_validate(data)
        assert restored.type == event.type
        assert restored.payload == event.payload

    def test_json_round_trip(self):
        event = user_submitted("es", "turn-2", "Hola mundo")
        json_str = event.model_dump_json()
        restored = Event.model_validate_json(json_str)
        assert restored.language == "es"
        assert restored.payload["text"] == "Hola mundo"


class TestBanditStateRoundTrip:
    def test_serialize_deserialize(self):
        state = BanditState(
            arms={
                "arm_a": ArmStats(n=5, mean=0.6),
                "arm_b": ArmStats(n=3, mean=0.4),
            },
            total_pulls=8,
            recent_arms=["arm_a", "arm_b"],
        )
        data = state.model_dump()
        restored = BanditState.model_validate(data)
        assert restored.arms["arm_a"].n == 5
        assert restored.total_pulls == 8

    def test_json_round_trip(self):
        state = BanditState(
            arms={"arm_a": ArmStats(n=1, mean=0.5)},
            total_pulls=1,
        )
        json_str = state.model_dump_json()
        restored = BanditState.model_validate_json(json_str)
        assert restored.arms["arm_a"].mean == 0.5


class TestLearnerProfileRoundTrip:
    def test_serialize_deserialize(self):
        profile = LearnerProfile(language="it", created_at="2024-01-01T00:00:00Z")
        data = profile.model_dump()
        restored = LearnerProfile.model_validate(data)
        assert restored.language == "it"
        assert restored.schema_version == 1


class TestEvaluationResultRoundTrip:
    def test_serialize_deserialize(self):
        result = EvaluationResult(
            praise="Great job!",
            fix_one="Watch your verb endings",
            target_attempted=True,
            target_success="partial",
            fluency_proxy=0.7,
            novelty_proxy=0.8,
            avoidance="none",
        )
        data = result.model_dump()
        restored = EvaluationResult.model_validate(data)
        assert restored.praise == "Great job!"
        assert restored.target_success == "partial"


class TestArmsLoader:
    def test_load_arms_yaml(self):
        arms_path = Path(__file__).parent.parent / "arms" / "arms.yaml"
        arms = load_arms(arms_path)
        assert len(arms) == 7
        arm_ids = {a.arm_id for a in arms}
        assert "narrative_yesterday" in arm_ids
        assert "opinion_compare" in arm_ids

    def test_arm_round_trip(self):
        arm = Arm(
            arm_id="test_arm",
            intent="Test intent",
            prompt_templates=["Template 1"],
            fallback_nudges=["Nudge 1"],
            tags=["test"],
        )
        data = arm.model_dump()
        restored = Arm.model_validate(data)
        assert restored.arm_id == "test_arm"

"""Tests for conversation compression."""

from language_learning.core.compression import compress_conversation
from language_learning.models.skill_state import SkillState, SkillStats
from language_learning.models.state import (
    ChatMessage,
    LearnerProfile,
    RecentFocus,
    RecurringItem,
)


def _make_messages(n: int) -> list[ChatMessage]:
    """Create n alternating user/assistant messages."""
    messages = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(ChatMessage(role=role, text=f"Message {i}", turn_id=f"t{i}"))
    return messages


class TestCompressConversation:
    def test_basic_compression(self):
        messages = _make_messages(10)
        profile = LearnerProfile(
            recurring_fixes=[
                RecurringItem(label="past_tense", count=4),
                RecurringItem(label="agreement", count=2),
            ],
            recurring_wins=[
                RecurringItem(label="narrative", count=3),
            ],
            recent_focus=RecentFocus(label="past_tense", since="2026-03-05"),
        )
        skill_state = SkillState(
            skills={
                "past_narration": SkillStats(mastery=0.3, confidence=0.5),
                "vocab": SkillStats(mastery=0.7, confidence=0.8),
                "agreement": SkillStats(mastery=0.4, confidence=0.3),
            }
        )

        result = compress_conversation(messages, profile, skill_state)

        assert result["message_count"] == 10
        assert len(result["learner_struggles"]) == 2
        assert result["learner_struggles"][0]["label"] == "past_tense"
        assert result["learner_struggles"][0]["count"] == 4
        assert len(result["learner_wins"]) == 1
        assert result["recent_focus"] == "past_tense"
        assert len(result["recent_messages"]) == 6  # default max_recent

        # Only skills with mastery < 0.5
        assert len(result["skill_summary"]) == 2
        skill_names = [s["skill"] for s in result["skill_summary"]]
        assert "past_narration" in skill_names
        assert "agreement" in skill_names
        assert "vocab" not in skill_names

    def test_empty_messages(self):
        profile = LearnerProfile()
        skill_state = SkillState()

        result = compress_conversation([], profile, skill_state)

        assert result["message_count"] == 0
        assert result["recent_messages"] == []
        assert result["learner_struggles"] == []
        assert result["learner_wins"] == []
        assert result["recent_focus"] is None
        assert result["skill_summary"] == []

    def test_fewer_messages_than_max_recent(self):
        messages = _make_messages(4)
        profile = LearnerProfile()
        skill_state = SkillState()

        result = compress_conversation(messages, profile, skill_state, max_recent=6)

        assert result["message_count"] == 4
        assert len(result["recent_messages"]) == 4

    def test_custom_max_recent(self):
        messages = _make_messages(20)
        profile = LearnerProfile()
        skill_state = SkillState()

        result = compress_conversation(messages, profile, skill_state, max_recent=3)

        assert result["message_count"] == 20
        assert len(result["recent_messages"]) == 3

    def test_system_messages_excluded(self):
        messages = [
            ChatMessage(role="user", text="Hello"),
            ChatMessage(role="system", text="System notice"),
            ChatMessage(role="assistant", text="Hi there"),
        ]
        profile = LearnerProfile()
        skill_state = SkillState()

        result = compress_conversation(messages, profile, skill_state)

        # System messages should not count
        assert result["message_count"] == 2
        assert len(result["recent_messages"]) == 2

"""Tests for session context model and /start command parsing."""

from language_learning.models.session_context import (
    PERSONALITIES,
    TOPICS,
    SessionContext,
    parse_start_command,
)


class TestSessionContext:
    def test_create_defaults(self):
        ctx = SessionContext.create()
        assert ctx.language == "it"
        assert ctx.topic == "free conversation"
        assert ctx.personality == "encouraging"
        assert ctx.started_at != ""

    def test_create_custom(self):
        ctx = SessionContext.create(
            language="es",
            topic="travel",
            personality="intellectual",
        )
        assert ctx.language == "es"
        assert ctx.topic == "travel"
        assert ctx.personality == "intellectual"

    def test_topics_list(self):
        assert "current news" in TOPICS
        assert "travel" in TOPICS
        assert "free conversation" in TOPICS
        assert len(TOPICS) == 8

    def test_personalities_dict(self):
        assert "encouraging" in PERSONALITIES
        assert "intellectual" in PERSONALITIES
        assert "formal" in PERSONALITIES
        assert len(PERSONALITIES) == 7


class TestParseStartCommand:
    def test_empty_returns_none(self):
        topic, personality, error = parse_start_command("")
        assert topic is None
        assert personality is None
        assert error is None

    def test_topic_only(self):
        topic, personality, error = parse_start_command("travel")
        assert topic == "travel"
        assert personality == "encouraging"
        assert error is None

    def test_topic_with_personality(self):
        topic, personality, error = parse_start_command("current news as intellectual")
        assert topic == "current news"
        assert personality == "intellectual"
        assert error is None

    def test_topic_case_insensitive(self):
        topic, personality, error = parse_start_command("Travel As Playful")
        assert topic == "travel"
        assert personality == "playful"
        assert error is None

    def test_invalid_topic(self):
        topic, personality, error = parse_start_command("cooking")
        assert topic is None
        assert error is not None
        assert "Unknown topic" in error

    def test_invalid_personality(self):
        topic, personality, error = parse_start_command("travel as angry")
        assert topic is None
        assert error is not None
        assert "Unknown personality" in error

    def test_daily_life_default_personality(self):
        topic, personality, error = parse_start_command("daily life")
        assert topic == "daily life"
        assert personality == "encouraging"
        assert error is None

    def test_free_conversation_as_formal(self):
        topic, personality, error = parse_start_command("free conversation as formal")
        assert topic == "free conversation"
        assert personality == "formal"
        assert error is None

"""Comprehensive tests for LLMClient — all LiteLLM calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from language_learning.config import TutorConfig
from language_learning.controller.llm_client import LLMClient, LLMError, _extract_json
from language_learning.models.evaluation import EvaluationResult
from language_learning.models.session_context import SessionContext
from language_learning.models.skill_state import SkillState, SkillStats
from language_learning.models.state import ChatMessage, LearnerProfile, RecurringItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(model: str = "openai/gpt-4o", **kw) -> TutorConfig:
    return TutorConfig(model=model, **kw)


def _client(model: str = "openai/gpt-4o", **kw) -> LLMClient:
    return LLMClient(_config(model, **kw))


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _good_eval() -> dict:
    return {
        "praise": "Great use of the past tense!",
        "fix_one": "Watch verb agreement: 'andato' not 'andato'",
        "micro_rule": "Past participle of -are verbs ends in -ato",
        "recast": "Sono andato al ristorante ieri sera.",
        "next_nudge": "What did you order?",
        "target_attempted": True,
        "target_success": "partial",
        "errors": [{"type": "spelling", "note": "andatto -> andato"}],
        "avoidance": "none",
        "fluency_proxy": 0.7,
        "novelty_proxy": 0.6,
    }


def _chat_history(n: int = 4) -> list[ChatMessage]:
    msgs = []
    for i in range(n):
        msgs.append(ChatMessage(role="user", text=f"User message {i}"))
        msgs.append(ChatMessage(role="assistant", text=f"Tutor reply {i}"))
    return msgs


# ---------------------------------------------------------------------------
# LLMClient._call_kwargs
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_clean_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_strips_prose_preamble(self):
        raw = 'Sure, here is the evaluation:\n{"a": 1}'
        assert _extract_json(raw) == {"a": 1}

    def test_strips_prose_suffix(self):
        raw = '{"a": 1}\nLet me know if you need anything else.'
        assert _extract_json(raw) == {"a": 1}

    def test_nested_objects(self):
        raw = '{"a": {"b": 2}, "c": [1, 2]}'
        assert _extract_json(raw) == {"a": {"b": 2}, "c": [1, 2]}

    def test_string_with_braces(self):
        raw = '{"msg": "use {this} carefully"}'
        assert _extract_json(raw) == {"msg": "use {this} carefully"}

    def test_truncated_repairs_open_brace(self):
        raw = '{"praise": "Good job"'
        result = _extract_json(raw)
        assert result["praise"] == "Good job"

    def test_truncated_repairs_open_string(self):
        # Mid-string truncation
        raw = '{"praise": "Good'
        result = _extract_json(raw)
        assert "praise" in result

    def test_no_json_raises(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _extract_json("No JSON here at all")

    def test_prose_then_nested_json(self):
        raw = 'Here:\n{"errors": [{"type": "grammar", "note": "fix it"}]}'
        result = _extract_json(raw)
        assert result["errors"][0]["type"] == "grammar"


class TestCallKwargs:
    def test_minimal_config(self):
        client = _client()
        kw = client._call_kwargs()
        assert kw["model"] == "openai/gpt-4o"
        assert kw["timeout"] == 30
        assert kw["max_tokens"] == 1024
        assert "api_base" not in kw
        assert "api_key" not in kw

    def test_ds4_config_includes_api_base(self):
        client = _client(api_base="http://10.0.4.105:8000/v1", api_key="not-needed")
        kw = client._call_kwargs()
        assert kw["api_base"] == "http://10.0.4.105:8000/v1"
        assert kw["api_key"] == "not-needed"

    def test_custom_timeout_and_tokens(self):
        client = _client(timeout=60, max_tokens=2048)
        kw = client._call_kwargs()
        assert kw["timeout"] == 60
        assert kw["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    async def test_happy_path_returns_evaluation_result(self):
        client = _client()
        payload = json.dumps(_good_eval())
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(payload)):
            result = await client.evaluate(
                user_text="Sono andato al ristorante",
                chat_history=_chat_history(),
                arm=None,
                language="it",
            )
        assert isinstance(result, EvaluationResult)
        assert result.target_attempted is True
        assert result.target_success == "partial"
        assert result.fluency_proxy == 0.7

    async def test_italian_language_in_system_prompt(self):
        client = _client()
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], None, "it")

        system_msg = captured[0]["messages"][0]["content"]
        assert "Italian" in system_msg

    async def test_spanish_language_in_system_prompt(self):
        client = _client()
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], None, "es")

        system_msg = captured[0]["messages"][0]["content"]
        assert "Spanish" in system_msg

    async def test_arm_intent_in_system_prompt(self):
        client = _client()
        arm = MagicMock()
        arm.intent = "encourage past-tense narration"
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], arm, "it")

        system_msg = captured[0]["messages"][0]["content"]
        assert "encourage past-tense narration" in system_msg

    async def test_session_context_in_system_prompt(self):
        client = _client()
        ctx = SessionContext.create(language="it", topic="travel", personality="encouraging")
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], None, "it", session_context=ctx)

        system_msg = captured[0]["messages"][0]["content"]
        assert "travel" in system_msg

    async def test_learner_profile_fixes_in_system_prompt(self):
        client = _client()
        profile = LearnerProfile(
            recurring_fixes=[
                RecurringItem(label="verb_agreement", count=5),
                RecurringItem(label="past_tense", count=3),
            ]
        )
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], None, "it", learner_profile=profile)

        system_msg = captured[0]["messages"][0]["content"]
        assert "verb_agreement" in system_msg

    async def test_skill_state_weak_skills_in_system_prompt(self):
        client = _client()
        skill_state = SkillState(skills={"past_narration": SkillStats(mastery=0.2)})
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], None, "it", skill_state=skill_state)

        system_msg = captured[0]["messages"][0]["content"]
        assert "past_narration" in system_msg

    async def test_none_arm_does_not_crash(self):
        client = _client()
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_mock_response(json.dumps(_good_eval()))):
            result = await client.evaluate("test", [], None, "it")
        assert isinstance(result, EvaluationResult)

    async def test_none_session_context_uses_defaults(self):
        client = _client()
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], None, "it", session_context=None)

        system_msg = captured[0]["messages"][0]["content"]
        assert "free conversation" in system_msg

    async def test_partial_json_uses_defaults(self):
        client = _client()
        partial = {"praise": "Good!", "target_attempted": True}
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_mock_response(json.dumps(partial))):
            result = await client.evaluate("test", [], None, "it")
        assert result.praise == "Good!"
        assert result.target_attempted is True
        assert result.target_success == "no"  # default
        assert result.fluency_proxy == 0.5   # default

    async def test_invalid_json_retries_3_times_then_raises(self):
        client = _client()
        call_count = 0

        async def bad_response(**kw):
            nonlocal call_count
            call_count += 1
            return _mock_response("not valid json")

        with patch("litellm.acompletion", side_effect=bad_response):
            with pytest.raises(LLMError, match="3 attempts"):
                await client.evaluate("test", [], None, "it")

        assert call_count == 3

    async def test_succeeds_on_second_attempt(self):
        client = _client()
        call_count = 0

        async def flaky(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response("not json")
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=flaky):
            result = await client.evaluate("test", [], None, "it")

        assert call_count == 2
        assert result.praise == _good_eval()["praise"]

    async def test_succeeds_on_third_attempt(self):
        client = _client()
        call_count = 0

        async def flaky(**kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _mock_response("not json")
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=flaky):
            result = await client.evaluate("test", [], None, "it")

        assert call_count == 3
        assert isinstance(result, EvaluationResult)

    async def test_api_exception_raises_llm_error(self):
        client = _client()
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   side_effect=Exception("connection refused")):
            with pytest.raises(LLMError, match="API call failed"):
                await client.evaluate("test", [], None, "it")

    async def test_uses_model_from_config(self):
        client = _client(model="anthropic/claude-sonnet-4-6")
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], None, "it")

        assert captured[0]["model"] == "anthropic/claude-sonnet-4-6"

    async def test_eval_uses_higher_max_tokens(self):
        """evaluate() overrides max_tokens to at least 2048 regardless of config."""
        client = _client()
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", [], None, "it")

        assert captured[0]["max_tokens"] >= 2048

    async def test_chat_history_included_in_messages(self):
        client = _client()
        history = [
            ChatMessage(role="user", text="Ciao!"),
            ChatMessage(role="assistant", text="Ciao! Come stai?"),
        ]
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("Bene grazie", history, None, "it")

        messages = captured[0]["messages"]
        contents = [m["content"] for m in messages]
        assert any("Ciao!" in c for c in contents)

    async def test_long_history_trimmed_to_window(self):
        client = _client()
        history = _chat_history(20)  # 40 messages total
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response(json.dumps(_good_eval()))

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.evaluate("test", history, None, "it")

        n_history_msgs = len([m for m in captured[0]["messages"] if m["role"] != "system"])
        assert n_history_msgs <= 17  # 8 turns * 2 + user prompt


# ---------------------------------------------------------------------------
# generate_response()
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    async def test_happy_path_returns_string(self):
        client = _client()
        eval_result = EvaluationResult.model_validate(_good_eval())
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_mock_response("Fantastico! E cosa hai ordinato?")):
            result = await client.generate_response(
                user_text="Sono andato al ristorante",
                evaluation=eval_result,
                chat_history=_chat_history(2),
                arm=None,
                language="it",
            )
        assert result == "Fantastico! E cosa hai ordinato?"

    async def test_recast_included_in_system_prompt(self):
        client = _client()
        eval_result = EvaluationResult(
            recast="Sono andato al ristorante ieri sera.",
            next_nudge="What did you order?",
        )
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response("OK!")

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.generate_response("test", eval_result, [], None, "it")

        system_msg = captured[0]["messages"][0]["content"]
        assert "Sono andato al ristorante ieri sera." in system_msg

    async def test_next_nudge_included_in_system_prompt(self):
        client = _client()
        eval_result = EvaluationResult(
            recast="Correct form.",
            next_nudge="What was the best part of your day?",
        )
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response("OK!")

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.generate_response("test", eval_result, [], None, "it")

        system_msg = captured[0]["messages"][0]["content"]
        assert "What was the best part of your day?" in system_msg

    async def test_personality_in_system_prompt(self):
        client = _client()
        ctx = SessionContext.create(language="it", topic="travel", personality="formal")
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response("OK!")

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.generate_response(
                "test", EvaluationResult(), [], None, "it", session_context=ctx
            )

        system_msg = captured[0]["messages"][0]["content"]
        assert "formal" in system_msg.lower() or "Polished" in system_msg

    async def test_api_error_raises_llm_error(self):
        client = _client()
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   side_effect=Exception("timeout")):
            with pytest.raises(LLMError, match="API call failed"):
                await client.generate_response(
                    "test", EvaluationResult(), [], None, "it"
                )

    async def test_history_included_in_messages(self):
        client = _client()
        history = [
            ChatMessage(role="user", text="What did you do?"),
            ChatMessage(role="assistant", text="I went to the market."),
        ]
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response("Great!")

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.generate_response("test", EvaluationResult(), history, None, "it")

        messages = captured[0]["messages"]
        contents = [m["content"] for m in messages]
        assert any("I went to the market." in c for c in contents)

    async def test_strips_whitespace_from_response(self):
        client = _client()
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_mock_response("  Ciao!  \n")):
            result = await client.generate_response(
                "test", EvaluationResult(), [], None, "it"
            )
        assert result == "Ciao!"

    async def test_empty_content_returns_empty_string(self):
        client = _client()
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_mock_response("")):
            result = await client.generate_response(
                "test", EvaluationResult(), [], None, "it"
            )
        assert result == ""


# ---------------------------------------------------------------------------
# initiate()
# ---------------------------------------------------------------------------

class TestInitiate:
    async def test_happy_path_returns_string(self):
        client = _client()
        ctx = SessionContext.create(language="it", topic="travel", personality="encouraging")
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_mock_response("Ciao! Parliamo di viaggi. Dove sei stato?")):
            result = await client.initiate(language="it", session_context=ctx)
        assert "Ciao" in result or result  # non-empty

    async def test_topic_in_prompt(self):
        client = _client()
        ctx = SessionContext.create(language="it", topic="sports", personality="direct")
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response("Parliamo di sport!")

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.initiate("it", session_context=ctx)

        user_msg = captured[0]["messages"][0]["content"]
        assert "sports" in user_msg

    async def test_language_in_prompt(self):
        client = _client()
        ctx = SessionContext.create(language="es", topic="culture", personality="intellectual")
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response("Hola!")

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.initiate("es", session_context=ctx)

        user_msg = captured[0]["messages"][0]["content"]
        assert "Spanish" in user_msg

    async def test_personality_in_prompt(self):
        client = _client()
        ctx = SessionContext.create(language="it", topic="travel", personality="funny")
        captured: list[dict] = []

        async def fake_completion(**kw):
            captured.append(kw)
            return _mock_response("Ha!")

        with patch("litellm.acompletion", side_effect=fake_completion):
            await client.initiate("it", session_context=ctx)

        user_msg = captured[0]["messages"][0]["content"]
        assert "funny" in user_msg.lower() or "humor" in user_msg.lower() or "joke" in user_msg.lower()

    async def test_none_session_context_does_not_crash(self):
        client = _client()
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_mock_response("Ciao!")):
            result = await client.initiate("it", session_context=None)
        assert isinstance(result, str)

    async def test_api_error_raises_llm_error(self):
        client = _client()
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   side_effect=Exception("network error")):
            with pytest.raises(LLMError, match="API call failed"):
                await client.initiate("it")

    async def test_different_personalities(self):
        for personality in ("encouraging", "funny", "patient", "intellectual", "direct", "playful", "formal"):
            client = _client()
            ctx = SessionContext.create(language="it", topic="travel", personality=personality)
            captured: list[dict] = []

            async def fake_completion(**kw):
                captured.append(kw)
                return _mock_response("Ciao!")

            with patch("litellm.acompletion", side_effect=fake_completion):
                await client.initiate("it", session_context=ctx)

            assert captured, f"No call for personality={personality}"


# ---------------------------------------------------------------------------
# _learner_context helper
# ---------------------------------------------------------------------------

class TestLearnerContext:
    def test_empty_profile_and_skills(self):
        ctx = LLMClient._learner_context(None, None)
        assert ctx == ""

    def test_recurring_fixes_listed(self):
        profile = LearnerProfile(
            recurring_fixes=[RecurringItem(label="verb_agreement", count=5)]
        )
        ctx = LLMClient._learner_context(profile, None)
        assert "verb_agreement" in ctx

    def test_recurring_wins_listed(self):
        profile = LearnerProfile(
            recurring_wins=[RecurringItem(label="past_tense", count=3)]
        )
        ctx = LLMClient._learner_context(profile, None)
        assert "past_tense" in ctx

    def test_weak_skills_listed(self):
        skills = SkillState(skills={"vocab": SkillStats(mastery=0.2)})
        ctx = LLMClient._learner_context(None, skills)
        assert "vocab" in ctx

    def test_strong_skills_not_listed(self):
        skills = SkillState(skills={"vocab": SkillStats(mastery=0.9)})
        ctx = LLMClient._learner_context(None, skills)
        assert "vocab" not in ctx

    def test_capped_at_four_weak_skills(self):
        skills = SkillState(skills={
            f"skill_{i}": SkillStats(mastery=0.1) for i in range(10)
        })
        ctx = LLMClient._learner_context(None, skills)
        assert ctx.count("skill_") <= 4


# ---------------------------------------------------------------------------
# _history_messages helper
# ---------------------------------------------------------------------------

class TestHistoryMessages:
    def test_empty_history(self):
        assert LLMClient._history_messages([]) == []

    def test_system_messages_excluded(self):
        history = [ChatMessage(role="system", text="System note")]
        result = LLMClient._history_messages(history)
        assert result == []

    def test_user_and_assistant_included(self):
        history = [
            ChatMessage(role="user", text="Hello"),
            ChatMessage(role="assistant", text="Hi"),
        ]
        result = LLMClient._history_messages(history)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}

    def test_long_history_trimmed(self):
        history = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", text=f"msg {i}")
            for i in range(40)
        ]
        result = LLMClient._history_messages(history, max_turns=5)
        assert len(result) == 10  # 5 turns * 2 messages

    def test_returns_last_n_turns(self):
        history = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", text=f"msg {i}")
            for i in range(10)
        ]
        result = LLMClient._history_messages(history, max_turns=2)
        assert result[0]["content"] == "msg 6"
        assert result[-1]["content"] == "msg 9"

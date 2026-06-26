"""LiteLLM-backed LLM client for evaluation and response generation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import litellm

from language_learning.config import TutorConfig
from language_learning.models.evaluation import EvaluationResult

if TYPE_CHECKING:
    from language_learning.models.arms import Arm
    from language_learning.models.cefr_state import CefrState
    from language_learning.models.session_context import SessionContext
    from language_learning.models.skill_state import SkillState
    from language_learning.models.state import ChatMessage, LearnerProfile

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

_LANGUAGE_NAMES = {"it": "Italian", "es": "Spanish"}

_EVALUATION_SCHEMA = """\
{
  "praise": "<string ≤120 chars — one specific thing done well>",
  "fix_one": "<string ≤160 chars — exactly one improvement>",
  "micro_rule": "<string ≤120 chars or null — brief grammar/vocab rule>",
  "recast": "<string — 1-2 sentences naturally modeling the correction>",
  "next_nudge": "<string — follow-up question to continue conversation>",
  "target_attempted": <true|false>,
  "target_success": "<no|partial|yes>",
  "errors": [{"type": "<string>", "note": "<string>"}],
  "avoidance": "<none|weak|strong>",
  "fluency_proxy": <0.0-1.0>,
  "novelty_proxy": <0.0-1.0>,
  "hint_phrase": "<string or null>",
  "retry_prompt": "<string or null>"
}"""

_EVAL_SYSTEM = """\
You are an expert {language_name} language tutor evaluating a student message.
Conversation topic: {topic}
Current conversational goal: {arm_intent}
{learner_context}

Respond with ONLY a JSON object matching this schema exactly:
{schema}

Rules:
- praise must highlight something genuinely good, not generic
- fix_one must name exactly one error or improvement — the highest-priority one
- target_success is whether the student demonstrated the conversational goal
- fluency_proxy and novelty_proxy are 0.0–1.0 floats
- avoidance is "strong" if the student deliberately sidestepped the goal
"""

_RESPONSE_SYSTEM = """\
You are a {personality_desc} {language_name} language tutor having a conversation about {topic}.
Your conversational goal this turn: {arm_intent}
{learner_context}
{coaching_note}

Rules:
- Respond naturally, primarily in {language_name}
- Keep your response to 2–4 sentences
- Do NOT explicitly mention grammar rules or corrections
- End with an engaging follow-up question
{cefr_rule}"""

_INITIATE_SYSTEM = """\
You are a {personality_desc} {language_name} language tutor.
You are opening a conversation about: {topic}
Your goal for this opening: {arm_intent}

Rules:
- Write a natural, engaging opening message of 2–3 sentences in {language_name}
- End with a question that invites the student to respond
{cefr_rule}"""


def _cefr_grammar_rule(level: str | None) -> str:
    """Return an explicit grammar constraint for the given CEFR level.

    Empty string at B1 and above — the LLM can use full grammar freely.
    """
    rules = {
        "A1": (
            "- HARD GRAMMAR RULE (student is A1): "
            "Use ONLY simple present tense. "
            "FORBIDDEN: conditional (farei/farías), subjunctive (trovassi/tuvieras), "
            "imperfect (avevo/tenía), future tense, complex relative clauses. "
            "Maximum 8 words per sentence. One idea per sentence."
        ),
        "A1+": (
            "- HARD GRAMMAR RULE (student is A1+): "
            "Use present tense and simple perfect past only (passato prossimo / pretérito perfecto). "
            "FORBIDDEN: conditional, subjunctive, imperfect (except essere/stare/avere for context). "
            "Keep sentences short and direct."
        ),
        "A2": (
            "- GRAMMAR RULE (student is A2): "
            "Use present, simple past, and simple future. "
            "AVOID: conditional, subjunctive, complex hypotheticals (if X then Y structures)."
        ),
        "A2+": (
            "- GRAMMAR RULE (student is A2+): "
            "Basic tenses OK (present, past, imperfect for background, near future). "
            "AVOID: conditional perfect, present/past subjunctive."
        ),
        "A2+/B1-": (
            "- GRAMMAR RULE (student is A2+/B1-): "
            "All common tenses OK. Limit complex subjunctive and conditional-perfect constructions."
        ),
    }
    return rules.get(level or "", "")


def _extract_json(raw: str) -> dict:
    """Extract the first complete JSON object from raw text.

    Handles models that prepend prose before the JSON block, and attempts
    to repair truncated JSON by closing any unclosed braces/brackets.
    """
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    # Find the matching closing brace by counting depth
    depth = 0
    end = -1
    in_str = False
    escape = False
    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        # Truncated — close open braces as a best-effort repair
        fragment = raw[start:]
        open_braces = fragment.count("{") - fragment.count("}")
        open_brackets = fragment.count("[") - fragment.count("]")
        fragment = fragment.rstrip(", \n\t")
        if in_str:
            fragment += '"'
        fragment += "]" * open_brackets + "}" * open_braces
        return json.loads(fragment)
    return json.loads(raw[start : end + 1])


class LLMError(Exception):
    """Raised when an LLM call fails in an unrecoverable way."""


class LLMClient:
    """Thin wrapper around litellm.acompletion for tutor-specific calls."""

    def __init__(self, config: TutorConfig) -> None:
        self._config = config

    def _call_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "timeout": self._config.timeout,
            "max_tokens": self._config.max_tokens,
        }
        if self._config.api_base:
            kwargs["api_base"] = self._config.api_base
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        return kwargs

    @staticmethod
    def _learner_context(
        learner_profile: LearnerProfile | None,
        skill_state: SkillState | None,
        cefr_state: Any | None = None,
        cefr_level_override: str | None = None,
    ) -> str:
        lines: list[str] = []

        level = cefr_level_override or (
            getattr(cefr_state, "overall_estimate", None) if cefr_state else None
        )
        if level:
            source = "self-reported" if cefr_level_override else "estimated"
            lines.append(f"Student CEFR level: {level} ({source})")

        if learner_profile:
            if learner_profile.recurring_fixes:
                fixes = ", ".join(
                    f"{f.label} (×{f.count})" for f in learner_profile.recurring_fixes[:3]
                )
                lines.append(f"Common errors to watch: {fixes}")
            if learner_profile.recurring_wins:
                wins = ", ".join(w.label for w in learner_profile.recurring_wins[:3])
                lines.append(f"Strengths: {wins}")
        if skill_state:
            weak = [
                f"{name} ({s.mastery:.0%})"
                for name, s in skill_state.skills.items()
                if s.mastery < 0.5
            ]
            if weak:
                lines.append(f"Weak skills: {', '.join(weak[:4])}")
        return "\n".join(lines)

    @staticmethod
    def _history_messages(
        chat_history: list[ChatMessage],
        max_turns: int = 8,
    ) -> list[dict[str, str]]:
        """Convert ChatMessage list to LiteLLM message dicts, last N turns."""
        return [
            {"role": m.role, "content": m.text}
            for m in chat_history[-max_turns * 2:]
            if m.role in ("user", "assistant")
        ]

    async def evaluate(
        self,
        user_text: str,
        chat_history: list[ChatMessage],
        arm: Arm | None,
        language: str,
        session_context: SessionContext | None = None,
        learner_profile: LearnerProfile | None = None,
        skill_state: SkillState | None = None,
        cefr_state: CefrState | None = None,
        cefr_level_override: str | None = None,
    ) -> EvaluationResult:
        """Call the LLM to evaluate the student's message.

        Returns EvaluationResult with safe defaults on partial JSON.
        Raises LLMError on total failure.
        """
        language_name = _LANGUAGE_NAMES.get(language, language.title())
        topic = session_context.topic if session_context else "free conversation"
        arm_intent = arm.intent if arm else "general conversation"
        learner_ctx = self._learner_context(learner_profile, skill_state, cefr_state, cefr_level_override)

        system = _EVAL_SYSTEM.format(
            language_name=language_name,
            topic=topic,
            arm_intent=arm_intent,
            learner_context=learner_ctx,
            schema=_EVALUATION_SCHEMA,
        )

        # Provide recent history as context, then ask to evaluate the latest turn
        history_msgs = self._history_messages(chat_history[:-1])  # exclude current user msg
        user_prompt = (
            f'Evaluate this student message in {language_name}:\n"{user_text}"'
        )

        messages = [{"role": "system", "content": system}]
        if history_msgs:
            messages.extend(history_msgs)
        messages.append({"role": "user", "content": user_prompt})

        eval_kwargs = self._call_kwargs()
        eval_kwargs["max_tokens"] = max(eval_kwargs.get("max_tokens", 1024), 2048)

        last_exc: Exception | None = None
        data: dict | None = None
        for attempt in range(1, 4):
            try:
                response = await litellm.acompletion(**eval_kwargs, messages=messages)
                raw = response.choices[0].message.content or ""
            except Exception as exc:
                raise LLMError(f"evaluate() API call failed: {exc}") from exc

            try:
                data = _extract_json(raw)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_exc = exc
                logger.warning("evaluate() attempt %d/%d invalid JSON: %s", attempt, 3, exc)

        if data is None:
            raise LLMError(
                f"evaluate() failed to return valid JSON after 3 attempts: {last_exc}"
            )

        try:
            return EvaluationResult.model_validate(data)
        except Exception as exc:
            logger.warning("EvaluationResult validation failed, using defaults: %s", exc)
            # Return what we can parse, falling back to model defaults
            safe: dict[str, Any] = {}
            for field in ("praise", "fix_one", "micro_rule", "recast", "next_nudge",
                          "target_attempted", "target_success", "errors",
                          "avoidance", "fluency_proxy", "novelty_proxy",
                          "hint_phrase", "retry_prompt"):
                if field in data:
                    safe[field] = data[field]
            return EvaluationResult.model_validate(safe)

    async def generate_response(
        self,
        user_text: str,
        evaluation: EvaluationResult,
        chat_history: list[ChatMessage],
        arm: Arm | None,
        language: str,
        session_context: SessionContext | None = None,
        learner_profile: LearnerProfile | None = None,
        skill_state: SkillState | None = None,
        cefr_state: CefrState | None = None,
        cefr_level_override: str | None = None,
    ) -> str:
        """Generate the tutor's conversational reply.

        Raises LLMError on failure.
        """
        from language_learning.models.session_context import PERSONALITIES

        language_name = _LANGUAGE_NAMES.get(language, language.title())
        topic = session_context.topic if session_context else "free conversation"
        personality = session_context.personality if session_context else "encouraging"
        personality_desc = PERSONALITIES.get(personality, personality)
        arm_intent = arm.intent if arm else "general conversation"
        level = cefr_level_override or (getattr(cefr_state, "overall_estimate", None) if cefr_state else None)
        learner_ctx = self._learner_context(learner_profile, skill_state, cefr_state, cefr_level_override)

        coaching_note = ""
        if evaluation.recast:
            coaching_note = f"Coaching note (do NOT quote directly): {evaluation.recast}"
            if evaluation.next_nudge:
                coaching_note += f"\nFollow-up to use: {evaluation.next_nudge}"

        system = _RESPONSE_SYSTEM.format(
            personality_desc=personality_desc,
            language_name=language_name,
            topic=topic,
            arm_intent=arm_intent,
            learner_context=learner_ctx,
            coaching_note=coaching_note,
            cefr_rule=_cefr_grammar_rule(level),
        )

        history_msgs = self._history_messages(chat_history)
        messages = [{"role": "system", "content": system}]
        messages.extend(history_msgs)

        try:
            response = await litellm.acompletion(
                **self._call_kwargs(),
                messages=messages,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise LLMError(f"generate_response() API call failed: {exc}") from exc

    async def initiate(
        self,
        language: str,
        session_context: SessionContext | None = None,
        arm: Arm | None = None,
        cefr_state: CefrState | None = None,
        cefr_level_override: str | None = None,
    ) -> str:
        """Generate the tutor's opening message for a new session.

        Raises LLMError on failure.
        """
        from language_learning.models.session_context import PERSONALITIES

        language_name = _LANGUAGE_NAMES.get(language, language.title())
        topic = session_context.topic if session_context else "free conversation"
        personality = session_context.personality if session_context else "encouraging"
        personality_desc = PERSONALITIES.get(personality, personality)

        level = cefr_level_override or (getattr(cefr_state, "overall_estimate", None) if cefr_state else None)
        arm_intent = arm.intent if arm else "open a natural, friendly conversation"

        system = _INITIATE_SYSTEM.format(
            personality_desc=personality_desc,
            language_name=language_name,
            topic=topic,
            arm_intent=arm_intent,
            cefr_rule=_cefr_grammar_rule(level),
        )

        try:
            response = await litellm.acompletion(
                **self._call_kwargs(),
                messages=[{"role": "user", "content": system}],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise LLMError(f"initiate() API call failed: {exc}") from exc

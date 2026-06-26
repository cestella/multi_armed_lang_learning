"""Evaluation result model returned by the LLM evaluator."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ErrorItem(BaseModel):
    type: str
    note: str


class EvaluationResult(BaseModel):
    praise: str = ""
    fix_one: str = ""
    micro_rule: str | None = None
    recast: str = ""
    next_nudge: str = ""
    target_attempted: bool = False
    target_success: Literal["no", "partial", "yes"] = "no"
    errors: list[ErrorItem] = Field(default_factory=list)
    avoidance: Literal["none", "weak", "strong"] = "none"
    fluency_proxy: float = 0.5
    novelty_proxy: float = 0.5
    hint_phrase: str | None = None
    retry_prompt: str | None = None

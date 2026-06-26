"""CEFR estimation state model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DomainEvidence(BaseModel):
    """Decayed accumulator for a single CEFR domain.

    Uses exponential decay so recent turns weigh more than old ones.
    After each turn: metric = metric * decay + new_value * (1 - decay)
    """

    attempt_rate: float = 0.0
    success_rate: float = 0.0
    avoidance_rate: float = 0.0
    fluency_avg: float = 0.5
    novelty_avg: float = 0.5
    turn_count: int = 0


class CefrState(BaseModel):
    schema_version: int = 1
    language: str = "it"
    overall_estimate: str = "A1"
    domains: dict[str, str] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, DomainEvidence] = Field(default_factory=dict)
    last_updated: str = ""

"""Skill state model for tracking learner mastery and confidence."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillStats(BaseModel):
    mastery: float = 0.5
    confidence: float = 0.0
    scaffold_need: float = 0.3
    last_seen: str | None = None


class SkillState(BaseModel):
    schema_version: int = 1
    language: str = "it"
    skills: dict[str, SkillStats] = Field(default_factory=dict)
    updated_at: str = ""

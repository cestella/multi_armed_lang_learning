"""Arm definitions and YAML loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Arm(BaseModel):
    arm_id: str
    intent: str = ""
    prompt_templates: list[str] = Field(default_factory=list)
    fallback_nudges: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    target_skills: dict[str, float] = Field(default_factory=dict)
    cefr_domains: list[str] = Field(default_factory=list)
    cefr_min: str | None = None  # minimum CEFR level to select this arm (None = always eligible)


def load_arms(path: str | Path) -> list[Arm]:
    """Load arms from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return [Arm(**arm) for arm in data["arms"]]

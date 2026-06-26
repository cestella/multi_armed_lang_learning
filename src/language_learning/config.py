"""Configuration loading for the language tutor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator, model_validator

_SEARCH_PATHS = [
    Path("config.yaml"),
    Path.home() / ".config" / "language-tutor" / "config.yaml",
]


class TutorConfig(BaseModel):
    """LLM provider configuration.

    model:    Any LiteLLM model string, e.g.:
                "anthropic/claude-sonnet-4-6"
                "openai/gpt-4o"
                "openai/deepseek-v4"   (with api_base pointing at ds4)
    api_base: Base URL override for OpenAI-compatible local servers.
              Leave null to use the provider's default endpoint.
    api_key:  API key. If null, LiteLLM reads the appropriate env var
              (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.).
    timeout:  Per-request timeout in seconds.
    max_tokens: Maximum tokens in the LLM response.
    """

    model: str
    api_base: str | None = None
    api_key: str | None = None
    timeout: int = 30
    max_tokens: int = 1024

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model must not be empty")
        return v

    @field_validator("timeout")
    @classmethod
    def timeout_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    model_config = {"extra": "ignore"}


def load_config(path: str | Path | None = None) -> TutorConfig:
    """Load TutorConfig from a YAML file.

    If path is None, searches ./config.yaml then
    ~/.config/language-tutor/config.yaml.

    Raises FileNotFoundError if no config is found.
    Raises ValueError (via Pydantic) for invalid config.
    """
    resolved: Path
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
    else:
        for candidate in _SEARCH_PATHS:
            if candidate.exists():
                resolved = candidate
                break
        else:
            paths = ", ".join(str(p) for p in _SEARCH_PATHS)
            raise FileNotFoundError(
                f"No config.yaml found. Searched: {paths}\n"
                "Copy config.example.yaml to config.yaml and edit it."
            )

    with open(resolved) as f:
        raw: Any = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    return TutorConfig.model_validate(raw)

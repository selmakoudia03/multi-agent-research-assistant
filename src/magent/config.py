"""Runtime configuration read from environment variables.

Kept dependency-free (no pydantic-settings) so the package has a minimal
install footprint. All settings have safe defaults that work offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.environ.get("MAGENT_LLM_PROVIDER", "mock")
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    openai_api_key: str | None = os.environ.get("OPENAI_API_KEY")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_reflection_iterations: int = int(os.environ.get("MAGENT_MAX_ITERATIONS", "3"))
    tool_timeout_seconds: float = float(os.environ.get("MAGENT_TOOL_TIMEOUT", "5"))


def get_settings() -> Settings:
    return Settings()

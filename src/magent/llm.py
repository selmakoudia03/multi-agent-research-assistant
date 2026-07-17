"""LLM provider abstraction.

Every agent talks to an `LLMProvider`, never to a vendor SDK directly. This
keeps the core framework testable offline (via `MockProvider`) and lets
callers swap Anthropic / OpenAI without touching agent code.

Real providers are implemented with plain HTTP calls (`requests`) so the
package does not hard-depend on the `anthropic` or `openai` SDKs.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

from magent.config import Settings, get_settings


class LLMProvider(ABC):
    """Minimal chat-completion interface used by all agents."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Return a completion for `prompt` given an optional `system` prompt."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Calls the Anthropic Messages API directly over HTTP."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicProvider")
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        import requests

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1024,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(block.get("text", "") for block in data.get("content", []))


class OpenAIProvider(LLMProvider):
    """Calls the OpenAI Chat Completions API directly over HTTP."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            json={"model": self.model, "messages": messages},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class MockProvider(LLMProvider):
    """Deterministic, offline stand-in for a real LLM.

    Used as the default provider so the whole codebase (tests, CI, demos)
    runs without any API key or network access. Responses are derived
    heuristically from the prompt so behaviour stays sensible and
    reproducible instead of being hardcoded per test case.
    """

    def complete(self, prompt: str, system: str = "") -> str:
        text = f"{system}\n{prompt}"

        if "Break the following user request into 2-4 short, concrete subtasks" in system:
            return self._mock_plan(prompt)
        if "Decide if the DRAFT ANSWER fully" in system:
            return self._mock_critique(prompt)
        if "Combine the research notes below into one clear, well-structured answer" in system:
            return self._mock_synthesis(prompt)

        # Generic fallback: short deterministic echo, useful for ad-hoc calls.
        digest = hashlib.sha256(text.encode()).hexdigest()[:8]
        return f"[mock-response {digest}] {prompt.strip()[:200]}"

    @staticmethod
    def _mock_plan(prompt: str) -> str:
        match = re.search(r"User request:\s*(.+)", prompt, re.DOTALL)
        query = match.group(1).strip() if match else prompt.strip()
        subtasks = [f"Research background information about: {query}"]
        if re.search(r"\d", query) or any(
            op in query for op in ("+", "-", "*", "/", "calculate", "compute")
        ):
            subtasks.append(f"Compute any numeric part of: {query}")
        subtasks.append(f"Summarize findings to directly answer: {query}")
        return "\n".join(f"- {s}" for s in subtasks)

    @staticmethod
    def _mock_synthesis(prompt: str) -> str:
        notes_match = re.search(r"Research notes:\s*(.+?)\n\nUser request:", prompt, re.DOTALL)
        query_match = re.search(r"User request:\s*(.+)", prompt, re.DOTALL)
        notes = notes_match.group(1).strip() if notes_match else ""
        query = query_match.group(1).strip() if query_match else ""
        bullet_notes = " ".join(line.strip("- ").strip() for line in notes.splitlines() if line.strip())
        return f"Answer to '{query}': {bullet_notes}".strip()

    @staticmethod
    def _mock_critique(prompt: str) -> str:
        # Approve once the draft references the original query, to guarantee
        # the reflection loop always converges within max_iterations in tests.
        if "Answer to '" in prompt:
            return "VERDICT: approve\nFEEDBACK: Draft directly addresses the request."
        return "VERDICT: revise\nFEEDBACK: Draft does not clearly answer the request yet."


def build_provider(settings: Settings | None = None) -> LLMProvider:
    """Factory that builds the configured `LLMProvider`."""
    settings = settings or get_settings()

    if settings.llm_provider == "anthropic":
        return AnthropicProvider(settings.anthropic_api_key or "", settings.anthropic_model)
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings.openai_api_key or "", settings.openai_model)
    if settings.llm_provider == "mock":
        return MockProvider()

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")

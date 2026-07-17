"""Shared state object threaded through every node of the agent graph."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    query: str
    plan: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    draft_answer: str = ""
    critique_feedback: str = ""
    verdict: str = ""  # "approve" | "revise"
    iterations: int = 0
    trace: list[str] = Field(default_factory=list)

    def log(self, message: str) -> None:
        self.trace.append(message)

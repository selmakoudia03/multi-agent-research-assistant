"""Breaks the user's request down into a short list of concrete subtasks."""

from __future__ import annotations

from magent.llm import LLMProvider
from magent.state import AgentState

SYSTEM_PROMPT = (
    "You are a planning agent. Break the following user request into 2-4 short, "
    "concrete subtasks, one per line, each starting with a dash. Prefer subtasks "
    "that are either research questions or computations."
)


class PlannerAgent:
    name = "planner"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        prompt = f"User request:\n{state.query}"
        response = self.llm.complete(prompt, system=SYSTEM_PROMPT)
        subtasks = [
            line.lstrip("-").strip()
            for line in response.splitlines()
            if line.strip().lstrip("-").strip()
        ]
        state.plan = subtasks or [state.query]
        state.log(f"planner: produced {len(state.plan)} subtask(s)")
        return state

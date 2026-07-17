"""Gathers background information for research-flavored subtasks."""

from __future__ import annotations

from magent.state import AgentState
from magent.tools.web_search import SearchTool

_SKIP_KEYWORDS = ("compute", "calculate")


class ResearcherAgent:
    name = "researcher"

    def __init__(self, search_tool: SearchTool) -> None:
        self.search_tool = search_tool

    def __call__(self, state: AgentState) -> AgentState:
        research_subtasks = [
            task
            for task in state.plan
            if not any(keyword in task.lower() for keyword in _SKIP_KEYWORDS)
        ]

        for subtask in research_subtasks:
            hits = self.search_tool.search(subtask, k=2)
            if hits:
                state.notes.append(f"[research: {subtask}] " + " | ".join(hits))
            else:
                state.notes.append(f"[research: {subtask}] no relevant results found")

        state.log(f"researcher: processed {len(research_subtasks)} subtask(s)")
        return state

"""Combines research/compute notes into a single draft answer."""

from __future__ import annotations

from magent.llm import LLMProvider
from magent.state import AgentState

SYSTEM_PROMPT = (
    "Combine the research notes below into one clear, well-structured answer "
    "to the user's request. If earlier feedback is provided, address it directly."
)


class SynthesizerAgent:
    name = "synthesizer"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        notes_block = "\n".join(f"- {note}" for note in state.notes) or "- (no notes gathered)"
        prompt = f"Research notes:\n{notes_block}\n\nUser request:\n{state.query}"
        if state.critique_feedback:
            prompt += f"\n\nPrevious critic feedback to address:\n{state.critique_feedback}"

        state.draft_answer = self.llm.complete(prompt, system=SYSTEM_PROMPT)
        state.iterations += 1
        state.log(f"synthesizer: produced draft (iteration {state.iterations})")
        return state

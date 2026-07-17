"""Reviews the draft answer and decides whether it should be revised.

This is the "reflection" step of the pipeline: it closes a loop back to the
synthesizer with concrete feedback instead of only accepting or rejecting.
"""

from __future__ import annotations

import re

from magent.llm import LLMProvider
from magent.state import AgentState

SYSTEM_PROMPT = (
    "You are a critic agent. Decide if the DRAFT ANSWER fully and correctly "
    "answers the USER REQUEST. Reply with exactly two lines:\n"
    "VERDICT: approve|revise\n"
    "FEEDBACK: <one sentence, empty if approved>"
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(approve|revise)", re.IGNORECASE)
_FEEDBACK_RE = re.compile(r"FEEDBACK:\s*(.*)")


class CriticAgent:
    name = "critic"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        prompt = f"USER REQUEST:\n{state.query}\n\nDRAFT ANSWER:\n{state.draft_answer}"
        response = self.llm.complete(prompt, system=SYSTEM_PROMPT)

        verdict_match = _VERDICT_RE.search(response)
        feedback_match = _FEEDBACK_RE.search(response)

        state.verdict = verdict_match.group(1).lower() if verdict_match else "revise"
        state.critique_feedback = feedback_match.group(1).strip() if feedback_match else ""
        state.log(f"critic: verdict={state.verdict!r}")
        return state

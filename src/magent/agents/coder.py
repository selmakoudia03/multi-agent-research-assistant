"""Handles computation-flavored subtasks with the calculator/REPL tools."""

from __future__ import annotations

import re

from magent.state import AgentState
from magent.tools.calculator import CalculatorError, CalculatorTool
from magent.tools.python_repl import PythonREPLTool

_COMPUTE_KEYWORDS = ("compute", "calculate")
_EXPRESSION_RE = re.compile(r"[0-9][0-9\.\s\+\-\*/\(\)%]*[0-9\)]")


class CoderAgent:
    name = "coder"

    def __init__(self, calculator: CalculatorTool, python_repl: PythonREPLTool) -> None:
        self.calculator = calculator
        self.python_repl = python_repl

    def __call__(self, state: AgentState) -> AgentState:
        compute_subtasks = [
            task for task in state.plan if any(k in task.lower() for k in _COMPUTE_KEYWORDS)
        ]

        for subtask in compute_subtasks:
            source = subtask if any(ch.isdigit() for ch in subtask) else state.query
            match = _EXPRESSION_RE.search(source)
            if not match:
                state.notes.append(f"[compute: {subtask}] no numeric expression found")
                continue

            expression = match.group(0)
            try:
                result = self.calculator.run(expression)
                state.notes.append(f"[compute: {subtask}] {expression} = {result}")
            except CalculatorError:
                repl_result = self.python_repl.run(f"print({expression})")
                if repl_result.ok:
                    state.notes.append(
                        f"[compute: {subtask}] {expression} = {repl_result.stdout.strip()}"
                    )
                else:
                    state.notes.append(f"[compute: {subtask}] could not evaluate {expression!r}")

        state.log(f"coder: processed {len(compute_subtasks)} subtask(s)")
        return state

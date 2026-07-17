"""Lightweight evaluation harness for the agent pipeline.

Runs every case in `eval_cases.jsonl` through the default graph and scores
each answer by whether it contains at least one expected keyword. This is
intentionally simple (no LLM-as-judge, no external eval framework) so it
runs offline in CI with the mock provider, while still giving a concrete,
trackable quality signal for the pipeline -- exactly the kind of guardrail
an AI engineer is expected to wire up before shipping an agent.

Usage:
    python -m evaluation.run_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from magent.pipeline import build_default_graph  # noqa: E402
from magent.state import AgentState  # noqa: E402

CASES_PATH = Path(__file__).parent / "eval_cases.jsonl"


def load_cases() -> list[dict]:
    with CASES_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval() -> float:
    graph = build_default_graph()
    cases = load_cases()
    passed = 0

    for case in cases:
        final_state = graph.run(AgentState(query=case["query"]))
        answer_lower = final_state.draft_answer.lower()
        hit = any(keyword.lower() in answer_lower for keyword in case["expected_keywords"])
        passed += int(hit)
        status = "PASS" if hit else "FAIL"
        print(f"[{status}] {case['query']}")
        print(f"    answer: {final_state.draft_answer[:160]}")

    score = passed / len(cases) if cases else 0.0
    print(f"\nScore: {passed}/{len(cases)} ({score:.0%})")
    return score


if __name__ == "__main__":
    run_eval()

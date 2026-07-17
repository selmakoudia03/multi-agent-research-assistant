"""Run a single query through the default agent pipeline and print the trace.

Usage:
    python examples/run_example.py "What is retrieval-augmented generation?"

Uses the offline mock LLM provider by default. Set MAGENT_LLM_PROVIDER=anthropic
(with ANTHROPIC_API_KEY) or MAGENT_LLM_PROVIDER=openai (with OPENAI_API_KEY) to
run against a real model instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from magent.pipeline import run_query  # noqa: E402


def main() -> None:
    query = " ".join(sys.argv[1:]) or "What is a multi-agent system and how does reflection help it?"
    final_state = run_query(query)

    print(f"Query: {query}\n")
    print("Plan:")
    for step in final_state.plan:
        print(f"  - {step}")

    print("\nTrace:")
    for line in final_state.trace:
        print(f"  - {line}")

    print(f"\nVerdict: {final_state.verdict} (after {final_state.iterations} iteration(s))")
    print(f"\nAnswer:\n{final_state.draft_answer}")


if __name__ == "__main__":
    main()

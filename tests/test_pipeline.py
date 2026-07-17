from magent.pipeline import build_default_graph, run_query
from magent.state import AgentState


def test_pipeline_produces_approved_answer_within_max_iterations() -> None:
    final_state = run_query("What is RAG and how does it use a retriever?")

    assert final_state.plan
    assert final_state.draft_answer
    assert final_state.verdict in {"approve", "revise"}
    assert final_state.iterations >= 1
    assert final_state.iterations <= 3


def test_compute_subtask_reaches_synthesized_answer() -> None:
    final_state = run_query("What is 12 * (3 + 4)?")

    assert any("compute" in note.lower() for note in final_state.notes)
    assert "84" in final_state.draft_answer or "84" in " ".join(final_state.notes)


def test_graph_is_reusable_across_multiple_queries() -> None:
    graph = build_default_graph()

    first = graph.run(AgentState(query="What is MCP?"))
    second = graph.run(AgentState(query="What is fine-tuning?"))

    assert first.query != second.query
    assert first.draft_answer and second.draft_answer

import pytest

from magent.graph import END, Graph, GraphError


def test_linear_graph_runs_to_completion() -> None:
    graph = Graph()
    graph.add_node("increment", lambda x: x + 1)
    graph.add_node("double", lambda x: x * 2)
    graph.set_entry_point("increment")
    graph.add_edge("increment", "double")
    graph.add_edge("double", END)

    assert graph.run(1) == 4  # (1 + 1) * 2


def test_conditional_edge_loops_until_condition_met() -> None:
    graph = Graph()
    graph.add_node("increment", lambda x: x + 1)
    graph.set_entry_point("increment")
    graph.add_conditional_edges(
        "increment",
        lambda x: "done" if x >= 5 else "loop",
        {"done": END, "loop": "increment"},
    )

    assert graph.run(0) == 5


def test_missing_entry_point_raises() -> None:
    graph = Graph()
    with pytest.raises(GraphError):
        graph.run(0)


def test_dead_end_node_raises() -> None:
    graph = Graph()
    graph.add_node("only", lambda x: x)
    graph.set_entry_point("only")
    with pytest.raises(GraphError):
        graph.run(0)


def test_exceeding_max_steps_raises() -> None:
    graph = Graph(max_steps=3)
    graph.add_node("loop", lambda x: x + 1)
    graph.set_entry_point("loop")
    graph.add_edge("loop", "loop")
    with pytest.raises(GraphError):
        graph.run(0)

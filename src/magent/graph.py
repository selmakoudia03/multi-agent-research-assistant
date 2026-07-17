"""A minimal, dependency-free graph orchestration engine.

The public API deliberately mirrors the core concepts of LangGraph
(`add_node`, `add_edge`, `add_conditional_edges`, `END`) so the design is
familiar to anyone who has used it, while keeping this project's only
hard dependency on the *idea* of a state graph, not on a specific
framework version. Swapping this engine for real LangGraph later only
means reimplementing `Graph.run` — every agent function keeps working
unchanged, since it only ever receives and returns an `AgentState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar

State = TypeVar("State")
Node = Callable[[State], State]
Condition = Callable[[State], str]

END = "__end__"


class GraphError(RuntimeError):
    pass


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    conditional_edges: dict[str, tuple[Condition, dict[str, str]]] = field(default_factory=dict)
    entry_point: str | None = None
    max_steps: int = 25

    def add_node(self, name: str, fn: Node) -> "Graph":
        self.nodes[name] = fn
        return self

    def add_edge(self, from_node: str, to_node: str) -> "Graph":
        self.edges[from_node] = to_node
        return self

    def add_conditional_edges(
        self, from_node: str, condition: Condition, mapping: dict[str, str]
    ) -> "Graph":
        self.conditional_edges[from_node] = (condition, mapping)
        return self

    def set_entry_point(self, name: str) -> "Graph":
        self.entry_point = name
        return self

    def _next_node(self, current: str, state: State) -> str:
        if current in self.conditional_edges:
            condition, mapping = self.conditional_edges[current]
            branch = condition(state)
            if branch not in mapping:
                raise GraphError(
                    f"Condition at {current!r} returned {branch!r}, "
                    f"which is not in mapping {list(mapping)}"
                )
            return mapping[branch]
        if current in self.edges:
            return self.edges[current]
        raise GraphError(f"Node {current!r} has no outgoing edge and did not reach END")

    def run(self, state: State) -> State:
        if self.entry_point is None:
            raise GraphError("Graph has no entry point; call set_entry_point() first")

        current = self.entry_point
        for _ in range(self.max_steps):
            if current == END:
                return state
            if current not in self.nodes:
                raise GraphError(f"Unknown node: {current!r}")
            state = self.nodes[current](state)
            current = self._next_node(current, state)

        raise GraphError(f"Graph exceeded max_steps={self.max_steps} without reaching END")

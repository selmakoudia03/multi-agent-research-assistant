"""Wires agents and tools into the default Planner -> Researcher/Coder ->
Synthesizer -> Critic graph, with a reflection loop back to the synthesizer.
"""

from __future__ import annotations

from magent.agents import CoderAgent, CriticAgent, PlannerAgent, ResearcherAgent, SynthesizerAgent
from magent.config import Settings, get_settings
from magent.graph import END, Graph
from magent.llm import LLMProvider, build_provider
from magent.state import AgentState
from magent.tools.calculator import CalculatorTool
from magent.tools.python_repl import PythonREPLTool
from magent.tools.web_search import LocalCorpusSearchTool, SearchTool


def build_default_graph(
    llm: LLMProvider | None = None,
    search_tool: SearchTool | None = None,
    settings: Settings | None = None,
) -> Graph:
    settings = settings or get_settings()
    llm = llm or build_provider(settings)
    search_tool = search_tool or LocalCorpusSearchTool()

    planner = PlannerAgent(llm)
    researcher = ResearcherAgent(search_tool)
    coder = CoderAgent(CalculatorTool(), PythonREPLTool(timeout_seconds=settings.tool_timeout_seconds))
    synthesizer = SynthesizerAgent(llm)
    critic = CriticAgent(llm)

    max_iterations = settings.max_reflection_iterations

    def route_after_critic(state: AgentState) -> str:
        if state.verdict == "approve" or state.iterations >= max_iterations:
            return "done"
        return "revise"

    graph = Graph()
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("coder", coder)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("critic", critic)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "coder")
    graph.add_edge("coder", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"done": END, "revise": "synthesizer"}
    )

    return graph


def run_query(query: str, graph: Graph | None = None) -> AgentState:
    graph = graph or build_default_graph()
    return graph.run(AgentState(query=query))

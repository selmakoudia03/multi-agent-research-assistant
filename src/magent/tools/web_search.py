"""Search tools available to the Researcher agent.

`LocalCorpusSearchTool` indexes a small bundled corpus with `VectorMemory`
so the whole system is demoable and testable with zero network access.
`SearchTool` is the interface a live web-search backend (Bing/Tavily/SerpAPI/
DuckDuckGo, ...) would implement in a production deployment — swapping it
in is a matter of implementing `search()`, nothing else changes.
"""

from __future__ import annotations

from typing import Protocol

from magent.memory import VectorMemory

_DEFAULT_CORPUS = [
    "LangGraph is a library for building stateful, multi-agent LLM applications "
    "as graphs of nodes and edges, supporting cycles and persistence.",
    "Retrieval-Augmented Generation (RAG) combines a retriever over an external "
    "knowledge base with a generator LLM to ground answers in retrieved documents.",
    "A multi-agent system decomposes a task across specialized agents (planner, "
    "researcher, coder, critic) that communicate through a shared state.",
    "Model Context Protocol (MCP) standardizes how LLM applications connect to "
    "external tools, data sources, and services.",
    "Reflection is an agentic pattern where a critic component reviews a draft "
    "output and asks the generator to revise it before returning a final answer.",
    "MLOps covers the practices for deploying, monitoring, and maintaining "
    "machine learning models in production, including CI/CD and observability.",
    "Vector databases index embeddings for approximate nearest-neighbor search, "
    "commonly used as the retrieval backend of a RAG pipeline.",
    "Prompt engineering is the practice of designing inputs to steer an LLM's "
    "output toward a desired format or behavior without changing model weights.",
    "Fine-tuning adapts a pretrained model's weights on a task-specific dataset, "
    "in contrast to prompting, which only changes the input at inference time.",
    "Tool use lets an LLM agent call external functions (search, code execution, "
    "calculators, APIs) and incorporate their results into its reasoning.",
]


class SearchTool(Protocol):
    name: str

    def search(self, query: str, k: int = 3) -> list[str]: ...


class LocalCorpusSearchTool:
    """Offline search tool backed by a small bundled knowledge corpus."""

    name = "web_search"

    def __init__(self, corpus: list[str] | None = None) -> None:
        self._memory = VectorMemory()
        self._memory.add_many(corpus if corpus is not None else _DEFAULT_CORPUS)

    def search(self, query: str, k: int = 3) -> list[str]:
        return [text for text, _score, _meta in self._memory.search(query, k=k)]

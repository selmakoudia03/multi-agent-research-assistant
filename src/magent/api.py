"""FastAPI service exposing the multi-agent pipeline over HTTP."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from magent.pipeline import build_default_graph

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="Planner -> Researcher/Coder -> Synthesizer -> Critic agent pipeline.",
    version="0.1.0",
)

_graph = build_default_graph()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    plan: list[str]
    iterations: int
    verdict: str
    trace: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    from magent.state import AgentState

    final_state = _graph.run(AgentState(query=request.query))
    return ChatResponse(
        answer=final_state.draft_answer,
        plan=final_state.plan,
        iterations=final_state.iterations,
        verdict=final_state.verdict,
        trace=final_state.trace,
    )

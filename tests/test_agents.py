from magent.agents import CriticAgent, PlannerAgent
from magent.llm import MockProvider
from magent.state import AgentState


def test_planner_produces_nonempty_plan() -> None:
    planner = PlannerAgent(MockProvider())
    state = planner(AgentState(query="What is retrieval-augmented generation?"))
    assert state.plan
    assert all(isinstance(task, str) and task for task in state.plan)


def test_planner_adds_compute_subtask_when_query_has_numbers() -> None:
    planner = PlannerAgent(MockProvider())
    state = planner(AgentState(query="What is 3 + 4?"))
    assert any("compute" in task.lower() for task in state.plan)


def test_critic_approves_well_formed_draft() -> None:
    critic = CriticAgent(MockProvider())
    state = AgentState(query="What is RAG?", draft_answer="Answer to 'What is RAG?': it combines retrieval and generation.")
    state = critic(state)
    assert state.verdict == "approve"


def test_critic_requests_revision_for_empty_draft() -> None:
    critic = CriticAgent(MockProvider())
    state = AgentState(query="What is RAG?", draft_answer="")
    state = critic(state)
    assert state.verdict == "revise"
    assert state.critique_feedback

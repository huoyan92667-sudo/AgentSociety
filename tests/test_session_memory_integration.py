from __future__ import annotations

from datetime import datetime

from yelp_agent.agent_benchmark import VisibleAgentScenario
from yelp_agent.agent_harness import AgentHarness, HarnessBudget, UserTurnInput
from yelp_agent.agent_harness.fakes import FakeClock
from yelp_agent.agent_harness.schema import ActionOutcome, AgentDecision
from yelp_agent.session_memory.config import SessionMemoryConfig
from yelp_agent.session_memory.integration import MemoryAwareRequestInterpreter
from yelp_agent.session_memory.manager import SessionMemoryManager


class _Policy:
    def allowed_actions(self, state):
        if state.readiness.information_gaps:
            return ["ask_clarification"]
        return ["return_recommendation"]


class _Router:
    def choose_action(self, state):
        if state.readiness.information_gaps:
            return AgentDecision(
                action="ask_clarification",
                arguments={"information_gaps": state.readiness.information_gaps},
                reason_code="MISSING_INFORMATION",
            )
        return AgentDecision(
            action="return_recommendation",
            arguments={"business_ids": ["b1", "b2", "b3", "b4", "b5"]},
            reason_code="READY",
        )


class _Executor:
    def execute(self, state, decision):
        if decision.action == "ask_clarification":
            from yelp_agent.agent_evaluation.schema import ClarificationQuestionTrace

            return ActionOutcome(
                status="completed",
                response_kind="clarification",
                clarification_question=ClarificationQuestionTrace(
                    question_text="Where are you?",
                    requested_information_gaps=state.readiness.information_gaps,
                ),
            )
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["b1", "b2", "b3", "b4", "b5"],
            recommended_business_ids=["b1", "b2", "b3", "b4", "b5"],
            business_scope=["b1", "b2", "b3", "b4", "b5"],
        )


def _harness() -> AgentHarness:
    manager = SessionMemoryManager(
        config=SessionMemoryConfig(
            memory_version="step34-test",
            enabled=False,
            prompt_version="step34-test",
            cache_relative_path="unused.sqlite3",
        )
    )
    return AgentHarness(
        agent_version="step34-test",
        interpreter=MemoryAwareRequestInterpreter(manager),
        action_policy=_Policy(),
        router=_Router(),
        executor=_Executor(),
        budget=HarnessBudget(max_steps=2),
        clock=FakeClock(),
    )


def test_memory_interpreter_pauses_and_resumes_without_raw_query_concatenation() -> None:
    harness = _harness()
    scenario = VisibleAgentScenario(
        scenario_id="a" * 64,
        split="development",
        language="en-US",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text="Find a steakhouse within 5 km",
    )

    paused = harness.start(scenario)
    resumed = harness.resume(
        paused.session,
        UserTurnInput(
            query_text="I am at Philadelphia City Hall.",
            user_latitude=39.9526,
            user_longitude=-75.1652,
        ),
    )

    assert paused.session.status == "awaiting_user"
    assert paused.session.memory is not None
    assert resumed.session.status == "completed"
    assert resumed.session.memory is not None
    assert resumed.session.memory.revision == 2
    assert resumed.session.memory.current_request.location_center is not None
    assert "Steakhouses" in resumed.session.request.desired_categories
    assert resumed.session.memory.last_presented_business_ids == [
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
    ]
    assert resumed.session.memory_fallback_count == 2
    assert resumed.run is not None
    effective = resumed.run.turns[-1].effective_request
    assert effective is not None
    assert effective["revision"] == 2
    assert "Steakhouses" in effective["request"]["query_text"]

"""End-to-end tests for the assembled Step 24 Rule Agent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import Field

from yelp_agent.agent_benchmark import ScriptedUserTurn, VisibleAgentScenario
from yelp_agent.agent_harness import BenchmarkSessionDriver, FakeClock, UserTurnInput
from yelp_agent.agent_tools import (
    AgentToolRegistry,
    ToolDefinition,
    ToolExecutionContext,
    ToolObservation,
)
from yelp_agent.models import StrictModel
from yelp_agent.rule_router import RuleAgentSourcePaths, build_rule_agent

PROJECT_ROOT = Path(__file__).parents[1]


class _Empty(StrictModel):
    pass


class _Ids(StrictModel):
    business_ids: list[str] = Field(min_length=1)


class _Candidates(StrictModel):
    candidate_business_ids: list[str]


class _Ranking(StrictModel):
    ranking: list[str]


class _Businesses(StrictModel):
    businesses: list[dict[str, object]]


class _SemanticMatches(StrictModel):
    matches: list[dict[str, object]]


class _SessionMemory(StrictModel):
    session_id: str
    turn_index: int
    observations: list[dict[str, object]]


class _Tool:
    def __init__(
        self,
        *,
        name: str,
        action: str,
        input_model: type[StrictModel],
        output_model: type[StrictModel],
        data: dict[str, object],
        kind: str = "deterministic",
    ) -> None:
        self.definition = ToolDefinition(
            name=name,
            version="test-v1",
            kind=kind,  # type: ignore[arg-type]
            allowed_actions=(action,),
            input_model=input_model,
            output_model=output_model,
            public_summary=f"Fake {name} for the complete Rule Agent path.",
        )
        self._data = data

    def run(
        self,
        arguments: StrictModel,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        del context
        data = self._data
        if self.definition.name == "GET_HYBRID_RANKING" and isinstance(
            arguments, _Ids
        ):
            preferred = ["b2", "b1", "b3"]
            data = {
                "ranking": [
                    business_id
                    for business_id in preferred
                    if business_id in arguments.business_ids
                ]
            }
        return ToolObservation.success(
            tool_name=self.definition.name,
            data=data,
            confidence=1.0,
        )


def _registry(*, semantic: bool = False) -> AgentToolRegistry:
    ids = ["b1", "b2", "b3"]
    tools = [
            _Tool(
                name="GET_SESSION_MEMORY",
                action="apply_feedback",
                input_model=_Empty,
                output_model=_SessionMemory,
                data={
                    "session_id": "session-1",
                    "turn_index": 2,
                    "observations": [],
                },
            ),
            _Tool(
                name="EXPAND_CANDIDATES",
                action="retrieve_candidates",
                input_model=_Empty,
                output_model=_Candidates,
                data={"candidate_business_ids": ids},
            ),
            _Tool(
                name="APPLY_CONSTRAINTS",
                action="apply_hard_constraints",
                input_model=_Ids,
                output_model=_Candidates,
                data={"candidate_business_ids": ids},
            ),
            _Tool(
                name="GET_HYBRID_RANKING",
                action="rank_candidates",
                input_model=_Ids,
                output_model=_Ranking,
                data={"ranking": ["b2", "b1", "b3"]},
            ),
            _Tool(
                name="GET_BUSINESS_DETAILS",
                action="get_business_details",
                input_model=_Ids,
                output_model=_Businesses,
                data={
                    "businesses": [
                        {
                            "business_id": business_id,
                            "name": business_id,
                            "categories": ["Steakhouses"],
                        }
                        for business_id in ("b2", "b1", "b3")
                    ]
                },
            ),
        ]
    if semantic:
        tools.append(
            _Tool(
                name="COMPUTE_EMBEDDING_MATCH",
                action="rank_candidates",
                input_model=_Ids,
                output_model=_SemanticMatches,
                kind="semantic",
                data={
                    "matches": [
                        {"business_id": "b3", "semantic_rank": 1},
                        {"business_id": "b1", "semantic_rank": 2},
                        {"business_id": "b2", "semantic_rank": 3},
                    ]
                },
            )
        )
    return AgentToolRegistry(tools)


def test_assembled_rule_agent_completes_the_recommendation_chain() -> None:
    harness = build_rule_agent(registry=_registry(), clock=FakeClock())
    scenario = VisibleAgentScenario(
        scenario_id="d" * 64,
        split="development",
        language="en-US",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text="I only want a steakhouse",
    )

    result = harness.start(scenario)

    assert result.run is not None
    assert result.run.fallback is False
    assert result.session.status == "completed"
    assert [item.action for item in result.run.turns[0].actions] == [
        "retrieve_candidates",
        "apply_hard_constraints",
        "rank_candidates",
        "get_business_details",
        "return_recommendation",
    ]
    assert result.run.turns[0].candidate_ranking == ["b2", "b1", "b3"]
    assert result.run.turns[0].recommended_business_ids == ["b2", "b1", "b3"]


def test_step25_agent_calls_semantic_tool_and_returns_fused_ranking() -> None:
    harness = build_rule_agent(
        registry=_registry(semantic=True),
        clock=FakeClock(),
        semantic_enabled=True,
        semantic_candidate_limit=3,
        fusion_alpha=0.6,
        agent_version="step25-test",
    )
    scenario = VisibleAgentScenario(
        scenario_id="8" * 64,
        split="development",
        language="en-US",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text="I only want a steakhouse",
    )

    result = harness.start(scenario)

    assert result.run is not None
    assert result.run.fallback is False
    assert [call.tool_name for call in result.run.turns[0].tool_calls] == [
        "EXPAND_CANDIDATES",
        "APPLY_CONSTRAINTS",
        "GET_HYBRID_RANKING",
        "COMPUTE_EMBEDDING_MATCH",
        "GET_BUSINESS_DETAILS",
    ]
    assert result.run.turns[0].candidate_ranking == ["b3", "b1", "b2"]
    assert result.run.turns[0].recommended_business_ids == ["b3", "b1", "b2"]


def test_assembled_rule_agent_pauses_and_resumes_after_location_reply() -> None:
    harness = build_rule_agent(registry=_registry(), clock=FakeClock())
    scenario = VisibleAgentScenario(
        scenario_id="e" * 64,
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
            query_text="I am near Philadelphia City Hall.",
        ),
    )

    assert paused.session.status == "awaiting_user"
    assert paused.run is None
    assert resumed.run is not None
    assert resumed.run.fallback is False
    assert len(resumed.run.turns) == 2
    assert resumed.run.turns[0].actions[0].action == "ask_clarification"
    assert resumed.run.turns[1].response_kind == "recommendation"


def test_answered_budget_clarification_is_not_asked_twice() -> None:
    harness = build_rule_agent(registry=_registry(), clock=FakeClock())
    scenario = VisibleAgentScenario(
        scenario_id="9" * 64,
        split="development",
        language="en-US",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text="Find a steakhouse, per person under $80",
    )

    paused = harness.start(scenario)
    resumed = harness.resume(
        paused.session,
        UserTurnInput(query_text="No more than $40 per person."),
    )

    assert paused.session.status == "awaiting_user"
    assert resumed.run is not None
    assert resumed.run.fallback is False
    assert "missing_budget" not in resumed.session.readiness.information_gaps
    assert resumed.run.turns[1].response_kind == "recommendation"


def test_assembled_rule_agent_applies_feedback_after_its_first_recommendation() -> None:
    harness = build_rule_agent(registry=_registry(), clock=FakeClock())
    scenario = VisibleAgentScenario(
        scenario_id="f" * 64,
        split="development",
        language="en-US",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text="Recommend a steakhouse",
    )
    feedback = ScriptedUserTurn(
        turn_index=2,
        trigger_action="return_recommendation",
        query_text="That is too far. Please choose something else.",
        expected_task_type="feedback_refinement",
        expected_information_gaps=[],
        added_conditions=[],
        state_updates={},
        rejected_business_ids=["b2"],
        expected_allowed_actions=[
            "apply_feedback",
            "rank_candidates",
            "return_recommendation",
        ],
    )

    driven = BenchmarkSessionDriver().drive(harness, scenario, [feedback])

    assert driven.stop_reason == "completed"
    assert driven.result.run is not None
    assert [action.action for action in driven.result.run.turns[1].actions] == [
        "apply_feedback",
        "rank_candidates",
        "return_recommendation",
    ]


def test_real_rule_agent_source_bundle_resolves_every_frozen_artifact() -> None:
    sources = RuleAgentSourcePaths.from_project_root(PROJECT_ROOT)

    missing = [str(path) for path in sources.required_files() if not path.is_file()]
    assert missing == []
    assert sources.user_profile_root.name == "v1"
    assert sources.business_profile_root.name == "v1"
    assert sources.hybrid_v2_root.name == "frozen"

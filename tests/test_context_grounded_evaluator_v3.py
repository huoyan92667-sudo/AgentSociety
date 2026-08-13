from __future__ import annotations

from yelp_agent.session_memory_benchmark.contextual_evaluation import (
    evaluation_candidate_scope,
    rebind_behavior_expectations,
)
from yelp_agent.session_memory_benchmark.schema import (
    BehaviorExpectation,
    ExpectedMemoryDeltaV2,
    FrozenPresentation,
    FrozenTurnGroundTruthV2,
    PresentedBusinessSnapshot,
)


def _business(business_id: str, rank: int, price: int) -> PresentedBusinessSnapshot:
    return PresentedBusinessSnapshot(
        business_id=business_id,
        rank=rank,
        name=business_id,
        categories=["Restaurants"],
        price_level=price,
    )


def test_relative_answer_is_rebound_to_the_business_the_agent_actually_showed() -> None:
    frozen = FrozenPresentation(
        session_case_id="a" * 64,
        turn_index=1,
        pipeline_version="old-ranking",
        source_run_sha256="b" * 64,
        candidate_business_ids=["frozen-first", "frozen-cheaper"],
        presented_businesses=[
            _business("frozen-first", 1, 4),
            _business("frozen-cheaper", 2, 2),
        ],
    )
    truth = FrozenTurnGroundTruthV2(
        turn_case_id="c" * 64,
        session_case_id=frozen.session_case_id,
        split="development",
        turn_index=2,
        family="relative_preference",
        expected_delta=ExpectedMemoryDeltaV2(task_type="feedback_refinement"),
        behaviors=[
            BehaviorExpectation(
                kind="cheaper",
                baseline_business_id="frozen-first",
                baseline_value=4,
                acceptable_business_ids=["frozen-cheaper"],
            )
        ],
    )
    actual_candidates = [
        _business("actual-first", 1, 2),
        _business("actual-cheaper", 2, 1),
        _business("actual-expensive", 3, 3),
    ]

    rebound = rebind_behavior_expectations(
        truth,
        frozen_presentation=frozen,
        previous_presented_business_ids=["actual-first"],
        candidate_snapshots=actual_candidates,
    )

    assert rebound.context_status == "rebound"
    assert rebound.behaviors[0].baseline_business_id == "actual-first"
    assert rebound.behaviors[0].baseline_value == 2
    assert rebound.behaviors[0].acceptable_business_ids == ["actual-cheaper"]


def test_rejection_is_rebound_to_the_same_ordinal_in_the_actual_list() -> None:
    frozen = FrozenPresentation(
        session_case_id="d" * 64,
        turn_index=1,
        pipeline_version="old-ranking",
        source_run_sha256="e" * 64,
        candidate_business_ids=["old-first", "old-second"],
        presented_businesses=[
            _business("old-first", 1, 2),
            _business("old-second", 2, 3),
        ],
    )
    truth = FrozenTurnGroundTruthV2(
        turn_case_id="f" * 64,
        session_case_id=frozen.session_case_id,
        split="development",
        turn_index=2,
        family="reference_rejection",
        expected_delta=ExpectedMemoryDeltaV2(task_type="feedback_refinement"),
        behaviors=[
            BehaviorExpectation(
                kind="constraint_satisfaction",
                excluded_business_ids=["old-second"],
            )
        ],
    )

    rebound = rebind_behavior_expectations(
        truth,
        frozen_presentation=frozen,
        previous_presented_business_ids=["actual-first", "actual-second"],
        candidate_snapshots=[
            _business("actual-first", 1, 2),
            _business("actual-second", 2, 3),
        ],
    )

    assert rebound.context_status == "rebound"
    assert rebound.behaviors[0].excluded_business_ids == ["actual-second"]


def test_no_recommendation_keeps_the_last_real_candidate_scope_for_scoring() -> None:
    scope, source = evaluation_candidate_scope([], ["candidate-1", "candidate-2"])

    assert scope == ["candidate-1", "candidate-2"]
    assert source == "carried_forward"

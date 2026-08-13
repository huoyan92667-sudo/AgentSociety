"""Deterministically turn frozen first-turn contexts into provider phrasing jobs."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from .config import SessionMemoryBenchmarkV2Config
from .dynamics import derive_relative_behavior
from .schema import (
    BenchmarkGenerationPlan,
    BehaviorExpectation,
    ExpectedConditionDelta,
    ExpectedMemoryDeltaV2,
    ExpectedRelativePreference,
    MemoryBenchmarkInitialSession,
    PlanningContext,
    PresentedBusinessSnapshot,
    TurnFamily,
    TurnGenerationSpec,
)


class BenchmarkV2Planner:
    """Own labels and distribution; providers are allowed to phrase them only."""

    def __init__(self, config: SessionMemoryBenchmarkV2Config) -> None:
        self._config = config
        self._rng = random.Random(config.seed)

    def plan(self, contexts: Sequence[PlanningContext]) -> BenchmarkGenerationPlan:
        grouped: dict[tuple[str, str], list[PlanningContext]] = defaultdict(list)
        for item in contexts:
            grouped[(item.initial_session.split, item.initial_session.language)].append(item)
        sessions: list[MemoryBenchmarkInitialSession] = []
        presentations = []
        specs: list[TurnGenerationSpec] = []
        used_source_ids: set[str] = set()
        for split, split_plan in self._config.split_plans.items():
            for language, language_plan in split_plan.languages.items():
                pool = sorted(
                    grouped[(split, language)],
                    key=lambda item: _stable_id(
                        str(self._config.seed), item.initial_session.session_case_id
                    ),
                )
                recommendation_pool = [
                    item
                    for item in pool
                    if item.presentation is not None
                    and len(item.presentation.presented_businesses) >= 2
                ]
                if len(recommendation_pool) < language_plan.recommendation_sessions:
                    raise ValueError(
                        f"insufficient recommendation contexts for {split}/{language}"
                    )
                families = _expanded(language_plan.recommendation_families)
                for context, chunk in zip(
                    recommendation_pool[: language_plan.recommendation_sessions],
                    _chunks(families, 3),
                    strict=True,
                ):
                    used_source_ids.add(context.initial_session.source_scenario_id)
                    sessions.append(context.initial_session)
                    assert context.presentation is not None
                    presentations.append(context.presentation)
                    prior_intents: list[str] = []
                    for offset, family in enumerate(chunk, start=2):
                        spec = self._spec(context, family, offset)
                        spec = spec.model_copy(
                            update={
                                "visible_context": {
                                    **spec.visible_context,
                                    "prior_planned_intents": list(prior_intents),
                                    "style_nonce": spec.turn_case_id[:8],
                                }
                            }
                        )
                        specs.append(spec)
                        prior_intents.append(spec.intent_code)

                support_families = _expanded(language_plan.support_families)
                support_pool = [
                    item
                    for item in pool
                    if item.initial_session.source_scenario_id not in used_source_ids
                ]
                for support_index, family in enumerate(support_families):
                    eligible = _support_pool(support_pool, family)
                    if not eligible:
                        raise ValueError(
                            f"insufficient {family} contexts for {split}/{language}"
                        )
                    context = eligible[support_index % len(eligible)]
                    cloned = _clone_session(context.initial_session, family, support_index)
                    cloned_context = context.model_copy(
                        update={"initial_session": cloned, "presentation": None}
                    )
                    sessions.append(cloned)
                    spec = self._spec(cloned_context, family, 2)
                    specs.append(
                        spec.model_copy(
                            update={
                                "visible_context": {
                                    **spec.visible_context,
                                    "style_nonce": spec.turn_case_id[:8],
                                }
                            }
                        )
                    )
        ordered_specs = sorted(
            specs, key=lambda item: (item.session_case_id, item.turn_index)
        )
        ordered_specs = [
            item.model_copy(
                update={
                    "visible_context": {
                        **item.visible_context,
                        "style_index": index,
                    }
                }
            )
            for index, item in enumerate(ordered_specs)
        ]
        return BenchmarkGenerationPlan(
            initial_sessions=sorted(sessions, key=lambda item: item.session_case_id),
            presentations=sorted(presentations, key=lambda item: item.session_case_id),
            turn_specs=ordered_specs,
        )

    def _spec(
        self,
        context: PlanningContext,
        family: TurnFamily,
        turn_index: int,
    ) -> TurnGenerationSpec:
        case_id = _stable_id(
            self._config.benchmark_version,
            context.initial_session.session_case_id,
            str(turn_index),
            family,
        )
        expected, behaviors, intent_code, required, forbidden = _intent(
            context, family, turn_index
        )
        presentation = context.presentation
        visible_businesses = [] if presentation is None else [
            item.model_dump(mode="json") for item in presentation.presented_businesses
        ]
        state_updates: dict[str, str | int | float | bool] = {}
        if intent_code == "answer_missing_location":
            state_updates = {
                "user_latitude": 39.9526,
                "user_longitude": -75.1652,
            }
        return TurnGenerationSpec(
            turn_case_id=case_id,
            session_case_id=context.initial_session.session_case_id,
            split=context.initial_session.split,
            turn_index=turn_index,
            language=context.initial_session.language,
            family=family,
            intent_code=intent_code,
            visible_context={
                "initial_query": context.initial_session.query_text,
                "presented_businesses": visible_businesses,
                "prior_turn_number": turn_index - 1,
            },
            required_meaning=required,
            forbidden_meaning=forbidden,
            expected_delta=expected,
            behaviors=behaviors,
            trigger_action=(
                "ask_clarification"
                if family in {"clarification_answer", "conflict_resolution"}
                else "return_recommendation"
            ),
            state_updates=state_updates,
        )


def _intent(
    context: PlanningContext,
    family: TurnFamily,
    turn_index: int,
) -> tuple[
    ExpectedMemoryDeltaV2,
    list[BehaviorExpectation],
    str,
    dict[str, object],
    list[str],
]:
    shown = [] if context.presentation is None else context.presentation.presented_businesses
    candidates = context.candidate_businesses
    if family == "relative_preference":
        kind, reference, behavior = _relative_example(shown, candidates)
        direction = {"cheaper": "lower", "closer": "closer", "quieter": "quieter"}[kind]
        field = {"cheaper": "price", "closer": "distance", "quieter": "noise"}[kind]
        expected = ExpectedMemoryDeltaV2(
            task_type="feedback_refinement",
            relative_preferences=[ExpectedRelativePreference(field=field, direction=direction)],
            resolved_business_ids=[reference.business_id],
        )
        required = {
            "relative_preference": {"field": field, "direction": direction},
            "reference_rank": reference.rank,
            "reference_name": reference.name,
        }
        return expected, [behavior], f"relative_{kind}", required, ["exact numeric threshold"]
    if family == "reference_rejection":
        reference = shown[(turn_index - 2) % len(shown)]
        expected = ExpectedMemoryDeltaV2(
            task_type="feedback_refinement",
            rejected_business_ids=[reference.business_id],
            resolved_business_ids=[reference.business_id],
        )
        behavior = BehaviorExpectation(
            kind="constraint_satisfaction",
            excluded_business_ids=[reference.business_id],
        )
        return (
            expected,
            [behavior],
            "reject_presented_business",
            {"reject_rank": reference.rank, "reject_name": reference.name},
            ["reject any other business"],
        )
    if family == "explicit_condition":
        variant = turn_index % 4
        if variant == 0:
            delta = ExpectedConditionDelta(
                operation="add", field="budget_per_person",
                operator="less_than_or_equal", value=35, importance="mandatory"
            )
            code, meaning = "add_budget_cap", {"budget_per_person_max": 35}
        elif variant == 1:
            delta = ExpectedConditionDelta(
                operation="add", field="distance_km",
                operator="less_than_or_equal", value=5, importance="mandatory"
            )
            code, meaning = "add_distance_cap", {"distance_km_max": 5}
        elif variant == 2:
            delta = ExpectedConditionDelta(
                operation="add", field="quiet_environment",
                operator="prefer", value=True, importance="preferred"
            )
            code, meaning = "prefer_quiet", {"quiet_environment": True}
        else:
            delta = ExpectedConditionDelta(
                operation="add", field="price_level",
                operator="less_than_or_equal", value=2, importance="strong"
            )
            code, meaning = "add_price_level_cap", {"price_level_max": 2}
        return (
            ExpectedMemoryDeltaV2(task_type="feedback_refinement", condition_deltas=[delta]),
            [_behavior_for_condition(delta, candidates)], code, meaning, []
        )
    if family == "combined_update":
        kind, reference, relative_behavior = _relative_example(shown, candidates)
        direction = {"cheaper": "lower", "closer": "closer", "quieter": "quieter"}[kind]
        field = {"cheaper": "price", "closer": "distance", "quieter": "noise"}[kind]
        quiet = ExpectedConditionDelta(
            operation="add", field="quiet_environment", operator="prefer",
            value=True, importance="preferred"
        )
        return (
            ExpectedMemoryDeltaV2(
                task_type="feedback_refinement",
                condition_deltas=[quiet],
                relative_preferences=[ExpectedRelativePreference(field=field, direction=direction)],
                resolved_business_ids=[reference.business_id],
            ),
            [relative_behavior, _behavior_for_condition(quiet, candidates)],
            f"combined_{kind}_and_quiet",
            {"relative_field": field, "direction": direction, "quiet": True,
             "reference_rank": reference.rank},
            ["exact numeric threshold"],
        )
    if family == "clarification_answer":
        gap = context.initial_information_gaps[0] if context.initial_information_gaps else "missing_party_size"
        value: str | int | float | bool = {
            "missing_location": "Philadelphia City Hall",
            "missing_budget": 40,
            "missing_party_size": 4,
            "ambiguous_reference": "the first recommendation",
            "constraint_conflict": "keep the budget and relax the category",
        }[gap]
        return (
            ExpectedMemoryDeltaV2(
                task_type=context.initial_task_type,
                clarification_answers={gap: value},
                party_size=value if gap == "missing_party_size" else None,  # type: ignore[arg-type]
            ),
            [], f"answer_{gap}", {"answer_gap": gap, "answer_value": value}, []
        )
    if family == "conflict_resolution":
        included = next(
            (
                item
                for item in context.initial_conditions
                if item.field == "category" and item.operator == "includes"
            ),
            ExpectedConditionDelta(
                operation="replace",
                field="category",
                operator="includes",
                value="Bars",
                importance="mandatory",
            ),
        )
        replacement = included.model_copy(update={"operation": "replace"})
        return (
            ExpectedMemoryDeltaV2(
                task_type="recommendation_request",
                condition_deltas=[replacement],
                conflict_resolutions=["keep_included_category_remove_exclusion"],
            ),
            [], "resolve_category_include_exclude_conflict",
            {
                "resolution": "keep the included category and remove its exclusion",
                "category": replacement.value,
            }, []
        )
    return (
        ExpectedMemoryDeltaV2(task_type="business_detail_question", no_state_change=True),
        [BehaviorExpectation(kind="none")], "ask_explanation_only",
        {
            "request": (
                "explain why the first option was recommended"
                if shown
                else "ask a factual follow-up about the current answer without adding a preference"
            )
        },
        ["new preference", "new hard constraint", "business rejection"],
    )


def _relative_example(
    shown: Sequence[PresentedBusinessSnapshot],
    candidates: Sequence[PresentedBusinessSnapshot],
) -> tuple[str, PresentedBusinessSnapshot, BehaviorExpectation]:
    for kind in ("cheaper", "closer", "quieter"):
        for reference in shown:
            try:
                return kind, reference, derive_relative_behavior(kind, reference, candidates)
            except ValueError:
                continue
    raise ValueError("context cannot support a grounded relative-preference turn")


def _behavior_for_condition(
    delta: ExpectedConditionDelta,
    candidates: Sequence[PresentedBusinessSnapshot],
) -> BehaviorExpectation:
    acceptable: list[str] = []
    if delta.field == "price_level":
        acceptable = [item.business_id for item in candidates if item.price_level is not None and item.price_level <= int(delta.value)]
    elif delta.field == "distance_km":
        acceptable = [item.business_id for item in candidates if item.distance_km is not None and item.distance_km <= float(delta.value)]
    elif delta.field == "quiet_environment":
        acceptable = [item.business_id for item in candidates if item.noise_level == "quiet"]
    return BehaviorExpectation(
        kind="constraint_satisfaction",
        acceptable_business_ids=acceptable,
        maximum_value=(delta.value if delta.field in {"price_level", "distance_km"} else None),
    )


def _expanded(values: dict[TurnFamily, int]) -> list[TurnFamily]:
    return [family for family, count in values.items() for _ in range(count)]


def _chunks(values: Sequence[TurnFamily], size: int) -> Iterable[list[TurnFamily]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _support_pool(
    contexts: Sequence[PlanningContext], family: TurnFamily
) -> list[PlanningContext]:
    if family == "clarification_answer":
        values = [
            item
            for item in contexts
            if set(item.initial_information_gaps)
            & {"missing_location", "missing_budget", "missing_party_size"}
        ]
        return values or list(contexts)
    if family == "conflict_resolution":
        values = [
            item
            for item in contexts
            if "constraint_conflict" in item.initial_information_gaps
        ]
        return values or list(contexts)
    return list(contexts)


def _clone_session(
    value: MemoryBenchmarkInitialSession, family: TurnFamily, variant: int
) -> MemoryBenchmarkInitialSession:
    session_case_id = _stable_id(value.session_case_id, "support", family, str(variant))
    return value.model_copy(update={
        "session_case_id": session_case_id,
        "session_id": f"memory-v2:{session_case_id[:16]}",
    })


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()

"""Deep evaluation module for one set of frozen Agent scenario runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log2

from yelp_agent.agent_benchmark.schema import (
    EvidenceLabel,
    ScenarioGroundTruth,
    VisibleAgentScenario,
)

from .definitions import metric_definitions
from .schema import (
    AgentEvaluationSlice,
    AgentEvaluationReport,
    AgentScenarioRun,
    MetricScore,
    ScenarioEvaluationResult,
)


def _evidence_key(value: object) -> tuple[str, str, str]:
    source_type = str(getattr(value, "source_type"))
    identifier = (
        getattr(value, "review_id", None)
        if source_type == "review"
        else getattr(value, "source_field", None)
    )
    return (str(getattr(value, "business_id")), source_type, str(identifier))


def _ratio(numerator: float, denominator: float) -> MetricScore:
    if denominator == 0:
        return MetricScore(
            status="not_applicable",
            value=None,
            numerator=0,
            denominator=0,
            unit="ratio",
            reason="no_applicable_scenarios",
        )
    return MetricScore(
        status="measured",
        value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        unit="ratio",
    )


def _unavailable(unit: str, reason: str) -> MetricScore:
    return MetricScore(
        status="unavailable",
        value=None,
        numerator=0,
        denominator=0,
        unit=unit,
        reason=reason,
    )


def _mean_score(values: Sequence[float], *, unit: str) -> MetricScore:
    if not values:
        return _unavailable(unit, "no_observations")
    return MetricScore(
        status="measured",
        value=sum(values) / len(values),
        numerator=sum(values),
        denominator=len(values),
        unit=unit,
    )


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _percentile_score(
    values: Sequence[float],
    *,
    quantile: float,
    unit: str,
) -> MetricScore:
    if not values:
        return _unavailable(unit, "no_observations")
    value = _linear_percentile(values, quantile)
    return MetricScore(
        status="measured",
        value=value,
        numerator=value,
        denominator=1,
        unit=unit,
    )


def _ranking_metrics(
    ranking: Sequence[str],
    relevant: set[str],
) -> dict[str, float | None]:
    names = {
        "hr_at_1",
        "hr_at_3",
        "hr_at_5",
        "mrr",
        "ndcg_at_5",
        "recall_at_50",
        "recall_at_100",
        "recall_at_500",
    }
    if not relevant:
        return {name: None for name in names}
    positions = [
        index
        for index, business_id in enumerate(ranking, start=1)
        if business_id in relevant
    ]
    result: dict[str, float | None] = {}
    for cutoff in (1, 3, 5):
        result[f"hr_at_{cutoff}"] = float(
            any(position <= cutoff for position in positions)
        )
    result["mrr"] = 1.0 / min(positions) if positions else 0.0
    dcg = sum(
        1.0 / log2(position + 1)
        for position in positions
        if position <= 5
    )
    ideal_count = min(len(relevant), 5)
    ideal_dcg = sum(1.0 / log2(position + 1) for position in range(1, ideal_count + 1))
    result["ndcg_at_5"] = dcg / ideal_dcg
    for cutoff in (50, 100, 500):
        result[f"recall_at_{cutoff}"] = None
    return result


def _evaluate_agent_scenario_runs(
    runs: Sequence[AgentScenarioRun],
    *,
    visible_scenarios: Sequence[VisibleAgentScenario],
    ground_truth: Sequence[ScenarioGroundTruth],
    evidence_labels: Sequence[EvidenceLabel],
    include_breakdowns: bool,
) -> AgentEvaluationReport:
    """Evaluate observable runs after joining them to evaluator-only labels."""

    visible = {item.scenario_id: item for item in visible_scenarios}
    hidden = {item.scenario_id: item for item in ground_truth}
    run_map = {item.scenario_id: item for item in runs}
    if len(run_map) != len(runs):
        raise ValueError("Agent scenario runs must have unique scenario IDs")
    if set(visible) != set(hidden) or set(run_map) != set(hidden):
        raise ValueError("visible scenarios, hidden answers, and runs must align")
    if any(item.scenario_id not in hidden for item in evidence_labels):
        raise ValueError("evidence labels contain an unknown scenario ID")
    labels_by_scenario: dict[str, list[EvidenceLabel]] = {}
    for label in evidence_labels:
        labels_by_scenario.setdefault(label.scenario_id, []).append(label)

    totals: Counter[str] = Counter()
    denominators: Counter[str] = Counter()
    micro_numerators: Counter[str] = Counter()
    micro_denominators: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    scenario_results: list[ScenarioEvaluationResult] = []
    latencies: list[float] = []
    token_totals: list[float] = []
    recommendation_costs: list[float] = []
    recommendation_cost_complete = True
    successful_recommendations = 0
    cache_hits = 0
    cache_calls = 0
    for scenario_id in sorted(run_map):
        run = run_map[scenario_id]
        truth = hidden[scenario_id]
        visible_item = visible[scenario_id]
        turn = run.turns[0]
        violations: list[str] = []
        turns_by_index = {item.turn_index: item for item in run.turns}
        scripted_by_index = {
            item.turn_index: item for item in truth.scripted_user_turns
        }
        for turn_index, actual_turn in sorted(turns_by_index.items()):
            if turn_index == 1:
                continue
            scripted = scripted_by_index.get(turn_index)
            if scripted is None:
                violations.append(f"unexpected_user_turn:{turn_index}")
                continue
            previous = turns_by_index[turn_index - 1]
            trigger_completed = any(
                action.action == scripted.trigger_action
                and action.status == "completed"
                for action in previous.actions
            )
            if not trigger_completed:
                violations.append(
                    f"scripted_turn_released_without_trigger:{turn_index}"
                )
        for turn_index, scripted in sorted(scripted_by_index.items()):
            previous = turns_by_index.get(turn_index - 1)
            if previous is None:
                continue
            trigger_completed = any(
                action.action == scripted.trigger_action
                and action.status == "completed"
                for action in previous.actions
            )
            if trigger_completed and turn_index not in turns_by_index:
                violations.append(f"missing_scripted_turn_after_trigger:{turn_index}")
        completed_actions = {
            item.action for item in turn.actions if item.status == "completed"
        }
        task_match = float(turn.predicted_task_type == truth.task_type)
        required = set(truth.required_actions)
        action_accuracy = (
            len(completed_actions.intersection(required)) / len(required)
            if required
            else 1.0
        )
        relevant = set(truth.acceptable_business_ids)
        ranking_turn = next(
            (
                item
                for item in reversed(run.turns)
                if item.candidate_ranking or item.recommended_business_ids
            ),
            turn,
        )
        ranking = ranking_turn.candidate_ranking
        has_hard_constraints = any(
            item.enforcement == "filter" for item in truth.expected_conditions
        ) or truth.scenario_category in {"hard_constraint", "profile_conflict"}
        hard_satisfaction = None
        if has_hard_constraints:
            shown = ranking_turn.recommended_business_ids
            hard_satisfaction = (
                sum(item in relevant for item in shown) / len(shown) if shown else 0.0
            )
        all_ranked_ids = [
            business_id
            for actual_turn in run.turns
            for business_id in actual_turn.candidate_ranking
        ]
        valid_candidate_rate = (
            sum(item in set(truth.business_scope) for item in all_ranked_ids)
            / len(all_ranked_ids)
            if all_ranked_ids
            else None
        )
        empty_result = (
            float(not ranking_turn.recommended_business_ids)
            if truth.task_type == "recommendation_request" and truth.business_scope
            else None
        )
        values = {
            "task_type_accuracy": task_match,
            "action_accuracy": action_accuracy,
            "hard_constraint_satisfaction": hard_satisfaction,
            "valid_candidate_rate": valid_candidate_rate,
            "empty_result_rate": empty_result,
        }
        values.update(_ranking_metrics(ranking, relevant))

        gap_true_positive = 0
        gap_false_positive = 0
        gap_false_negative = 0
        question_count = 0
        unnecessary_questions = 0
        answerable_questions = 0
        task_type_correct = 0
        task_type_count = 0
        for actual_turn in run.turns:
            if actual_turn.turn_index == 1:
                expected_gaps = set(truth.expected_information_gaps)
                expected_task_type = truth.task_type
                scripted_reply = next(
                    (
                        item
                        for item in truth.scripted_user_turns
                        if item.turn_index == 2
                        and item.trigger_action == "ask_clarification"
                    ),
                    None,
                )
            else:
                scripted = next(
                    (
                        item
                        for item in truth.scripted_user_turns
                        if item.turn_index == actual_turn.turn_index
                    ),
                    None,
                )
                expected_gaps = (
                    set(scripted.expected_information_gaps) if scripted else set()
                )
                expected_task_type = scripted.expected_task_type if scripted else None
                scripted_reply = next(
                    (
                        item
                        for item in truth.scripted_user_turns
                        if item.turn_index == actual_turn.turn_index + 1
                        and item.trigger_action == "ask_clarification"
                    ),
                    None,
                )
            actual_gaps = set(actual_turn.detected_information_gaps)
            task_type_count += 1
            task_type_correct += int(
                expected_task_type is not None
                and actual_turn.predicted_task_type == expected_task_type
            )
            gap_true_positive += len(actual_gaps.intersection(expected_gaps))
            gap_false_positive += len(actual_gaps.difference(expected_gaps))
            gap_false_negative += len(expected_gaps.difference(actual_gaps))
            for question in actual_turn.clarification_questions:
                requested = set(question.requested_information_gaps)
                question_count += 1
                unnecessary_questions += int(
                    not requested or not requested.issubset(expected_gaps)
                )
                answerable_questions += int(
                    bool(requested)
                    and requested.issubset(expected_gaps)
                    and scripted_reply is not None
                )
        precision_denominator = gap_true_positive + gap_false_positive
        recall_denominator = gap_true_positive + gap_false_negative
        values["missing_field_detection_precision"] = (
            gap_true_positive / precision_denominator
            if precision_denominator
            else None
        )
        values["missing_field_detection_recall"] = (
            gap_true_positive / recall_denominator if recall_denominator else None
        )
        values["unnecessary_question_rate"] = (
            unnecessary_questions / question_count if question_count else None
        )
        values["question_answerability_rate"] = (
            answerable_questions / question_count if question_count else None
        )
        terminal_actions = {
            "return_recommendation",
            "return_grounded_answer",
            "return_uncertain_answer",
            "safe_fallback",
        }
        questions_before_finalize = 0
        finalized = False
        for actual_turn in run.turns:
            questions_before_finalize += len(actual_turn.clarification_questions)
            if any(
                action.status == "completed" and action.action in terminal_actions
                for action in actual_turn.actions
            ):
                finalized = True
                break
        values["average_questions_before_finalize"] = (
            float(questions_before_finalize) if finalized else None
        )
        clarification_index = next(
            (
                index
                for index, actual_turn in enumerate(run.turns)
                if actual_turn.clarification_questions
            ),
            None,
        )
        post_clarification_gain = None
        if clarification_index is not None and relevant:
            before_ranking = run.turns[clarification_index].candidate_ranking
            after_ranking = next(
                (
                    actual_turn.candidate_ranking
                    for actual_turn in run.turns[clarification_index + 1 :]
                    if actual_turn.candidate_ranking
                ),
                None,
            )
            if after_ranking is not None:
                before_utility = _ranking_metrics(before_ranking, relevant)[
                    "ndcg_at_5"
                ]
                after_utility = _ranking_metrics(after_ranking, relevant)[
                    "ndcg_at_5"
                ]
                if before_utility is not None and after_utility is not None:
                    post_clarification_gain = after_utility - before_utility
        values["post_clarification_utility_gain"] = post_clarification_gain
        values["task_type_accuracy"] = task_type_correct / task_type_count
        micro_numerators["task_type_accuracy"] += task_type_correct
        micro_denominators["task_type_accuracy"] += task_type_count
        micro_numerators["action_accuracy"] += len(
            completed_actions.intersection(required)
        )
        micro_denominators["action_accuracy"] += len(required)
        micro_numerators["missing_field_detection_precision"] += gap_true_positive
        micro_denominators["missing_field_detection_precision"] += (
            gap_true_positive + gap_false_positive
        )
        micro_numerators["missing_field_detection_recall"] += gap_true_positive
        micro_denominators["missing_field_detection_recall"] += (
            gap_true_positive + gap_false_negative
        )
        micro_numerators["unnecessary_question_rate"] += unnecessary_questions
        micro_denominators["unnecessary_question_rate"] += question_count
        micro_numerators["question_answerability_rate"] += answerable_questions
        micro_denominators["question_answerability_rate"] += question_count

        selected_action_count = 0
        invalid_action_count = 0
        tool_call_count = 0
        correct_tool_count = 0
        repeated_tool_count = 0
        semantic_tool_count = 0
        unnecessary_semantic_count = 0
        rag_tool_count = 0
        unnecessary_rag_count = 0
        direct_return_count = 0
        valid_direct_return_count = 0
        seen_tool_signatures: set[tuple[str, str]] = set()
        direct_return_actions = {
            "return_recommendation",
            "return_grounded_answer",
            "return_uncertain_answer",
        }
        for actual_turn in run.turns:
            if actual_turn.turn_index == 1:
                allowed = set(truth.allowed_actions)
                forbidden = set(truth.forbidden_actions)
                required_for_turn = set(truth.required_actions)
                gaps_for_turn = set(truth.expected_information_gaps)
            else:
                scripted = next(
                    (
                        item
                        for item in truth.scripted_user_turns
                        if item.turn_index == actual_turn.turn_index
                    ),
                    None,
                )
                allowed = set(scripted.expected_allowed_actions) if scripted else set()
                forbidden = set()
                required_for_turn = set()
                gaps_for_turn = (
                    set(scripted.expected_information_gaps) if scripted else set()
                )
            completed_before: set[str] = set()
            for action in actual_turn.actions:
                selected_action_count += 1
                invalid_action_count += int(
                    action.action not in allowed or action.action in forbidden
                )
                if action.action in direct_return_actions:
                    direct_return_count += 1
                    valid_direct_return_count += int(
                        not gaps_for_turn
                        and required_for_turn.difference(direct_return_actions)
                        .issubset(completed_before)
                    )
                if action.status == "completed":
                    completed_before.add(action.action)
            for call in actual_turn.tool_calls:
                tool_call_count += 1
                correct_tool_count += int(
                    call.action in allowed and call.action not in forbidden
                )
                signature = (call.tool_name, call.arguments_sha256)
                repeated_tool_count += int(signature in seen_tool_signatures)
                seen_tool_signatures.add(signature)
                if call.tool_kind == "semantic":
                    semantic_tool_count += 1
                    unnecessary_semantic_count += int(
                        call.action not in required_for_turn
                    )
                if call.tool_kind == "review_rag":
                    rag_tool_count += 1
                    unnecessary_rag_count += int(
                        "retrieve_business_reviews" not in required_for_turn
                    )
        values["invalid_action_rate"] = (
            invalid_action_count / selected_action_count
            if selected_action_count
            else None
        )
        values["tool_selection_accuracy"] = (
            correct_tool_count / tool_call_count if tool_call_count else None
        )
        values["repeated_tool_call_rate"] = (
            repeated_tool_count / tool_call_count if tool_call_count else None
        )
        values["direct_return_precision"] = (
            valid_direct_return_count / direct_return_count
            if direct_return_count
            else None
        )
        values["unnecessary_semantic_tool_rate"] = (
            unnecessary_semantic_count / semantic_tool_count
            if semantic_tool_count
            else None
        )
        values["unnecessary_rag_call_rate"] = (
            unnecessary_rag_count / rag_tool_count if rag_tool_count else None
        )
        values["fallback_rate"] = float(run.fallback)
        micro_numerators["invalid_action_rate"] += invalid_action_count
        micro_denominators["invalid_action_rate"] += selected_action_count
        micro_numerators["tool_selection_accuracy"] += correct_tool_count
        micro_denominators["tool_selection_accuracy"] += tool_call_count
        micro_numerators["repeated_tool_call_rate"] += repeated_tool_count
        micro_denominators["repeated_tool_call_rate"] += tool_call_count
        micro_numerators["direct_return_precision"] += valid_direct_return_count
        micro_denominators["direct_return_precision"] += direct_return_count
        micro_numerators["unnecessary_semantic_tool_rate"] += (
            unnecessary_semantic_count
        )
        micro_denominators["unnecessary_semantic_tool_rate"] += semantic_tool_count
        micro_numerators["unnecessary_rag_call_rate"] += unnecessary_rag_count
        micro_denominators["unnecessary_rag_call_rate"] += rag_tool_count

        latencies.append(run.latency_ms)
        if run.input_tokens is not None and run.output_tokens is not None:
            token_totals.append(float(run.input_tokens + run.output_tokens))
        all_tool_calls = [
            call for actual_turn in run.turns for call in actual_turn.tool_calls
        ]
        cache_hits += sum(call.cache_hit for call in all_tool_calls)
        cache_calls += len(all_tool_calls)
        if truth.task_type == "recommendation_request":
            if run.cost_usd is None:
                recommendation_cost_complete = False
            else:
                recommendation_costs.append(run.cost_usd)
            recommendations = ranking_turn.recommended_business_ids
            in_scope = bool(recommendations) and all(
                item in set(truth.business_scope) for item in recommendations
            )
            relevant_success = not relevant or any(
                item in relevant for item in recommendations
            )
            hard_success = hard_satisfaction in {None, 1.0}
            successful_recommendations += int(
                in_scope and relevant_success and hard_success
            )

        scenario_labels = labels_by_scenario.get(scenario_id, [])
        relevant_labels = {
            _evidence_key(item): item
            for item in scenario_labels
            if item.relevance == "relevant"
        }
        relevant_review_keys = {
            key
            for key, label in relevant_labels.items()
            if label.source_type == "review"
        }
        retrieved = []
        for actual_turn in run.turns:
            for call in actual_turn.tool_calls:
                retrieved.extend(
                    item.evidence
                    for item in sorted(
                        call.retrieved_evidence,
                        key=lambda item: item.rank,
                    )
                )
        unique_retrieved = []
        seen_evidence: set[tuple[str, str, str]] = set()
        for reference in retrieved:
            key = _evidence_key(reference)
            if key not in seen_evidence:
                seen_evidence.add(key)
                unique_retrieved.append(reference)
        if unique_retrieved:
            values["business_scope_isolation_rate"] = sum(
                item.business_id in set(truth.business_scope)
                for item in unique_retrieved
            ) / len(unique_retrieved)
        elif "retrieve_business_reviews" in truth.required_actions:
            values["business_scope_isolation_rate"] = 0.0
        else:
            values["business_scope_isolation_rate"] = None
        if unique_retrieved:
            micro_numerators["business_scope_isolation_rate"] += sum(
                item.business_id in set(truth.business_scope)
                for item in unique_retrieved
            )
            micro_denominators["business_scope_isolation_rate"] += len(
                unique_retrieved
            )
        elif "retrieve_business_reviews" in truth.required_actions:
            micro_denominators["business_scope_isolation_rate"] += 1

        review_retrieved = [
            item for item in unique_retrieved if item.source_type == "review"
        ]
        for cutoff in (1, 3, 5):
            if relevant_review_keys:
                found = {
                    _evidence_key(item)
                    for item in review_retrieved[:cutoff]
                    if _evidence_key(item) in relevant_review_keys
                }
                values[f"review_retrieval_recall_at_{cutoff}"] = (
                    len(found) / len(relevant_review_keys)
                )
            else:
                values[f"review_retrieval_recall_at_{cutoff}"] = None
            if relevant_labels:
                top_evidence = unique_retrieved[:cutoff]
                values[f"evidence_precision_at_{cutoff}"] = (
                    sum(
                        _evidence_key(item) in relevant_labels
                        for item in top_evidence
                    )
                    / len(top_evidence)
                    if top_evidence
                    else 0.0
                )
            else:
                values[f"evidence_precision_at_{cutoff}"] = None

        claims = [claim for actual_turn in run.turns for claim in actual_turn.claims]
        supported_claims = 0
        citations = []
        correct_citations = 0
        for claim in claims:
            claim_citations = list(claim.evidence_refs)
            citations.extend(claim_citations)
            relevant_citations = [
                item
                for item in claim_citations
                if _evidence_key(item) in relevant_labels
            ]
            supported_claims += int(bool(relevant_citations))
            correct_citations += len(relevant_citations)
        evidence_response = any(
            turn.response_kind in {"grounded_answer", "uncertain_answer"}
            for turn in run.turns
        )
        if evidence_response:
            uncertain_without_claim = not claims and any(
                turn.response_kind == "uncertain_answer" for turn in run.turns
            )
            values["grounded_answer_rate"] = float(
                uncertain_without_claim
                or (bool(claims) and supported_claims == len(claims))
            )
        else:
            values["grounded_answer_rate"] = None
        values["unsupported_claim_rate"] = (
            (len(claims) - supported_claims) / len(claims) if claims else None
        )
        values["citation_correctness"] = (
            correct_citations / len(citations)
            if citations
            else (0.0 if claims else None)
        )
        micro_numerators["unsupported_claim_rate"] += len(claims) - supported_claims
        micro_denominators["unsupported_claim_rate"] += len(claims)
        micro_numerators["citation_correctness"] += correct_citations
        micro_denominators["citation_correctness"] += len(citations)
        values["evidence_recency_reporting_rate"] = (
            float(any(turn.reported_evidence_recency for turn in run.turns))
            if relevant_review_keys
            else None
        )
        evidence_uncertainty_applicable = bool(relevant_labels) or (
            truth.scenario_category == "evidence_uncertainty"
        )
        values["conflict_detection_accuracy"] = (
            float(
                any(turn.reported_conflict for turn in run.turns)
                == (truth.uncertainty_policy == "report_conflict")
            )
            if evidence_uncertainty_applicable
            else None
        )
        official_policy_applicable = (
            truth.task_type == "official_policy_question"
            or truth.uncertainty_policy == "require_official_verification"
        )
        values["official_policy_caution_accuracy"] = (
            float(
                any(
                    turn.recommended_official_verification for turn in run.turns
                )
            )
            if official_policy_applicable
            else None
        )
        for name, value in values.items():
            if value is not None:
                totals[name] += value
                denominators[name] += 1
        split_counts[visible_item.split] += 1
        scenario_results.append(
            ScenarioEvaluationResult(
                scenario_id=scenario_id,
                split=visible_item.split,
                category=truth.scenario_category,
                metrics=values,
                violations=violations,
            )
        )
    aggregate_metrics = {
        name: _ratio(totals[name], denominators[name])
        for name in sorted(totals | denominators)
    }
    for name in sorted(micro_numerators | micro_denominators):
        aggregate_metrics[name] = _ratio(
            micro_numerators[name],
            micro_denominators[name],
        )
    aggregate_metrics.update(
        {
            "mean_latency_ms": _mean_score(latencies, unit="milliseconds"),
            "p50_latency_ms": _percentile_score(
                latencies,
                quantile=0.5,
                unit="milliseconds",
            ),
            "p95_latency_ms": _percentile_score(
                latencies,
                quantile=0.95,
                unit="milliseconds",
            ),
            "mean_tokens": _mean_score(token_totals, unit="tokens"),
            "p95_tokens": _percentile_score(
                token_totals,
                quantile=0.95,
                unit="tokens",
            ),
            "cache_hit_rate": _ratio(cache_hits, cache_calls),
        }
    )
    for cutoff in (50, 100, 500):
        aggregate_metrics[f"recall_at_{cutoff}"] = _unavailable(
            "ratio",
            "requires_step11_full_retrieval_benchmark",
        )
    if recommendation_cost_complete and recommendation_costs:
        aggregate_metrics["cost_per_successful_recommendation"] = (
            MetricScore(
                status=(
                    "measured" if successful_recommendations else "unavailable"
                ),
                value=(
                    sum(recommendation_costs) / successful_recommendations
                    if successful_recommendations
                    else None
                ),
                numerator=sum(recommendation_costs),
                denominator=successful_recommendations,
                unit="usd",
                reason=(
                    None if successful_recommendations else "no_successful_recommendations"
                ),
            )
        )
    else:
        aggregate_metrics["cost_per_successful_recommendation"] = _unavailable(
            "usd",
            "cost_observations_incomplete",
        )
    for definition in metric_definitions():
        if definition.name not in aggregate_metrics:
            aggregate_metrics[definition.name] = _ratio(0, 0).model_copy(
                update={"unit": definition.unit}
            )
    by_split: dict[str, AgentEvaluationSlice] = {}
    by_category: dict[str, AgentEvaluationSlice] = {}
    if include_breakdowns:
        for split in sorted({item.split for item in visible_scenarios}):
            ids = {
                item.scenario_id for item in visible_scenarios if item.split == split
            }
            sliced = _evaluate_agent_scenario_runs(
                [item for item in runs if item.scenario_id in ids],
                visible_scenarios=[
                    item for item in visible_scenarios if item.scenario_id in ids
                ],
                ground_truth=[
                    item for item in ground_truth if item.scenario_id in ids
                ],
                evidence_labels=[
                    item for item in evidence_labels if item.scenario_id in ids
                ],
                include_breakdowns=False,
            )
            by_split[split] = AgentEvaluationSlice(
                scenario_count=sliced.scenario_count,
                metrics=sliced.metrics,
            )
        for category in sorted(
            {item.scenario_category for item in ground_truth}
        ):
            ids = {
                item.scenario_id
                for item in ground_truth
                if item.scenario_category == category
            }
            sliced = _evaluate_agent_scenario_runs(
                [item for item in runs if item.scenario_id in ids],
                visible_scenarios=[
                    item for item in visible_scenarios if item.scenario_id in ids
                ],
                ground_truth=[
                    item for item in ground_truth if item.scenario_id in ids
                ],
                evidence_labels=[
                    item for item in evidence_labels if item.scenario_id in ids
                ],
                include_breakdowns=False,
            )
            by_category[category] = AgentEvaluationSlice(
                scenario_count=sliced.scenario_count,
                metrics=sliced.metrics,
            )
    return AgentEvaluationReport(
        scenario_count=len(runs),
        split_counts=dict(sorted(split_counts.items())),
        metrics=dict(sorted(aggregate_metrics.items())),
        by_split=by_split,
        by_category=by_category,
        scenario_results=scenario_results,
    )


def evaluate_agent_scenario_runs(
    runs: Sequence[AgentScenarioRun],
    *,
    visible_scenarios: Sequence[VisibleAgentScenario],
    ground_truth: Sequence[ScenarioGroundTruth],
    evidence_labels: Sequence[EvidenceLabel],
) -> AgentEvaluationReport:
    """Evaluate runs once and return overall, split, and scenario-category views."""

    return _evaluate_agent_scenario_runs(
        runs,
        visible_scenarios=visible_scenarios,
        ground_truth=ground_truth,
        evidence_labels=evidence_labels,
        include_breakdowns=True,
    )

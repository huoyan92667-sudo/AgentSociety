"""Behavior tests for the behavior-anchored Query recommendation benchmark."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.query.benchmark import ExpectedRequestCondition
from yelp_agent.query_recommendation_benchmark import (
    BehaviorAnchorCandidate,
    AnchorQueryKnowledge,
    QueryFramePlanConfig,
    QueryRecommendationBenchmarkConfig,
    QueryRecommendationSources,
    QueryRecommendationBenchmarkBundle,
    QueryRecommendationFrame,
    QueryRecommendationGroundTruth,
    VisibleQueryRecommendationCase,
    load_query_recommendation_bundle,
    publish_query_recommendation_bundle,
    select_behavior_anchors,
    load_behavior_anchor_candidates,
    plan_query_recommendation_frames,
    QueryRewrite,
    assemble_query_recommendation_bundle,
    build_query_rewrite_prompt,
    audit_query_recommendation_benchmark,
    BenchmarkRetrievalRun,
    evaluate_query_recommendation_retrieval,
)


def _condition() -> ExpectedRequestCondition:
    return ExpectedRequestCondition(
        field="category",
        operator="includes",
        value="Steakhouses",
        importance="strong",
        enforcement="rank",
    )


def _bundle() -> QueryRecommendationBenchmarkBundle:
    case_id = "a" * 64
    return QueryRecommendationBenchmarkBundle(
        visible_cases=(
            VisibleQueryRecommendationCase(
                case_id=case_id,
                split="development",
                language="en-US",
                user_id="user-1",
                session_id="query-recommendation:user-1",
                cutoff_time=datetime(2021, 1, 1),
                query_text="Find me a steakhouse.",
                generator_kind="deterministic",
            ),
        ),
        ground_truth=(
            QueryRecommendationGroundTruth(
                case_id=case_id,
                source_task_id="train:user-1:000010",
                target_business_id="business-secret",
                target_review_id="review-secret",
                target_stars=5.0,
                target_time=datetime(2021, 1, 1),
            ),
        ),
        frames=(
            QueryRecommendationFrame(
                case_id=case_id,
                frame_family="category_only",
                conditions=[_condition()],
                location_source="none",
                target_support_sources=["static_category"],
            ),
        ),
    )


def test_published_bundle_keeps_the_real_positive_out_of_visible_cases(tmp_path) -> None:
    result = publish_query_recommendation_bundle(
        _bundle(),
        tmp_path / "query_recommendation_v1",
    )

    visible_text = result.visible_path.read_text(encoding="utf-8")
    hidden_text = result.ground_truth_path.read_text(encoding="utf-8")
    assert "business-secret" not in visible_text
    assert "review-secret" not in visible_text
    assert "target_business_id" not in visible_text
    assert "business-secret" in hidden_text
    assert result.manifest.hidden_labels_visible_to_agent is False
    assert result.manifest.llm_determined_hidden_labels is False

    loaded = load_query_recommendation_bundle(result.root)
    assert loaded == _bundle()


def test_visible_case_contract_rejects_any_hidden_target_field() -> None:
    payload = _bundle().visible_cases[0].model_dump(mode="json")
    payload["target_business_id"] = "leaked-business"

    with pytest.raises(ValueError):
        VisibleQueryRecommendationCase.model_validate(payload)


def test_publisher_refuses_to_mix_with_an_existing_partial_bundle(tmp_path) -> None:
    root = tmp_path / "query_recommendation_v1"
    root.mkdir()
    (root / "visible").mkdir()
    (root / "visible" / "cases.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_query_recommendation_bundle(_bundle(), root)


def _anchor(
    task_id: str,
    split: str,
    user_id: str,
    *,
    stars: float = 5.0,
    target_business_id: str | None = None,
    history_business_ids: tuple[str, ...] = ("history-a", "history-b"),
    pre_cutoff_reviews: int = 3,
) -> BehaviorAnchorCandidate:
    return BehaviorAnchorCandidate(
        source_task_id=task_id,
        source_split=split,
        user_id=user_id,
        cutoff_time=datetime(2021, 1, 1),
        history_business_ids=history_business_ids,
        target_business_id=target_business_id or f"target-{user_id}",
        target_review_id=f"review-{task_id}",
        target_stars=stars,
        target_time=datetime(2021, 1, 1),
        target_pre_cutoff_review_count=pre_cutoff_reviews,
    )


def test_anchor_selector_keeps_real_positive_rules_and_separates_users() -> None:
    candidates = (
        _anchor("validation:v1", "validation", "user-v1"),
        _anchor("validation:v2", "validation", "shared-user"),
        _anchor("validation:low", "validation", "low-user", stars=3.0),
        _anchor("train:d1", "train", "user-d1"),
        _anchor("train:d2", "train", "user-d2"),
        _anchor("train:shared", "train", "shared-user"),
        _anchor(
            "train:revisit",
            "train",
            "revisit-user",
            target_business_id="history-a",
        ),
        _anchor("train:future", "train", "future-user", pre_cutoff_reviews=0),
    )
    config = QueryRecommendationBenchmarkConfig(
        development_count=2,
        validation_count=2,
        seed=42,
    )

    selected = select_behavior_anchors(candidates, config)

    assert len(selected) == 4
    development = [item for item in selected if item.benchmark_split == "development"]
    validation = [item for item in selected if item.benchmark_split == "validation"]
    assert {item.user_id for item in development} == {"user-d1", "user-d2"}
    assert {item.user_id for item in validation} == {"user-v1", "shared-user"}
    assert not {item.user_id for item in development}.intersection(
        item.user_id for item in validation
    )
    assert all(item.target_stars >= 4 for item in selected)
    assert all(
        item.target_business_id not in item.history_business_ids for item in selected
    )
    assert selected == select_behavior_anchors(tuple(reversed(candidates)), config)


def _write_rows(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_source_loader_joins_real_targets_without_loading_target_review_text(tmp_path) -> None:
    reviews = tmp_path / "reviews.parquet"
    train_contexts = tmp_path / "train_contexts.parquet"
    train_histories = tmp_path / "train_histories.parquet"
    train_truth = tmp_path / "train_truth.parquet"
    validation_contexts = tmp_path / "validation_contexts.parquet"
    validation_histories = tmp_path / "validation_histories.parquet"
    validation_truth = tmp_path / "validation_truth.parquet"
    _write_rows(
        reviews,
        [
            {"review_id": "h-train", "user_id": "u-train", "business_id": "old-a", "stars": 5.0, "text": "history", "date": datetime(2020, 1, 1)},
            {"review_id": "pre-train", "user_id": "other", "business_id": "target-a", "stars": 3.0, "text": "pre-existing", "date": datetime(2020, 6, 1)},
            {"review_id": "t-train", "user_id": "u-train", "business_id": "target-a", "stars": 5.0, "text": "SECRET TARGET TEXT", "date": datetime(2021, 1, 1)},
            {"review_id": "h-val", "user_id": "u-val", "business_id": "old-b", "stars": 4.0, "text": "history", "date": datetime(2020, 2, 1)},
            {"review_id": "pre-val", "user_id": "other", "business_id": "target-b", "stars": 4.0, "text": "pre-existing", "date": datetime(2020, 7, 1)},
            {"review_id": "t-val", "user_id": "u-val", "business_id": "target-b", "stars": 4.0, "text": "SECRET VALIDATION TEXT", "date": datetime(2021, 2, 1)},
        ],
    )
    _write_rows(train_contexts, [{"task_id": "train:u:1", "split": "train", "user_id": "u-train", "cutoff_time": datetime(2021, 1, 1), "history_count": 1}])
    _write_rows(train_histories, [{"task_id": "train:u:1", "position": 1, "review_id": "h-train"}])
    _write_rows(train_truth, [{"task_id": "train:u:1", "target_review_id": "t-train", "target_business_id": "target-a"}])
    _write_rows(validation_contexts, [{"task_id": "validation:u-val", "split": "validation", "user_id": "u-val", "cutoff_time": datetime(2021, 2, 1), "history_count": 1}])
    _write_rows(validation_histories, [{"task_id": "validation:u-val", "position": 1, "review_id": "h-val"}])
    _write_rows(validation_truth, [{"task_id": "validation:u-val", "target_business_id": "target-b"}])

    loaded = load_behavior_anchor_candidates(
        QueryRecommendationSources(
            reviews=reviews,
            training_contexts=train_contexts,
            training_histories=train_histories,
            training_ground_truth=train_truth,
            validation_contexts=validation_contexts,
            validation_histories=validation_histories,
            validation_ground_truth=validation_truth,
        )
    )

    assert [(item.source_split, item.target_review_id) for item in loaded] == [
        ("train", "t-train"),
        ("validation", "t-val"),
    ]
    assert [item.target_pre_cutoff_review_count for item in loaded] == [1, 1]
    assert loaded[0].history_business_ids == ("old-a",)
    assert "SECRET" not in repr(loaded)


class _KnowledgeReader:
    def describe(self, anchor):
        values = {
            "user-d1": AnchorQueryKnowledge(
                business_id="target-user-d1",
                fine_categories=("Steakhouses",),
                is_food_business=True,
                price_level=3,
                user_latitude=39.95,
                user_longitude=-75.16,
                target_distance_km=2.2,
                positive_aspects=("quiet_environment", "date_suitable"),
                aspect_source_scope="selected_user_interactions",
            ),
            "user-d2": AnchorQueryKnowledge(
                business_id="target-user-d2",
                fine_categories=("Sushi Bars",),
                is_food_business=True,
                price_level=None,
                user_latitude=39.96,
                user_longitude=-75.17,
                target_distance_km=1.0,
                positive_aspects=("service",),
                aspect_source_scope="selected_user_interactions",
            ),
            "user-v1": AnchorQueryKnowledge(
                business_id="target-user-v1",
                fine_categories=("Italian",),
                is_food_business=True,
                price_level=2,
                user_latitude=39.94,
                user_longitude=-75.15,
                target_distance_km=4.1,
                positive_aspects=(),
                aspect_source_scope="selected_user_interactions",
            ),
            "shared-user": AnchorQueryKnowledge(
                business_id="target-shared-user",
                fine_categories=("Coffee & Tea",),
                is_food_business=True,
                price_level=1,
                user_latitude=39.93,
                user_longitude=-75.14,
                target_distance_km=0.8,
                positive_aspects=(),
                aspect_source_scope="selected_user_interactions",
            ),
        }
        return values[anchor.user_id]


def test_frame_planner_uses_only_conditions_the_target_satisfied_at_cutoff() -> None:
    selected = select_behavior_anchors(
        (
            _anchor("validation:v1", "validation", "user-v1"),
            _anchor("validation:v2", "validation", "shared-user"),
            _anchor("train:d1", "train", "user-d1"),
            _anchor("train:d2", "train", "user-d2"),
        ),
        QueryRecommendationBenchmarkConfig(
            development_count=2,
            validation_count=2,
        ),
    )
    plan = QueryFramePlanConfig(
        development_family_counts={
            "category_two_aspects": 1,
            "category_single_aspect": 1,
        },
        validation_family_counts={
            "category_distance": 1,
            "category_price": 1,
        },
        development_language_counts={"zh-CN": 1, "en-US": 1},
        validation_language_counts={"zh-CN": 1, "en-US": 1},
    )

    drafts = plan_query_recommendation_frames(selected, _KnowledgeReader(), plan)

    assert len(drafts) == 4
    assert {draft.frame.frame_family for draft in drafts} == {
        "category_two_aspects",
        "category_single_aspect",
        "category_distance",
        "category_price",
    }
    for draft in drafts:
        knowledge = _KnowledgeReader().describe(draft.anchor)
        category = next(
            condition for condition in draft.frame.conditions if condition.field == "category"
        )
        assert str(category.value) in knowledge.fine_categories
        for condition in draft.frame.conditions:
            if condition.field in knowledge.positive_aspects:
                assert condition.operator == "prefer"
        if any(item.field == "distance_km" for item in draft.frame.conditions):
            distance = next(
                float(item.value)
                for item in draft.frame.conditions
                if item.field == "distance_km"
            )
            assert knowledge.target_distance_km <= distance
            assert draft.user_latitude == knowledge.user_latitude
        if any(item.field == "price_level" for item in draft.frame.conditions):
            price = next(
                int(item.value)
                for item in draft.frame.conditions
                if item.field == "price_level"
            )
            assert price == knowledge.price_level
    assert {draft.language for draft in drafts} == {"zh-CN", "en-US"}


def test_rewrite_prompt_and_visible_output_never_expose_target_identity() -> None:
    anchor = BehaviorAnchorCandidate(
        source_task_id="train:secret-user:000010",
        source_split="train",
        user_id="secret-user",
        cutoff_time=datetime(2021, 1, 1),
        history_business_ids=("history-a",),
        target_business_id="secret-business-id",
        target_review_id="secret-review-id",
        target_stars=5.0,
        target_time=datetime(2021, 1, 1),
        target_pre_cutoff_review_count=3,
    )
    selected = select_behavior_anchors(
        (
            anchor,
            _anchor("validation:v", "validation", "validation-user"),
        ),
        QueryRecommendationBenchmarkConfig(
            development_count=1,
            validation_count=1,
        ),
    )

    class Reader:
        def describe(self, item):
            return AnchorQueryKnowledge(
                business_id=item.target_business_id,
                fine_categories=("Steakhouses",),
                is_food_business=True,
                user_latitude=39.95,
                user_longitude=-75.16,
                target_distance_km=1.0,
                positive_aspects=(),
                aspect_source_scope="selected_user_interactions",
            )

    drafts = plan_query_recommendation_frames(
        selected,
        Reader(),
        QueryFramePlanConfig(
            development_family_counts={"category_only": 1},
            validation_family_counts={"category_only": 1},
            development_language_counts={"en-US": 1},
            validation_language_counts={"zh-CN": 1},
        ),
    )
    system, user, prompt_hash = build_query_rewrite_prompt(drafts)

    prompt = system + user
    assert "secret-business-id" not in prompt
    assert "secret-review-id" not in prompt
    assert "secret-user" not in prompt
    assert len(prompt_hash) == 64

    rewrites = tuple(
        QueryRewrite(
            case_id=draft.frame.case_id,
            language=draft.language,
            query_text=(
                "Find me a steakhouse."
                if draft.language == "en-US"
                else "请帮我找一家牛排馆。"
            ),
            condition_fidelity_passed=True,
            audit_reason_codes=[],
        )
        for draft in drafts
    )
    bundle = assemble_query_recommendation_bundle(
        drafts,
        rewrites,
        generator_model="fake-deepseek",
        generator_prompt_sha256=prompt_hash,
    )
    visible = "\n".join(item.model_dump_json() for item in bundle.visible_cases)
    assert "secret-business-id" not in visible
    assert "secret-review-id" not in visible
    assert {item.generator_kind for item in bundle.visible_cases} == {
        "openai_compatible"
    }
    audit = audit_query_recommendation_benchmark(bundle, drafts, Reader())
    assert audit.passed is True
    assert audit.violation_count == 0
    assert audit.target_review_text_loaded is False
    assert audit.test_source_tasks_used is False
    assert audit.aspect_source_scope == "selected_user_interactions"


def test_assembler_rejects_failed_fidelity_or_identity_leak() -> None:
    selected = select_behavior_anchors(
        (
            _anchor("train:d", "train", "user-d1"),
            _anchor("validation:v", "validation", "user-v1"),
        ),
        QueryRecommendationBenchmarkConfig(
            development_count=1,
            validation_count=1,
        ),
    )
    plan = QueryFramePlanConfig(
        development_family_counts={"category_only": 1},
        validation_family_counts={"category_only": 1},
        development_language_counts={"en-US": 1},
        validation_language_counts={"en-US": 1},
    )
    drafts = plan_query_recommendation_frames(selected, _KnowledgeReader(), plan)
    failed = tuple(
        QueryRewrite(
            case_id=draft.frame.case_id,
            language=draft.language,
            query_text=(
                draft.anchor.target_business_id
                if index == 0
                else "Find a steakhouse."
            ),
            condition_fidelity_passed=index != 1,
            audit_reason_codes=[] if index == 0 else ["MISSING_CATEGORY"],
        )
        for index, draft in enumerate(drafts)
    )

    with pytest.raises(ValueError):
        assemble_query_recommendation_bundle(
            drafts,
            failed,
            generator_model="fake-deepseek",
            generator_prompt_sha256="b" * 64,
        )


def test_retrieval_evaluator_compares_only_history_query_and_dual_channels() -> None:
    cases = tuple(
        VisibleQueryRecommendationCase(
            case_id=letter * 64,
            split="development" if letter != "c" else "validation",
            language="en-US",
            user_id=f"user-{letter}",
            session_id=f"session-{letter}",
            cutoff_time=datetime(2021, 1, 1),
            query_text=f"query {letter}",
            generator_kind="deterministic",
        )
        for letter in "abc"
    )
    truths = tuple(
        QueryRecommendationGroundTruth(
            case_id=letter * 64,
            source_task_id=f"task-{letter}",
            target_business_id=f"target-{letter}",
            target_review_id=f"review-{letter}",
            target_stars=5.0,
            target_time=datetime(2021, 1, 1),
        )
        for letter in "abc"
    )
    rankings = {
        "history_only": {
            "a": ["target-a"],
            "b": ["other"],
            "c": ["other"],
        },
        "query_only": {
            "a": ["other"],
            "b": ["target-b"],
            "c": ["target-c"],
        },
        "history_query": {
            "a": ["target-a"],
            "b": ["target-b"],
            "c": ["other"],
        },
    }
    runs = tuple(
        BenchmarkRetrievalRun(
            case_id=letter * 64,
            method=method,
            ranking=values[letter],
            latency_ms=1.0,
        )
        for method, values in rankings.items()
        for letter in "abc"
    )

    report = evaluate_query_recommendation_retrieval(cases, truths, runs)

    assert set(report.overall) == {
        "history_only",
        "query_only",
        "history_query",
    }
    assert report.overall["history_only"].recall_at_500 == pytest.approx(1 / 3)
    assert report.overall["query_only"].recall_at_500 == pytest.approx(2 / 3)
    assert report.overall["history_query"].recall_at_500 == pytest.approx(2 / 3)
    assert report.query_rescue_rate == pytest.approx(1.0)
    assert report.fusion_loss_rate == pytest.approx(1 / 3)
    assert report.performance_claim_scope == "single_known_positive_retrieval"


def test_frame_planner_does_not_render_split_yelp_category_fragments() -> None:
    selected = select_behavior_anchors(
        (
            _anchor("train:d", "train", "user-d1"),
            _anchor("validation:v", "validation", "user-v1"),
        ),
        QueryRecommendationBenchmarkConfig(
            development_count=1,
            validation_count=1,
        ),
    )

    class Reader:
        def describe(self, anchor):
            return AnchorQueryKnowledge(
                business_id=anchor.target_business_id,
                fine_categories=("Books", "Mags", "Music & Video", "Bookstores"),
                is_food_business=False,
                positive_aspects=(),
                aspect_source_scope="selected_user_interactions",
            )

    drafts = plan_query_recommendation_frames(
        selected,
        Reader(),
        QueryFramePlanConfig(
            development_family_counts={"category_only": 1},
            validation_family_counts={"category_only": 1},
            development_language_counts={"en-US": 1},
            validation_language_counts={"en-US": 1},
        ),
    )

    assert {
        str(draft.frame.conditions[0].value) for draft in drafts
    } == {"Bookstores"}

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from yelp_agent.agent.llm import LLMAttemptTrace, LLMCallResult, LLMMessage
from yelp_agent.agent.tools import AgentToolbox
from yelp_agent.config import AppConfig, load_config
from yelp_agent.data.businesses import preprocess_businesses
from yelp_agent.data.candidates import build_candidate_tasks
from yelp_agent.data.reviews import preprocess_reviews
from yelp_agent.data.temporal import build_temporal_splits
from yelp_agent.data.users import preprocess_users_and_interactions
from yelp_agent.evaluation.evaluator import evaluate_prediction_file
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.hybrid import HybridFeatureStore, HybridWeights
from yelp_agent.features.location import TemporalLocationStore
from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.features.text import TemporalTextStore, fit_tfidf_model
from yelp_agent.models import Prediction, RecommendationTask
from yelp_agent.rankers.agent_ranker import AgentRanker
from yelp_agent.rankers.agent_runner import run_agent_ranker
from yelp_agent.rankers.hybrid_ranker import HybridRanker
from yelp_agent.rankers.runner import run_ranker
from yelp_agent.tuning.hybrid import (
    fingerprint_hybrid_feature_sources,
    tune_hybrid_weights,
)


@dataclass(frozen=True)
class AgentOutcome:
    ranking: list[str]
    fallback: bool
    fallback_reason: str | None
    task_count: int
    valid_output_rate: float
    llm_failure_rate: float


@dataclass(frozen=True)
class SyntheticPipelineRun:
    root: Path
    validation_task_count: int
    test_task_count: int
    hybrid_ranking: list[str]
    agent_ranking: list[str]
    agent_fallback: bool
    agent_tool_calls: int
    agent_evaluation_task_count: int
    agent_valid_output_rate: float
    fake_llm_call_count: int
    hybrid_score_snapshot: dict[str, object]
    agent_outcomes: dict[str, AgentOutcome]
    agent_prompt_payloads: dict[str, dict[str, object]]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _business(
    business_id: str,
    categories: str,
    *,
    offset: int,
) -> dict[str, object]:
    return {
        "business_id": business_id,
        "name": business_id.replace("-", " ").title(),
        "address": f"{offset + 1} Market Street",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19103",
        "latitude": 39.95 + offset * 0.001,
        "longitude": -75.16 - offset * 0.001,
        "stars": 5.0,
        "review_count": 20,
        "is_open": 1,
        "attributes": {
            "OutdoorSeating": offset % 2 == 0,
            "WiFi": "free" if offset % 3 == 0 else "no",
        },
        "categories": categories,
    }


def _review(
    review_id: str,
    user_id: str,
    business_id: str,
    stars: int,
    date: str,
    text: str,
) -> dict[str, object]:
    return {
        "review_id": review_id,
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        "useful": 0,
        "funny": 0,
        "cool": 0,
        "text": text,
        "date": date,
    }


def _raw_businesses() -> list[dict[str, object]]:
    specs = [
        ("history-mexican", "Restaurants, Mexican"),
        ("history-coffee", "Food, Coffee & Tea"),
        ("history-hair", "Beauty & Spas, Hair Salons"),
        ("cocktail-a", "Nightlife, Bars, Cocktail Bars"),
        ("cocktail-b", "Nightlife, Bars, Cocktail Bars"),
        *[
            (f"same-{index}", "Nightlife, Bars, Cocktail Bars")
            for index in range(9)
        ],
        *[
            (f"related-{index}", "Nightlife, Bars, Dive Bars")
            for index in range(5)
        ],
        *[
            (f"preference-{index}", "Restaurants, Mexican")
            for index in range(3)
        ],
        *[
            (f"random-{index}", "Shopping, Bookstores")
            for index in range(3)
        ],
    ]
    return [
        _business(business_id, categories, offset=index)
        for index, (business_id, categories) in enumerate(specs)
    ]


def _raw_reviews(businesses: list[dict[str, object]]) -> list[dict[str, object]]:
    availability = [
        _review(
            f"availability-{business['business_id']}",
            "availability-user",
            str(business["business_id"]),
            4,
            "2019-01-01 00:00:00",
            "steady historical quality",
        )
        for business in businesses
    ]
    personal_history = [
        _review(
            "user-review-1",
            "user-1",
            "history-mexican",
            5,
            "2020-01-01 10:00:00",
            "excellent spicy tacos friendly service",
        ),
        _review(
            "user-review-2",
            "user-1",
            "history-coffee",
            1,
            "2020-01-02 10:00:00",
            "awful bitter coffee noisy room",
        ),
        _review(
            "user-review-3",
            "user-1",
            "history-hair",
            3,
            "2020-01-03 10:00:00",
            "ordinary haircut acceptable visit",
        ),
        _review(
            "user-review-4",
            "user-1",
            "cocktail-a",
            4,
            "2020-01-10 10:00:00",
            "creative cocktails relaxed atmosphere",
        ),
        _review(
            "user-review-5",
            "user-1",
            "cocktail-b",
            2,
            "2020-01-20 10:00:00",
            "slow service but interesting drinks",
        ),
    ]
    return [*availability, *personal_history]


def _raw_users() -> list[dict[str, object]]:
    return [
        {
            "user_id": "user-1",
            "name": "Synthetic User",
            "yelping_since": "2010-01-01 00:00:00",
            "review_count": 5,
            "average_stars": 3.0,
        }
    ]


def _future_reviews() -> list[dict[str, object]]:
    return [
        *[
            _review(
                f"future-five-{index}",
                "future-user",
                "same-0",
                5,
                f"2030-01-{index + 1:02d} 10:00:00",
                "future-only spectacular impossible leakage keyword",
            )
            for index in range(10)
        ],
        *[
            _review(
                f"future-one-{index}",
                "future-user",
                "same-1",
                1,
                f"2030-02-{index + 1:02d} 10:00:00",
                "future-only disastrous impossible leakage keyword",
            )
            for index in range(10)
        ],
    ]


def _test_config() -> AppConfig:
    config = load_config()
    return config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "min_user_reviews": 5,
                    "min_distinct_businesses": 5,
                    "min_distinct_ratings": 3,
                    "max_users": 10,
                    "review_chunk_size": 7,
                }
            ),
            "tfidf": config.tfidf.model_copy(
                update={"min_df": 1, "max_features": 1000}
            ),
        }
    )


def _build_feature_store(
    root: Path,
    config: AppConfig,
) -> tuple[HybridFeatureStore, TemporalQualityStore]:
    businesses = root / "processed" / "businesses.parquet"
    reviews = root / "processed" / "reviews.parquet"
    interactions = root / "processed" / "interactions.parquet"
    histories = root / "task_dataset" / "tasks" / "temporal_histories.parquet"
    tfidf_artifact = root / "features" / "tfidf_vectorizer.joblib"
    tfidf_manifest = root / "features" / "tfidf_manifest.json"
    quality_store = TemporalQualityStore(
        reviews,
        prior_count=config.hybrid.bayesian_prior_count,
    )
    feature_store = HybridFeatureStore(
        category_store=TemporalCategoryStore(
            businesses,
            interactions,
            histories,
            broad_categories=set(config.data.broad_categories),
        ),
        text_store=TemporalTextStore(
            businesses,
            interactions,
            histories,
            tfidf_artifact,
            tfidf_manifest,
        ),
        quality_store=quality_store,
        location_store=TemporalLocationStore(
            businesses,
            interactions,
            histories,
            scale_km=config.hybrid.location_scale_km,
        ),
    )
    return feature_store, quality_store


class _BehaviorFakeLLM:
    def __init__(self, behavior: str) -> None:
        self._behavior = behavior
        self.requests: list[list[LLMMessage]] = []

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        self.requests.append(messages)
        payload = json.loads(messages[-1].content)
        top_ids = [
            candidate["business_id"] for candidate in payload["candidates"]
        ]
        if self._behavior == "exception":
            raise RuntimeError("simulated provider exception")
        if self._behavior == "timeout":
            return LLMCallResult(
                status="failure",
                content=None,
                model="fake-llm",
                latency_ms=0.0,
                attempt_count=1,
                failure_reason="timeout",
                attempts=[
                    LLMAttemptTrace(
                        attempt_index=1,
                        status="failure",
                        latency_ms=0.0,
                        failure_reason="timeout",
                        retryable=True,
                        usage_unknown=True,
                    )
                ],
                unknown_usage_attempts=1,
            )
        if self._behavior == "disabled":
            return LLMCallResult(
                status="disabled",
                content=None,
                model="fake-llm",
                latency_ms=0.0,
                attempt_count=0,
                failure_reason="llm_disabled",
            )

        if self._behavior == "reverse":
            content = json.dumps(
                {
                    "ranking": list(reversed(top_ids)),
                    "reason": "deterministic fake response",
                }
            )
        elif self._behavior == "non_json":
            content = "not a JSON response"
        elif self._behavior == "duplicate":
            content = json.dumps({"ranking": [*top_ids[:7], top_ids[0]]})
        elif self._behavior == "missing":
            content = json.dumps({"ranking": top_ids[:7]})
        elif self._behavior == "unknown":
            content = json.dumps(
                {"ranking": [*top_ids[:7], "outside-candidate"]}
            )
        elif self._behavior == "empty":
            content = ""
        else:
            raise ValueError(f"unsupported fake LLM behavior: {self._behavior}")
        return LLMCallResult(
            status="success",
            content=content,
            model="fake-llm",
            latency_ms=0.0,
            attempt_count=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            attempts=[
                LLMAttemptTrace(
                    attempt_index=1,
                    status="success",
                    latency_ms=0.0,
                    retryable=False,
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    provider_request_id="fake-request",
                )
            ],
            observed_total_tokens=15,
        )


def _load_prediction(path: Path) -> Prediction:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise AssertionError(f"expected one prediction in {path}, found {len(lines)}")
    return Prediction.model_validate_json(lines[0])


def run_synthetic_pipeline(
    root: Path,
    *,
    reverse_review_rows: bool = False,
    include_future_reviews: bool = False,
    llm_behaviors: tuple[str, ...] = ("reverse",),
) -> SyntheticPipelineRun:
    if not llm_behaviors or len(set(llm_behaviors)) != len(llm_behaviors):
        raise ValueError("llm_behaviors must contain unique behavior names")
    config = _test_config()
    raw_root = root / "raw"
    processed_root = root / "processed"
    task_root = root / "task_dataset"
    businesses = _raw_businesses()
    reviews = _raw_reviews(businesses)
    if include_future_reviews:
        reviews.extend(_future_reviews())
    if reverse_review_rows:
        reviews.reverse()
    business_jsonl = raw_root / "business.jsonl"
    review_jsonl = raw_root / "review.jsonl"
    user_jsonl = raw_root / "user.jsonl"
    _write_jsonl(business_jsonl, businesses)
    _write_jsonl(review_jsonl, reviews)
    _write_jsonl(user_jsonl, _raw_users())

    businesses_parquet = processed_root / "businesses.parquet"
    reviews_parquet = processed_root / "reviews.parquet"
    users_parquet = processed_root / "users.parquet"
    interactions_parquet = processed_root / "interactions.parquet"
    preprocess_businesses(business_jsonl, businesses_parquet, config.data)
    preprocess_reviews(
        review_jsonl,
        businesses_parquet,
        reviews_parquet,
        config.data,
    )
    preprocess_users_and_interactions(
        reviews_parquet,
        user_jsonl,
        users_parquet,
        interactions_parquet,
        config.data,
    )
    split_result = build_temporal_splits(interactions_parquet, task_root)
    candidate_result = build_candidate_tasks(
        processed_root,
        task_root,
        config.data,
    )

    histories = Path(split_result.histories_path)
    tfidf_artifact = root / "features" / "tfidf_vectorizer.joblib"
    tfidf_manifest = root / "features" / "tfidf_manifest.json"
    fit_tfidf_model(
        businesses_parquet,
        interactions_parquet,
        histories,
        tfidf_artifact,
        tfidf_manifest,
        config.tfidf,
    )
    feature_store, quality_store = _build_feature_store(root, config)
    feature_fingerprints = fingerprint_hybrid_feature_sources(
        {
            "businesses": businesses_parquet,
            "reviews": reviews_parquet,
            "interactions": interactions_parquet,
            "histories": histories,
            "tfidf_artifact": tfidf_artifact,
            "tfidf_manifest": tfidf_manifest,
        }
    )
    tune_result = tune_hybrid_weights(
        candidate_result.validation_tasks_path,
        candidate_result.ground_truth_path,
        feature_store,
        root / "hybrid" / "weights.json",
        initial_weights=HybridWeights.from_config(config.hybrid),
        feature_sources_sha256=feature_fingerprints,
        step=0.5,
    )
    hybrid_ranker = HybridRanker(feature_store, tune_result.selected_weights)
    hybrid_predictions = root / "hybrid" / "predictions.jsonl"
    run_ranker(
        candidate_result.test_tasks_path,
        hybrid_ranker,
        hybrid_predictions,
    )
    hybrid_prediction = _load_prediction(hybrid_predictions)

    toolbox = AgentToolbox(
        businesses_parquet,
        interactions_parquet,
        histories,
        hybrid_ranker=hybrid_ranker,
        quality_store=quality_store,
    )
    agent_outcomes: dict[str, AgentOutcome] = {}
    agent_prompt_payloads: dict[str, dict[str, object]] = {}
    fake_llm_call_count = 0
    primary_prediction: Prediction | None = None
    primary_evaluation = None
    for behavior_index, behavior in enumerate(llm_behaviors):
        fake_llm = _BehaviorFakeLLM(behavior)
        agent_ranker = AgentRanker(
            hybrid_ranker=hybrid_ranker,
            toolbox=toolbox,
            llm=fake_llm,
        )
        output_dir = root / (
            "agent" if behavior == "reverse" else f"agent-{behavior}"
        )
        agent_result = run_agent_ranker(
            candidate_result.test_tasks_path,
            agent_ranker,
            output_dir,
        )
        agent_prediction = _load_prediction(Path(agent_result.predictions_path))
        evaluation = evaluate_prediction_file(
            candidate_result.test_tasks_path,
            candidate_result.ground_truth_path,
            agent_result.predictions_path,
        )
        agent_outcomes[behavior] = AgentOutcome(
            ranking=agent_prediction.ranking,
            fallback=agent_prediction.fallback,
            fallback_reason=agent_prediction.fallback_reason,
            task_count=evaluation.metrics.task_count,
            valid_output_rate=evaluation.metrics.valid_output_rate,
            llm_failure_rate=evaluation.metrics.llm_failure_rate,
        )
        if len(fake_llm.requests) != 1:
            raise AssertionError("Fake LLM did not receive exactly one request")
        agent_prompt_payloads[behavior] = json.loads(
            fake_llm.requests[0][-1].content
        )
        fake_llm_call_count += len(fake_llm.requests)
        if behavior_index == 0:
            primary_prediction = agent_prediction
            primary_evaluation = evaluation

    if primary_prediction is None or primary_evaluation is None:
        raise AssertionError("synthetic Agent scenarios produced no result")

    validation_tasks = [
        RecommendationTask.model_validate_json(line)
        for line in Path(candidate_result.validation_tasks_path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    test_tasks = [
        RecommendationTask.model_validate_json(line)
        for line in Path(candidate_result.test_tasks_path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    hybrid_score_snapshot = hybrid_ranker.score(test_tasks[0]).model_dump(
        mode="json"
    )
    return SyntheticPipelineRun(
        root=root,
        validation_task_count=len(validation_tasks),
        test_task_count=len(test_tasks),
        hybrid_ranking=hybrid_prediction.ranking,
        agent_ranking=primary_prediction.ranking,
        agent_fallback=primary_prediction.fallback,
        agent_tool_calls=primary_prediction.tool_calls,
        agent_evaluation_task_count=primary_evaluation.metrics.task_count,
        agent_valid_output_rate=primary_evaluation.metrics.valid_output_rate,
        fake_llm_call_count=fake_llm_call_count,
        hybrid_score_snapshot=hybrid_score_snapshot,
        agent_outcomes=agent_outcomes,
        agent_prompt_payloads=agent_prompt_payloads,
    )

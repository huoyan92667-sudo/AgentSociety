"""Load Step 34.5 planning facts from V1 inputs and an actual frozen Agent run."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.agent_benchmark import (
    load_scenario_ground_truth,
    load_visible_scenarios,
)
from yelp_agent.agent_benchmark.artifacts import sha256_file
from yelp_agent.agent_evaluation import load_agent_scenario_runs
from yelp_agent.features.location import haversine_km
from yelp_agent.models import StrictModel

from .schema import (
    ExpectedConditionDelta,
    FrozenPresentation,
    MemoryBenchmarkInitialSession,
    PlanningContext,
    PresentedBusinessSnapshot,
)


class BenchmarkV2Sources(StrictModel):
    benchmark_root: Path
    frozen_runs: Path
    businesses: Path
    pipeline_version: str = Field(min_length=1)

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path,
        *,
        pipeline_version: str,
    ) -> "BenchmarkV2Sources":
        root = Path(project_root)
        return cls(
            benchmark_root=root / "benchmarks" / "agent_scenarios_v1",
            frozen_runs=root / "runs" / "semantic_ranking_v1" / "llm_protected" / "full" / "scenario_runs.jsonl",
            businesses=root / "data" / "processed" / "businesses.parquet",
            pipeline_version=pipeline_version,
        )


def load_planning_contexts(sources: BenchmarkV2Sources) -> tuple[PlanningContext, ...]:
    """Join by scenario ID; hidden labels never enter the eventual visible bundle."""

    visible = {
        item.scenario_id: item
        for item in load_visible_scenarios(
            sources.benchmark_root / "visible" / "scenarios.jsonl"
        )
    }
    truths = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(
            sources.benchmark_root / "hidden" / "ground_truth.jsonl"
        )
    }
    runs = {item.scenario_id: item for item in load_agent_scenario_runs(sources.frozen_runs)}
    if set(visible) != set(truths) or set(visible) != set(runs):
        raise ValueError("V1 visible, hidden, and frozen-run scenario IDs must align")
    businesses = _business_catalog(sources.businesses)
    run_hash = sha256_file(sources.frozen_runs)
    contexts: list[PlanningContext] = []
    for scenario_id in sorted(visible):
        scenario = visible[scenario_id]
        truth = truths[scenario_id]
        run = runs[scenario_id]
        first = run.turns[0]
        ranking = list(first.candidate_ranking)
        candidate_facts = [
            _snapshot(
                businesses[business_id],
                business_id=business_id,
                rank=index,
                user_latitude=scenario.user_latitude,
                user_longitude=scenario.user_longitude,
            )
            for index, business_id in enumerate(ranking, start=1)
            if business_id in businesses
        ]
        facts_by_id = {item.business_id: item for item in candidate_facts}
        presented = [
            facts_by_id[business_id]
            for business_id in first.recommended_business_ids[:5]
            if business_id in facts_by_id
        ]
        session_case_id = _digest("memory-v2", scenario_id)
        initial = MemoryBenchmarkInitialSession(
            session_case_id=session_case_id,
            source_scenario_id=scenario_id,
            split=scenario.split,
            language=scenario.language,
            user_id=scenario.user_id,
            session_id=f"memory-v2:{session_case_id[:16]}",
            cutoff_time=scenario.cutoff_time,
            query_text=scenario.query_text,
            user_latitude=scenario.user_latitude,
            user_longitude=scenario.user_longitude,
            referenced_business_ids=scenario.referenced_business_ids,
        )
        presentation = None
        if presented:
            presentation = FrozenPresentation(
                session_case_id=session_case_id,
                turn_index=1,
                pipeline_version=sources.pipeline_version,
                source_run_sha256=run_hash,
                candidate_business_ids=[item.business_id for item in candidate_facts],
                presented_businesses=presented,
            )
        contexts.append(
            PlanningContext(
                initial_session=initial,
                source_category=truth.scenario_category,
                source_frame_family=truth.frame_family,
                initial_task_type=truth.task_type,
                initial_information_gaps=truth.expected_information_gaps,
                initial_conditions=[
                    ExpectedConditionDelta(
                        operation="add",
                        field=item.field,
                        operator=item.operator,
                        value=item.value,
                        importance=item.importance,
                    )
                    for item in truth.expected_conditions
                ],
                presentation=presentation,
                candidate_businesses=candidate_facts,
            )
        )
    return tuple(contexts)


def _business_catalog(path: Path) -> dict[str, dict[str, object]]:
    table = pq.read_table(
        path,
        columns=[
            "business_id",
            "name",
            "categories",
            "attributes_json",
            "latitude",
            "longitude",
        ],
    )
    rows = table.to_pylist()
    name_counts = Counter(str(item.get("name") or "").strip().casefold() for item in rows)
    result: dict[str, dict[str, object]] = {}
    for item in rows:
        business_id = str(item["business_id"])
        copied = dict(item)
        copied["is_chain"] = name_counts[str(item.get("name") or "").strip().casefold()] > 1
        result[business_id] = copied
    return result


def _snapshot(
    row: dict[str, object],
    *,
    business_id: str,
    rank: int,
    user_latitude: float | None,
    user_longitude: float | None,
) -> PresentedBusinessSnapshot:
    raw_attributes = row.get("attributes_json")
    attributes = json.loads(raw_attributes) if isinstance(raw_attributes, str) and raw_attributes else {}
    raw_price = attributes.get("RestaurantsPriceRange2")
    try:
        price = int(str(raw_price).strip("'\"")) if raw_price is not None else None
    except ValueError:
        price = None
    raw_noise = str(attributes.get("NoiseLevel") or "").strip("'\"").casefold().replace(" ", "_")
    noise = raw_noise if raw_noise in {"quiet", "average", "loud", "very_loud"} else None
    latitude, longitude = row.get("latitude"), row.get("longitude")
    distance = None
    if all(value is not None for value in (user_latitude, user_longitude, latitude, longitude)):
        distance = haversine_km(
            float(user_latitude), float(user_longitude), float(latitude), float(longitude)
        )
    raw_categories = row.get("categories")
    categories = (
        [part.strip() for part in raw_categories.split(",") if part.strip()]
        if isinstance(raw_categories, str)
        else [str(part).strip() for part in (raw_categories or []) if str(part).strip()]
    )
    return PresentedBusinessSnapshot(
        business_id=business_id,
        rank=rank,
        name=str(row.get("name") or business_id),
        categories=categories,
        price_level=price if price in {1, 2, 3, 4} else None,
        distance_km=distance,
        noise_level=noise,  # type: ignore[arg-type]
        is_chain=bool(row.get("is_chain")),
    )


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

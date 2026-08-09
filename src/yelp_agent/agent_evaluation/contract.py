"""Atomic freezing of the Step 21 evaluation contract manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from yelp_agent.agent_benchmark.artifacts import sha256_file
from yelp_agent.config import AgentEvaluationConfig
from yelp_agent.models import StrictModel

from .definitions import metric_definitions


class AgentEvaluationContractManifest(StrictModel):
    schema_version: Literal[1] = 1
    contract_version: Literal["1.0.0"]
    benchmark_version: Literal["1.0.0"]
    metric_count: int = Field(ge=1)
    metric_definitions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_artifact_sha256: dict[str, str]
    agent_scenario_metrics: list[str]
    full_retrieval_metrics: list[str]
    percentile_method: Literal["linear"]
    missing_observation_policy: Literal["explicit_status"]
    hidden_labels_visible_to_agent: Literal[False] = False
    full_retrieval_inferred_from_agent_scenarios: Literal[False] = False
    frozen: Literal[True] = True


class AgentEvaluationContractResult(StrictModel):
    status: Literal["written", "reused"]
    output_path: str
    manifest: AgentEvaluationContractManifest


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_manifest(
    benchmark_root: Path,
    config: AgentEvaluationConfig,
) -> AgentEvaluationContractManifest:
    paths = {
        "manifest": benchmark_root / "manifest.json",
        "visible": benchmark_root / "visible" / "scenarios.jsonl",
        "ground_truth": benchmark_root / "hidden" / "ground_truth.jsonl",
        "evidence": benchmark_root / "hidden" / "evidence_labels.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Step 20 benchmark artifacts are missing: {missing}")
    benchmark_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if benchmark_manifest.get("benchmark_version") != config.benchmark_version:
        raise ValueError("Step 20 benchmark and Step 21 contract versions disagree")
    definitions = metric_definitions()
    serialized_definitions = [item.model_dump(mode="json") for item in definitions]
    full_retrieval_metrics = [
        f"recall_at_{cutoff}" for cutoff in config.full_retrieval_cutoffs
    ]
    return AgentEvaluationContractManifest(
        contract_version=config.contract_version,
        benchmark_version=config.benchmark_version,
        metric_count=len(definitions),
        metric_definitions_sha256=_stable_sha256(serialized_definitions),
        configuration_sha256=_stable_sha256(config.model_dump(mode="json")),
        benchmark_artifact_sha256={
            name: sha256_file(path) for name, path in sorted(paths.items())
        },
        agent_scenario_metrics=[
            item.name for item in definitions if item.source == "agent_scenario"
        ],
        full_retrieval_metrics=full_retrieval_metrics,
        percentile_method=config.percentile_method,
        missing_observation_policy=config.missing_observation_policy,
    )


def freeze_agent_evaluation_contract(
    benchmark_root: str | Path,
    config: AgentEvaluationConfig,
    *,
    output_path: str | Path,
) -> AgentEvaluationContractResult:
    """Freeze or byte-safely reuse the evaluation definitions for one benchmark."""

    root = Path(benchmark_root)
    output = Path(output_path)
    expected = _expected_manifest(root, config)
    if output.is_file():
        existing = AgentEvaluationContractManifest.model_validate_json(
            output.read_text(encoding="utf-8")
        )
        if existing != expected:
            raise ValueError(
                "existing Step 21 contract differs; write a new versioned artifact"
            )
        return AgentEvaluationContractResult(
            status="reused",
            output_path=str(output),
            manifest=existing,
        )
    if output.exists():
        raise FileExistsError(f"contract output is not a file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text(
            expected.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return AgentEvaluationContractResult(
        status="written",
        output_path=str(output),
        manifest=expected,
    )

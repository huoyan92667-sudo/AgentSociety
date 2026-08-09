"""Atomic, hash-verified publication of Step 20 visible and hidden artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.config import AgentBenchmarkConfig
from yelp_agent.models import StrictModel

from .audit import AgentBenchmarkAuditReport, audit_agent_benchmark_bundle
from .builder import AgentBenchmarkBundle, build_agent_benchmark_bundle
from .rewriting import ScenarioRewriter
from .schema import (
    EVIDENCE_LABEL_SCHEMA,
    EvidenceLabel,
    ScenarioGroundTruth,
    VisibleAgentScenario,
)
from .sources import AgentBenchmarkSources, load_benchmark_catalog


class AgentBenchmarkManifest(StrictModel):
    schema_version: Literal[1] = 1
    benchmark_version: Literal["1.0.0"]
    scenario_count: int = Field(ge=1)
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    language_counts: dict[str, int]
    source_sha256: dict[str, str]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: dict[str, str]
    generator_kinds: list[str]
    contains_test_source_tasks: Literal[False] = False
    hidden_labels_visible_to_agent: Literal[False] = False
    llm_determined_hidden_labels: Literal[False] = False


class AgentBenchmarkBuildResult(StrictModel):
    status: Literal["written", "reused"]
    root: str
    visible_path: str
    ground_truth_path: str
    evidence_path: str
    manifest_path: str
    audit_path: str
    manifest: AgentBenchmarkManifest
    audit: AgentBenchmarkAuditReport


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(config: AgentBenchmarkConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_hashes(sources: AgentBenchmarkSources) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in (
            ("businesses", sources.businesses),
            ("rating_events", sources.rating_events),
            ("aspect_records", sources.aspect_records),
            ("profile_snapshots", sources.profile_snapshots),
            ("preference_signals", sources.preference_signals),
            ("task_profile_map", sources.task_profile_map),
            ("query_benchmark", sources.query_benchmark),
        )
    }


def _jsonl(values: list[StrictModel]) -> str:
    return "".join(
        item.model_dump_json() + "\n"
        for item in sorted(values, key=lambda item: getattr(item, "scenario_id"))
    )


def _write_bundle(root: Path, bundle: AgentBenchmarkBundle) -> None:
    visible_path = root / "visible" / "scenarios.jsonl"
    truth_path = root / "hidden" / "ground_truth.jsonl"
    evidence_path = root / "hidden" / "evidence_labels.parquet"
    visible_path.parent.mkdir(parents=True)
    truth_path.parent.mkdir(parents=True)
    visible_path.write_text(
        _jsonl(list(bundle.visible_scenarios)),
        encoding="utf-8",
        newline="\n",
    )
    truth_path.write_text(
        _jsonl(list(bundle.ground_truth)),
        encoding="utf-8",
        newline="\n",
    )
    evidence_rows = [
        item.model_dump()
        for item in sorted(
            bundle.evidence_labels,
            key=lambda item: (
                item.scenario_id,
                item.business_id,
                item.review_id or "",
                item.source_field or "",
                item.aspect or "",
            ),
        )
    ]
    pq.write_table(
        pa.Table.from_pylist(evidence_rows, schema=EVIDENCE_LABEL_SCHEMA),
        evidence_path,
        compression="zstd",
        write_statistics=True,
    )


def load_visible_scenarios(path: str | Path) -> tuple[VisibleAgentScenario, ...]:
    values = [
        VisibleAgentScenario.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(sorted(values, key=lambda item: item.scenario_id))


def load_scenario_ground_truth(path: str | Path) -> tuple[ScenarioGroundTruth, ...]:
    values = [
        ScenarioGroundTruth.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(sorted(values, key=lambda item: item.scenario_id))


def load_evidence_labels(path: str | Path) -> tuple[EvidenceLabel, ...]:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != EVIDENCE_LABEL_SCHEMA:
        raise ValueError("Step 20 evidence-label schema mismatch")
    return tuple(
        EvidenceLabel.model_validate(row) for row in parquet.read().to_pylist()
    )


def _bundle_from_root(root: Path) -> AgentBenchmarkBundle:
    return AgentBenchmarkBundle(
        visible_scenarios=load_visible_scenarios(root / "visible" / "scenarios.jsonl"),
        ground_truth=load_scenario_ground_truth(root / "hidden" / "ground_truth.jsonl"),
        evidence_labels=load_evidence_labels(root / "hidden" / "evidence_labels.parquet"),
    )


def build_agent_benchmark(
    sources: AgentBenchmarkSources,
    config: AgentBenchmarkConfig,
    *,
    output_root: str | Path,
    rewriter: ScenarioRewriter | None = None,
) -> AgentBenchmarkBuildResult:
    """Build, audit, freeze, and safely reuse all Step 20 artifacts."""

    root = Path(output_root)
    visible_path = root / "visible" / "scenarios.jsonl"
    truth_path = root / "hidden" / "ground_truth.jsonl"
    evidence_path = root / "hidden" / "evidence_labels.parquet"
    manifest_path = root / "manifest.json"
    audit_path = root / "audit_report.json"
    required = (visible_path, truth_path, evidence_path, manifest_path, audit_path)
    source_hashes = _source_hashes(sources)
    config_hash = _configuration_sha256(config)
    if all(path.is_file() for path in required):
        manifest = AgentBenchmarkManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.source_sha256 != source_hashes:
            raise ValueError("Step 20 source data changed; rebuild into a new root")
        if manifest.configuration_sha256 != config_hash:
            raise ValueError("Step 20 config changed; rebuild into a new root")
        current_hashes = {
            "visible": sha256_file(visible_path),
            "ground_truth": sha256_file(truth_path),
            "evidence": sha256_file(evidence_path),
            "audit": sha256_file(audit_path),
        }
        if manifest.output_sha256 != current_hashes:
            raise ValueError("Step 20 output hash verification failed")
        bundle = _bundle_from_root(root)
        audit = audit_agent_benchmark_bundle(bundle, config)
        return AgentBenchmarkBuildResult(
            status="reused",
            root=str(root),
            visible_path=str(visible_path),
            ground_truth_path=str(truth_path),
            evidence_path=str(evidence_path),
            manifest_path=str(manifest_path),
            audit_path=str(audit_path),
            manifest=manifest,
            audit=audit,
        )
    if any(path.exists() for path in required) or root.exists():
        raise FileExistsError("Step 20 output root is incomplete; do not mix runs")

    catalog = load_benchmark_catalog(sources, config)
    bundle = build_agent_benchmark_bundle(catalog, config, rewriter=rewriter)
    audit = audit_agent_benchmark_bundle(bundle, config)
    partial = root.with_name(root.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    try:
        _write_bundle(partial, bundle)
        partial_audit = partial / "audit_report.json"
        partial_audit.write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
        output_hashes = {
            "visible": sha256_file(partial / "visible" / "scenarios.jsonl"),
            "ground_truth": sha256_file(partial / "hidden" / "ground_truth.jsonl"),
            "evidence": sha256_file(partial / "hidden" / "evidence_labels.parquet"),
            "audit": sha256_file(partial_audit),
        }
        manifest = AgentBenchmarkManifest(
            benchmark_version=config.benchmark_version,
            scenario_count=len(bundle.visible_scenarios),
            split_counts=dict(sorted(Counter(item.split for item in bundle.visible_scenarios).items())),
            category_counts=dict(
                sorted(Counter(item.scenario_category for item in bundle.ground_truth).items())
            ),
            language_counts=dict(
                sorted(Counter(item.language for item in bundle.visible_scenarios).items())
            ),
            source_sha256=source_hashes,
            configuration_sha256=config_hash,
            output_sha256=output_hashes,
            generator_kinds=sorted({item.generator_kind for item in bundle.visible_scenarios}),
        )
        (partial / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, root)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return AgentBenchmarkBuildResult(
        status="written",
        root=str(root),
        visible_path=str(visible_path),
        ground_truth_path=str(truth_path),
        evidence_path=str(evidence_path),
        manifest_path=str(manifest_path),
        audit_path=str(audit_path),
        manifest=manifest,
        audit=audit,
    )

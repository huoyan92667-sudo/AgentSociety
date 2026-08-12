"""Atomic visible/hidden publication for Query recommendation cases."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from yelp_agent.models import StrictModel

from .schema import (
    QueryRecommendationBenchmarkBundle,
    QueryRecommendationFrame,
    QueryRecommendationGroundTruth,
    VisibleQueryRecommendationCase,
)
from .audit import QueryRecommendationAuditReport
from .generation import QueryGenerationReport


class QueryRecommendationBenchmarkManifest(StrictModel):
    schema_version: Literal[1] = 1
    benchmark_version: Literal["1.0.0"] = "1.0.0"
    case_count: int = Field(ge=1)
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    output_sha256: dict[str, str]
    source_sha256: dict[str, str] = Field(default_factory=dict)
    configuration_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    generator_model: str | None = None
    hidden_labels_visible_to_agent: Literal[False] = False
    llm_determined_hidden_labels: Literal[False] = False
    contains_test_source_tasks: Literal[False] = False
    aspect_source_scope: Literal["selected_user_interactions"] = (
        "selected_user_interactions"
    )


@dataclass(frozen=True, slots=True)
class QueryRecommendationBenchmarkBuildResult:
    root: Path
    visible_path: Path
    ground_truth_path: Path
    frames_path: Path
    manifest_path: Path
    audit_path: Path | None
    generation_report_path: Path | None
    manifest: QueryRecommendationBenchmarkManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(values: tuple[StrictModel, ...]) -> str:
    return "".join(
        item.model_dump_json() + "\n"
        for item in sorted(values, key=lambda value: getattr(value, "case_id"))
    )


def _load_jsonl(path: Path, model: type[StrictModel]) -> tuple[StrictModel, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def publish_query_recommendation_bundle(
    bundle: QueryRecommendationBenchmarkBundle,
    output_root: str | Path,
    *,
    audit: QueryRecommendationAuditReport | None = None,
    generation_report: QueryGenerationReport | None = None,
    source_sha256: dict[str, str] | None = None,
    configuration_sha256: str | None = None,
) -> QueryRecommendationBenchmarkBuildResult:
    """Publish one immutable bundle; never mix old and new benchmark files."""

    root = Path(output_root)
    if root.exists():
        raise FileExistsError("benchmark output root already exists")
    partial = root.with_name(root.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    try:
        visible = partial / "visible" / "cases.jsonl"
        truth = partial / "hidden" / "ground_truth.jsonl"
        frames = partial / "hidden" / "structured_frames.jsonl"
        manifest_path = partial / "manifest.json"
        visible.parent.mkdir(parents=True)
        truth.parent.mkdir(parents=True)
        visible.write_text(_jsonl(bundle.visible_cases), encoding="utf-8", newline="\n")
        truth.write_text(_jsonl(bundle.ground_truth), encoding="utf-8", newline="\n")
        frames.write_text(_jsonl(bundle.frames), encoding="utf-8", newline="\n")
        hashes = {
            "visible": _sha256(visible),
            "ground_truth": _sha256(truth),
            "structured_frames": _sha256(frames),
        }
        if audit is not None:
            audit_path = partial / "audit" / "leakage_audit.json"
            audit_path.parent.mkdir(parents=True)
            audit_path.write_text(
                audit.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            hashes["leakage_audit"] = _sha256(audit_path)
        if generation_report is not None:
            generation_path = partial / "audit" / "generation_report.json"
            generation_path.parent.mkdir(parents=True, exist_ok=True)
            generation_path.write_text(
                generation_report.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            hashes["generation_report"] = _sha256(generation_path)
        manifest = QueryRecommendationBenchmarkManifest(
            case_count=len(bundle.visible_cases),
            split_counts=dict(
                sorted(Counter(item.split for item in bundle.visible_cases).items())
            ),
            language_counts=dict(
                sorted(Counter(item.language for item in bundle.visible_cases).items())
            ),
            output_sha256=hashes,
            source_sha256=source_sha256 or {},
            configuration_sha256=configuration_sha256,
            generator_model=(
                None if generation_report is None else generation_report.generator_model
            ),
        )
        manifest_path.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, root)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return QueryRecommendationBenchmarkBuildResult(
        root=root,
        visible_path=root / "visible" / "cases.jsonl",
        ground_truth_path=root / "hidden" / "ground_truth.jsonl",
        frames_path=root / "hidden" / "structured_frames.jsonl",
        manifest_path=root / "manifest.json",
        audit_path=(
            None if audit is None else root / "audit" / "leakage_audit.json"
        ),
        generation_report_path=(
            None
            if generation_report is None
            else root / "audit" / "generation_report.json"
        ),
        manifest=manifest,
    )


def load_query_recommendation_bundle(
    root: str | Path,
) -> QueryRecommendationBenchmarkBundle:
    source = Path(root)
    manifest = QueryRecommendationBenchmarkManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    paths = {
        "visible": source / "visible" / "cases.jsonl",
        "ground_truth": source / "hidden" / "ground_truth.jsonl",
        "structured_frames": source / "hidden" / "structured_frames.jsonl",
    }
    if "leakage_audit" in manifest.output_sha256:
        paths["leakage_audit"] = source / "audit" / "leakage_audit.json"
    if "generation_report" in manifest.output_sha256:
        paths["generation_report"] = source / "audit" / "generation_report.json"
    if manifest.output_sha256 != {name: _sha256(path) for name, path in paths.items()}:
        raise ValueError("benchmark output hash verification failed")
    return QueryRecommendationBenchmarkBundle(
        visible_cases=_load_jsonl(paths["visible"], VisibleQueryRecommendationCase),
        ground_truth=_load_jsonl(
            paths["ground_truth"], QueryRecommendationGroundTruth
        ),
        frames=_load_jsonl(paths["structured_frames"], QueryRecommendationFrame),
    )

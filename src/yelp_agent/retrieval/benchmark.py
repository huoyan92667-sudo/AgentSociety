"""Build deterministic, label-free full retrieval benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from yelp_agent.config import AppConfig, RetrievalConfig
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.experiments.artifacts import write_json_artifact
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.location import TemporalLocationStore
from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.features.text import TemporalTextStore
from yelp_agent.models import StrictModel
from yelp_agent.retrieval.multi_route import (
    MultiRouteRetriever,
    RetrievalResult,
    RetrievalTaskContext,
)

CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("rank", pa.int32(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("fusion_score", pa.float64(), nullable=False),
        pa.field("route_count", pa.int8(), nullable=False),
        pa.field("quality_rank", pa.int32()),
        pa.field("quality_score", pa.float64()),
        pa.field("category_rank", pa.int32()),
        pa.field("category_score", pa.float64()),
        pa.field("text_rank", pa.int32()),
        pa.field("text_score", pa.float64()),
        pa.field("location_rank", pa.int32()),
        pa.field("location_score", pa.float64()),
        pa.field("distance_km", pa.float64()),
    ]
)
TASK_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("catalog_size", pa.int32(), nullable=False),
        pa.field("eligible_candidate_count", pa.int32(), nullable=False),
        pa.field("excluded_history_businesses", pa.int32(), nullable=False),
        pa.field("candidate_count", pa.int32(), nullable=False),
        pa.field("quality_result_count", pa.int32(), nullable=False),
        pa.field("category_result_count", pa.int32(), nullable=False),
        pa.field("text_result_count", pa.int32(), nullable=False),
        pa.field("location_result_count", pa.int32(), nullable=False),
        pa.field("latency_ms", pa.float64(), nullable=False),
    ]
)
ROUTE_PROVENANCE_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("route", pa.string(), nullable=False),
        pa.field("route_rank", pa.int32(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("route_score", pa.float64(), nullable=False),
    ]
)


class RetrievalBenchmarkError(RuntimeError):
    """Raised when benchmark inputs or frozen outputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class RetrievalSourcePaths:
    businesses: Path
    reviews: Path
    interactions: Path
    tfidf_artifact: Path
    tfidf_manifest: Path


class RetrievalSplitManifest(StrictModel):
    task_count: int = Field(ge=1)
    candidate_rows: int = Field(ge=1)
    minimum_candidates: int = Field(ge=1)
    maximum_candidates: int = Field(ge=1)
    provenance_rows: int = Field(ge=0)


class RetrievalBenchmarkManifest(StrictModel):
    format_version: Literal[1] = 1
    benchmark_name: Literal["Full Retrieval Benchmark V1"] = (
        "Full Retrieval Benchmark V1"
    )
    target_conditioned: Literal[False] = False
    ground_truth_files_read: Literal[False] = False
    retrieval_configuration: RetrievalConfig
    broad_categories: list[str]
    requested_splits: list[Literal["train", "validation", "test"]]
    source_sha256: dict[str, str]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: dict[str, str]
    splits: dict[str, RetrievalSplitManifest]


class RetrievalBenchmarkBuildResult(StrictModel):
    status: Literal["written", "skipped"]
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root: str
    splits: dict[str, RetrievalSplitManifest]
    output_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(
    config: RetrievalConfig,
    broad_categories: list[str],
) -> str:
    payload = {
        "retrieval": config.model_dump(mode="json"),
        "broad_categories": broad_categories,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _artifact_paths(
    output_root: Path,
    splits: list[str],
    provenance_splits: set[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split in splits:
        paths[f"{split}_candidates"] = (
            output_root / f"{split}_candidates.parquet"
        )
        paths[f"{split}_task_audit"] = (
            output_root / f"{split}_task_audit.parquet"
        )
        if split in provenance_splits:
            paths[f"{split}_route_provenance"] = (
                output_root / f"{split}_route_provenance.parquet"
            )
    return paths


def _load_contexts(path: Path, split: str) -> list[RetrievalTaskContext]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Retrieval context Parquet does not exist: {path}"
        )
    try:
        rows = pq.read_table(
            path,
            columns=[
                "task_id",
                "split",
                "user_id",
                "cutoff_time",
                "history_count",
            ],
        ).to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise RetrievalBenchmarkError(
            f"Could not read retrieval contexts from {path}: {exc}"
        ) from exc
    selected = [
        RetrievalTaskContext.model_validate(row)
        for row in rows
        if row.get("split") == split
    ]
    selected.sort(key=lambda task: task.task_id)
    if not selected:
        raise RetrievalBenchmarkError(
            f"No {split!r} contexts were found in {path}"
        )
    task_ids = [task.task_id for task in selected]
    if len(task_ids) != len(set(task_ids)):
        raise RetrievalBenchmarkError(f"Duplicate {split!r} retrieval task_id")
    return selected


def _candidate_rows(result: RetrievalResult) -> list[dict[str, object]]:
    return [
        {
            "task_id": result.task.task_id,
            "rank": candidate.rank,
            "business_id": candidate.business_id,
            "fusion_score": candidate.fusion_score,
            "route_count": candidate.route_count,
            "quality_rank": candidate.quality_rank,
            "quality_score": candidate.quality_score,
            "category_rank": candidate.category_rank,
            "category_score": candidate.category_score,
            "text_rank": candidate.text_rank,
            "text_score": candidate.text_score,
            "location_rank": candidate.location_rank,
            "location_score": candidate.location_score,
            "distance_km": candidate.distance_km,
        }
        for candidate in result.candidates
    ]


def _audit_row(result: RetrievalResult) -> dict[str, object]:
    return {
        "task_id": result.task.task_id,
        "split": result.task.split,
        "catalog_size": result.catalog_size,
        "eligible_candidate_count": result.eligible_candidate_count,
        "excluded_history_businesses": result.excluded_history_businesses,
        "candidate_count": len(result.candidates),
        "quality_result_count": result.route_result_counts["quality"],
        "category_result_count": result.route_result_counts["category"],
        "text_result_count": result.route_result_counts["text"],
        "location_result_count": result.route_result_counts["location"],
        "latency_ms": result.latency_ms,
    }


def _provenance_rows(result: RetrievalResult) -> list[dict[str, object]]:
    return [
        {
            "task_id": result.task.task_id,
            "route": candidate.route,
            "route_rank": candidate.rank,
            "business_id": candidate.business_id,
            "route_score": candidate.score,
        }
        for candidate in result.route_candidates
    ]


def _write_split(
    split: str,
    tasks: list[RetrievalTaskContext],
    retriever: MultiRouteRetriever,
    paths: dict[str, Path],
    *,
    write_provenance: bool,
) -> RetrievalSplitManifest:
    candidate_path = paths[f"{split}_candidates"]
    audit_path = paths[f"{split}_task_audit"]
    provenance_path = paths.get(f"{split}_route_provenance")
    candidate_partial = candidate_path.with_name(candidate_path.name + ".partial")
    audit_partial = audit_path.with_name(audit_path.name + ".partial")
    provenance_partial = (
        None
        if provenance_path is None
        else provenance_path.with_name(provenance_path.name + ".partial")
    )
    partials = [candidate_partial, audit_partial]
    if provenance_partial is not None:
        partials.append(provenance_partial)
    for partial in partials:
        partial.unlink(missing_ok=True)

    candidate_writer = pq.ParquetWriter(
        candidate_partial,
        CANDIDATE_SCHEMA,
        compression="zstd",
        use_dictionary=["task_id", "business_id"],
    )
    audit_writer = pq.ParquetWriter(
        audit_partial,
        TASK_AUDIT_SCHEMA,
        compression="zstd",
        use_dictionary=["task_id", "split"],
    )
    provenance_writer = (
        pq.ParquetWriter(
            provenance_partial,
            ROUTE_PROVENANCE_SCHEMA,
            compression="zstd",
            use_dictionary=["task_id", "route", "business_id"],
        )
        if write_provenance and provenance_partial is not None
        else None
    )
    candidate_count_per_task: list[int] = []
    candidate_rows_written = 0
    provenance_rows_written = 0
    candidate_buffer: list[dict[str, object]] = []
    audit_buffer: list[dict[str, object]] = []
    provenance_buffer: list[dict[str, object]] = []

    def flush() -> None:
        if candidate_buffer:
            candidate_writer.write_table(
                pa.Table.from_pylist(candidate_buffer, schema=CANDIDATE_SCHEMA)
            )
            candidate_buffer.clear()
        if audit_buffer:
            audit_writer.write_table(
                pa.Table.from_pylist(audit_buffer, schema=TASK_AUDIT_SCHEMA)
            )
            audit_buffer.clear()
        if provenance_writer is not None and provenance_buffer:
            provenance_writer.write_table(
                pa.Table.from_pylist(
                    provenance_buffer,
                    schema=ROUTE_PROVENANCE_SCHEMA,
                )
            )
            provenance_buffer.clear()

    try:
        for task in tasks:
            result = retriever.retrieve(
                task,
                include_route_provenance=write_provenance,
            )
            candidates = _candidate_rows(result)
            candidate_buffer.extend(candidates)
            audit_buffer.append(_audit_row(result))
            if provenance_writer is not None:
                provenance = _provenance_rows(result)
                provenance_buffer.extend(provenance)
                provenance_rows_written += len(provenance)
            candidate_count_per_task.append(len(candidates))
            candidate_rows_written += len(candidates)
            if len(audit_buffer) >= 25:
                flush()
        flush()
    except Exception:
        candidate_writer.close()
        audit_writer.close()
        if provenance_writer is not None:
            provenance_writer.close()
        for partial in partials:
            partial.unlink(missing_ok=True)
        raise
    candidate_writer.close()
    audit_writer.close()
    if provenance_writer is not None:
        provenance_writer.close()

    os.replace(candidate_partial, candidate_path)
    os.replace(audit_partial, audit_path)
    if provenance_partial is not None and provenance_path is not None:
        os.replace(provenance_partial, provenance_path)
    return RetrievalSplitManifest(
        task_count=len(tasks),
        candidate_rows=candidate_rows_written,
        minimum_candidates=min(candidate_count_per_task),
        maximum_candidates=max(candidate_count_per_task),
        provenance_rows=provenance_rows_written,
    )


def _load_manifest(path: Path) -> RetrievalBenchmarkManifest:
    try:
        return RetrievalBenchmarkManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise RetrievalBenchmarkError(
            f"Retrieval benchmark manifest is invalid: {path}"
        ) from exc


def _result(
    status: Literal["written", "skipped"],
    output_root: Path,
    manifest_path: Path,
    manifest: RetrievalBenchmarkManifest,
) -> RetrievalBenchmarkBuildResult:
    return RetrievalBenchmarkBuildResult(
        status=status,
        manifest_path=str(manifest_path),
        manifest_sha256=_sha256(manifest_path),
        output_root=str(output_root),
        splits=manifest.splits,
        output_sha256=manifest.output_sha256,
    )


def build_full_retrieval_benchmark(
    sources: RetrievalSourcePaths,
    split_contexts: Mapping[str, str | Path],
    output_root: str | Path,
    app_config: AppConfig,
    retrieval_config: RetrievalConfig,
    *,
    force: bool = False,
) -> RetrievalBenchmarkBuildResult:
    """Build candidate files without accepting or reading ground-truth paths."""

    if not split_contexts:
        raise ValueError("split_contexts cannot be empty")
    requested_splits = sorted(split_contexts)
    allowed_splits = {"train", "validation", "test"}
    if not set(requested_splits).issubset(allowed_splits):
        raise ValueError("split_contexts contains an unsupported split")

    source_files = {
        "businesses": sources.businesses,
        "reviews": sources.reviews,
        "interactions": sources.interactions,
        "tfidf_artifact": sources.tfidf_artifact,
        "tfidf_manifest": sources.tfidf_manifest,
        **{
            f"{split}_contexts": Path(path)
            for split, path in split_contexts.items()
        },
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Retrieval source files do not exist: " + ", ".join(missing)
        )
    source_sha256 = {
        name: _sha256(path) for name, path in sorted(source_files.items())
    }
    broad_categories = sorted(app_config.data.broad_categories)
    configuration_sha256 = _configuration_sha256(
        retrieval_config,
        broad_categories,
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    paths = _artifact_paths(
        root,
        requested_splits,
        set(retrieval_config.provenance_splits),
    )

    if manifest_path.exists() and not force:
        manifest = _load_manifest(manifest_path)
        if (
            manifest.requested_splits != requested_splits
            or manifest.source_sha256 != source_sha256
            or manifest.configuration_sha256 != configuration_sha256
            or manifest.retrieval_configuration != retrieval_config
            or manifest.broad_categories != broad_categories
        ):
            raise RetrievalBenchmarkError(
                "Retrieval benchmark inputs or configuration changed; "
                "use force=True to rebuild"
            )
        for name, path in paths.items():
            if (
                not path.is_file()
                or _sha256(path) != manifest.output_sha256.get(name)
            ):
                raise RetrievalBenchmarkError(
                    f"Retrieval benchmark output changed or is missing: {name}"
                )
        return _result("skipped", root, manifest_path, manifest)
    if any(path.exists() for path in paths.values()) and not force:
        raise RetrievalBenchmarkError(
            "Retrieval benchmark outputs are incomplete; use force=True to rebuild"
        )

    data_view = TemporalDataView(
        sources.businesses,
        sources.reviews,
        sources.interactions,
    )
    retriever = MultiRouteRetriever(
        data_view,
        category_store=TemporalCategoryStore(
            data_view,
            broad_categories=set(app_config.data.broad_categories),
        ),
        text_store=TemporalTextStore(
            data_view,
            sources.tfidf_artifact,
            sources.tfidf_manifest,
        ),
        quality_store=TemporalQualityStore(
            data_view,
            prior_count=retrieval_config.bayesian_prior_count,
        ),
        location_store=TemporalLocationStore(
            data_view,
            scale_km=retrieval_config.location_scale_km,
        ),
        config=retrieval_config,
    )
    split_manifests: dict[str, RetrievalSplitManifest] = {}
    for split in requested_splits:
        tasks = _load_contexts(Path(split_contexts[split]), split)
        split_manifests[split] = _write_split(
            split,
            tasks,
            retriever,
            paths,
            write_provenance=split in retrieval_config.provenance_splits,
        )
    output_sha256 = {
        name: _sha256(path) for name, path in sorted(paths.items())
    }
    manifest = RetrievalBenchmarkManifest(
        retrieval_configuration=retrieval_config,
        broad_categories=broad_categories,
        requested_splits=requested_splits,
        source_sha256=source_sha256,
        configuration_sha256=configuration_sha256,
        output_sha256=output_sha256,
        splits=split_manifests,
    )
    write_json_artifact(manifest_path, manifest)
    return _result("written", root, manifest_path, manifest)

"""Audit and atomically freeze the visible/hidden Benchmark V2 bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from pydantic import Field

from yelp_agent.models import StrictModel

from .config import SessionMemoryBenchmarkV2Config
from .generation import TurnGenerationResult
from .schema import (
    BenchmarkGenerationPlan,
    BenchmarkV2Manifest,
    FrozenPresentation,
    FrozenScriptedTurnV2,
    FrozenTurnGroundTruthV2,
    MemoryBenchmarkInitialSession,
)


class BenchmarkV2AuditReport(StrictModel):
    passed: bool
    session_count: int = Field(ge=0)
    turn_count: int = Field(ge=0)
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    family_counts: dict[str, int]
    exact_duplicate_count: int = Field(ge=0)
    cross_split_duplicate_count: int = Field(ge=0)
    near_duplicate_count: int = Field(ge=0)
    leaked_business_id_count: int = Field(ge=0)
    relative_numeric_leak_count: int = Field(ge=0)
    invalid_session_sequence_count: int = Field(ge=0)
    issues: list[str] = Field(default_factory=list)


class BenchmarkV2BuildResult(StrictModel):
    benchmark_root: Path
    visible_sessions_path: Path
    visible_turns_path: Path
    presentations_path: Path
    hidden_ground_truth_path: Path
    manifest_path: Path
    audit_path: Path
    generation_calls_path: Path
    manifest: BenchmarkV2Manifest
    audit: BenchmarkV2AuditReport


def freeze_benchmark_v2(
    *,
    plan: BenchmarkGenerationPlan,
    generated: TurnGenerationResult,
    config: SessionMemoryBenchmarkV2Config,
    output_root: str | Path,
    run_output_root: str | Path,
) -> BenchmarkV2BuildResult:
    audit = audit_benchmark_v2(plan, generated.turns, generated.ground_truth, config)
    if not audit.passed:
        raise ValueError("Benchmark V2 audit failed: " + "; ".join(audit.issues[:8]))
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"Benchmark V2 output already exists: {root}")
    partial = root.with_name(root.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    sessions_path = partial / "visible" / "initial_sessions.jsonl"
    turns_path = partial / "visible" / "scripted_turns.jsonl"
    presentations_path = partial / "visible" / "frozen_presentations.jsonl"
    truth_path = partial / "hidden" / "ground_truth.jsonl"
    audit_path = partial / "audit_report.json"
    manifest_path = partial / "manifest.json"
    try:
        _write_models(sessions_path, plan.initial_sessions, "session_case_id")
        _write_models(turns_path, generated.turns, "turn_case_id")
        _write_models(presentations_path, plan.presentations, "session_case_id")
        _write_models(truth_path, generated.ground_truth, "turn_case_id")
        _write_text(audit_path, audit.model_dump_json(indent=2) + "\n")
        output_hashes = {
            "visible_initial_sessions": _sha256_file(sessions_path),
            "visible_scripted_turns": _sha256_file(turns_path),
            "visible_frozen_presentations": _sha256_file(presentations_path),
            "hidden_ground_truth": _sha256_file(truth_path),
            "audit_report": _sha256_file(audit_path),
        }
        source_hashes = {item.source_run_sha256 for item in plan.presentations}
        if len(source_hashes) != 1:
            raise ValueError("all frozen presentations must originate from one frozen run")
        manifest = BenchmarkV2Manifest(
            benchmark_version=config.benchmark_version,
            session_count=len(plan.initial_sessions),
            turn_count=len(generated.turns),
            split_counts=audit.split_counts,
            language_counts=audit.language_counts,
            family_counts=audit.family_counts,
            generator_model=generated.turns[0].generator_model,
            reviewer_model=generated.turns[0].reviewer_model,
            source_run_sha256=next(iter(source_hashes)),
            output_sha256=output_hashes,
            generation_input_tokens=generated.generation_input_tokens,
            generation_output_tokens=generated.generation_output_tokens,
            review_input_tokens=generated.review_input_tokens,
            review_output_tokens=generated.review_output_tokens,
            provider_call_count=generated.provider_call_count,
            rejected_generation_count=generated.rejected_generation_count,
        )
        _write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, root)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    run_root = Path(run_output_root)
    calls_path = run_root / "generation" / "provider_calls.jsonl"
    _write_text(
        calls_path,
        "".join(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n" for item in generated.calls),
    )
    return BenchmarkV2BuildResult(
        benchmark_root=root,
        visible_sessions_path=root / "visible" / "initial_sessions.jsonl",
        visible_turns_path=root / "visible" / "scripted_turns.jsonl",
        presentations_path=root / "visible" / "frozen_presentations.jsonl",
        hidden_ground_truth_path=root / "hidden" / "ground_truth.jsonl",
        manifest_path=root / "manifest.json",
        audit_path=root / "audit_report.json",
        generation_calls_path=calls_path,
        manifest=manifest,
        audit=audit,
    )


def audit_benchmark_v2(
    plan: BenchmarkGenerationPlan,
    turns: Sequence[FrozenScriptedTurnV2],
    truths: Sequence[FrozenTurnGroundTruthV2],
    config: SessionMemoryBenchmarkV2Config,
) -> BenchmarkV2AuditReport:
    issues: list[str] = []
    truth_by_id = {item.turn_case_id: item for item in truths}
    if len(truth_by_id) != len(truths) or set(truth_by_id) != {item.turn_case_id for item in turns}:
        issues.append("visible and hidden turn IDs do not form a bijection")
    sessions = {item.session_case_id: item for item in plan.initial_sessions}
    presentation_by_session = {item.session_case_id: item for item in plan.presentations}
    normalized = [_normalize(item.query_text) for item in turns]
    exact_duplicates = len(normalized) - len(set(normalized))
    if exact_duplicates:
        issues.append(f"found {exact_duplicates} exact duplicate messages")
    text_splits: dict[str, set[str]] = defaultdict(set)
    for item, text in zip(turns, normalized, strict=True):
        text_splits[text].add(item.split)
    cross_split = sum(len(values) > 1 for values in text_splits.values())
    if cross_split:
        issues.append(f"found {cross_split} messages shared across splits")
    near_duplicates = _near_duplicate_count(normalized, config.duplicate_similarity_threshold)
    if near_duplicates:
        issues.append(f"found {near_duplicates} near-duplicate message pairs")
    leaked_ids = 0
    numeric_leaks = 0
    by_session: dict[str, list[int]] = defaultdict(list)
    for turn in turns:
        by_session[turn.session_case_id].append(turn.turn_index)
        presentation = presentation_by_session.get(turn.session_case_id)
        if presentation is not None and any(
            business_id in turn.query_text
            for business_id in presentation.candidate_business_ids
        ):
            leaked_ids += 1
        if turn.family in {"relative_preference", "combined_update"} and re.search(
            r"\d+(?:\.\d+)?\s*(?:km|公里|美元|块|元|级)", turn.query_text, re.I
        ):
            numeric_leaks += 1
    if leaked_ids:
        issues.append(f"found {leaked_ids} raw business-ID leaks")
    if numeric_leaks:
        issues.append(f"found {numeric_leaks} relative numeric leaks")
    invalid_sequences = sum(
        sorted(indices) != list(range(2, max(indices) + 1))
        for indices in by_session.values()
    )
    if invalid_sequences:
        issues.append(f"found {invalid_sequences} non-contiguous sessions")
    split_counts = dict(sorted(Counter(item.split for item in turns).items()))
    language_counts = dict(sorted(Counter(item.language for item in turns).items()))
    family_counts = dict(sorted(Counter(item.family for item in turns).items()))
    expected_split, expected_language, expected_family = _expected_counts(config)
    if split_counts != expected_split:
        issues.append(f"split counts differ: {split_counts} != {expected_split}")
    if language_counts != expected_language:
        issues.append(f"language counts differ: {language_counts} != {expected_language}")
    if family_counts != expected_family:
        issues.append(f"family counts differ: {family_counts} != {expected_family}")
    if set(by_session) != set(sessions):
        issues.append("turn sessions and initial sessions do not align")
    return BenchmarkV2AuditReport(
        passed=not issues,
        session_count=len(sessions),
        turn_count=len(turns),
        split_counts=split_counts,
        language_counts=language_counts,
        family_counts=family_counts,
        exact_duplicate_count=exact_duplicates,
        cross_split_duplicate_count=cross_split,
        near_duplicate_count=near_duplicates,
        leaked_business_id_count=leaked_ids,
        relative_numeric_leak_count=numeric_leaks,
        invalid_session_sequence_count=invalid_sequences,
        issues=issues,
    )


def _expected_counts(config: SessionMemoryBenchmarkV2Config):
    split: Counter[str] = Counter()
    language: Counter[str] = Counter()
    family: Counter[str] = Counter()
    for split_name, split_plan in config.split_plans.items():
        for language_name, language_plan in split_plan.languages.items():
            counts = Counter(language_plan.recommendation_families) + Counter(language_plan.support_families)
            expanded = {
                key: language_plan.recommendation_families.get(key, 0)
                + language_plan.support_families.get(key, 0)
                for key in set(language_plan.recommendation_families) | set(language_plan.support_families)
            }
            count = sum(expanded.values())
            split[split_name] += count
            language[language_name] += count
            family.update(expanded)
    return dict(sorted(split.items())), dict(sorted(language.items())), dict(sorted(family.items()))


def _near_duplicate_count(texts: Sequence[str], threshold: float) -> int:
    shingles = [_trigrams(item) for item in texts]
    count = 0
    for left in range(len(shingles)):
        for right in range(left + 1, len(shingles)):
            union = shingles[left] | shingles[right]
            score = len(shingles[left] & shingles[right]) / len(union) if union else 1.0
            if score >= threshold:
                count += 1
    return count


def _trigrams(value: str) -> set[str]:
    compact = re.sub(r"\s+", "", value.casefold())
    return {compact[index : index + 3] for index in range(max(1, len(compact) - 2))}


def _normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def _write_models(path: Path, values: Sequence[StrictModel], key: str) -> None:
    ordered = sorted(values, key=lambda item: str(getattr(item, key)))
    _write_text(path, "".join(item.model_dump_json() + "\n" for item in ordered))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(content, encoding="utf-8", newline="\n")
    partial.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

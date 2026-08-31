"""通过Claude Code调用GLM，为候选评论生成可续跑的教师标签。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from dotenv import load_dotenv

from yelp_agent.recommendation_v2.rag_benchmark.claude_worker import (
    ClaudeCodeWorker,
    ClaudeStructuredOutputError,
)
from yelp_agent.recommendation_v2.rag_benchmark.schema import ClaudeWorkerTrace
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS, AspectField

from .contracts import LabeledTeacherSample, TeacherCandidate, TeacherLabel


class TeacherWorker(Protocol):
    """真实Claude Code和测试替身共同遵守的最小调用接口。"""

    def generate(
        self,
        prompt: str,
        output_model: type[TeacherLabel],
        *,
        system_prompt: str | None = None,
    ) -> tuple[TeacherLabel, ClaudeWorkerTrace, dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class TeacherRunConfig:
    """一轮教师标注的固定运行参数。"""

    dataset_root: Path
    template_path: Path
    run_id: str = "glm53flash_v1"
    model: str = "glm-5.3-flash"
    claude_command: tuple[str, ...] = ("claude",)
    legacy_cli: bool = False
    max_workers: int = 4
    max_attempts: int = 3
    request_timeout_seconds: int = 120
    limit_per_aspect: int | None = None

    def __post_init__(self) -> None:
        if not self.run_id or any(char in self.run_id for char in "\\/:*?\"<>|"):
            raise ValueError("run_id must be a nonempty Windows-safe name")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.request_timeout_seconds < 1:
            raise ValueError("request_timeout_seconds must be positive")
        if self.limit_per_aspect is not None and self.limit_per_aspect < 1:
            raise ValueError("limit_per_aspect must be positive")


@dataclass(frozen=True, slots=True)
class _AttemptEvent:
    attempt: int
    status: str
    wrapper: dict[str, object] | None
    trace: ClaudeWorkerTrace | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class _CandidateResult:
    candidate: TeacherCandidate
    label: TeacherLabel | None
    input_hash: str
    events: tuple[_AttemptEvent, ...]


def run_teacher_labeling(
    config: TeacherRunConfig,
    *,
    worker: TeacherWorker | None = None,
) -> dict[str, object]:
    """读取候选、调用教师、保存原始结果和正式标签并返回统计。"""

    load_dotenv()
    dataset_root = config.dataset_root.resolve()
    template_path = config.template_path.resolve()
    template = _read_json(template_path)
    system_prompt = _required_text(template, "system_prompt")
    all_candidates = _load_all_candidates(dataset_root)
    selected = _select_candidates(all_candidates, config.limit_per_aspect)

    run_root = dataset_root / "teacher_runs" / config.run_id
    raw_root = run_root / "raw_responses"
    labeled_root = dataset_root / "labeled"
    raw_root.mkdir(parents=True, exist_ok=True)
    labeled_root.mkdir(parents=True, exist_ok=True)
    _ensure_run_manifest(
        config=config,
        run_root=run_root,
        template_path=template_path,
        system_prompt=system_prompt,
        dataset_candidate_count=len(all_candidates),
    )

    completed_ids = _load_completed_ids(labeled_root)
    pending = [item for item in selected if item.sample_id not in completed_ids]
    active_worker = worker or cast(
        TeacherWorker,
        ClaudeCodeWorker(
            model=config.model,
            timeout_seconds=config.request_timeout_seconds,
            executable=config.claude_command,
            legacy_cli=config.legacy_cli,
        ),
    )

    started = perf_counter()
    invocation_success = 0
    invocation_rejected = 0
    if pending:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures: dict[Future[_CandidateResult], TeacherCandidate] = {
                executor.submit(
                    _label_candidate,
                    candidate,
                    worker=active_worker,
                    system_prompt=system_prompt,
                    max_attempts=config.max_attempts,
                ): candidate
                for candidate in pending
            }
            for future in as_completed(futures):
                result = future.result()
                _save_attempts(raw_root, result)
                if result.label is None:
                    invocation_rejected += 1
                    _append_jsonl(
                        run_root / "rejected.jsonl",
                        _rejected_record(result),
                    )
                else:
                    invocation_success += 1
                    completed_ids.add(result.candidate.sample_id)
                    _append_jsonl(
                        labeled_root / f"{result.candidate.model_input.aspect_id}.jsonl",
                        LabeledTeacherSample(
                            sample_id=result.candidate.sample_id,
                            model_input=result.candidate.model_input,
                            model_output=result.label,
                        ).model_dump(mode="json"),
                    )
                _write_progress(
                    run_root=run_root,
                    dataset_total=len(all_candidates),
                    selected_total=len(selected),
                    completed_ids=completed_ids,
                    invocation_rejected=invocation_rejected,
                )

    summary = _build_summary(
        config=config,
        run_root=run_root,
        dataset_total=len(all_candidates),
        selected_total=len(selected),
        completed_ids=completed_ids,
        pending_at_start=len(pending),
        invocation_success=invocation_success,
        invocation_rejected=invocation_rejected,
        wall_latency_ms=(perf_counter() - started) * 1000,
    )
    _write_json(run_root / "summary.json", summary)
    return summary


def _load_all_candidates(dataset_root: Path) -> list[TeacherCandidate]:
    # 校验逻辑会比较解析后的绝对路径；这里先统一根目录，确保调用方传入
    # 相对路径或绝对路径时得到完全一致的结果。
    dataset_root = dataset_root.resolve()
    manifest = _read_json(dataset_root / "candidate_manifest.json")
    aspect_entries = manifest.get("aspects")
    if not isinstance(aspect_entries, dict):
        raise ValueError("candidate manifest has no aspects object")

    candidates: list[TeacherCandidate] = []
    seen_ids: set[str] = set()
    for aspect in ASPECT_FIELDS:
        entry = aspect_entries.get(aspect)
        if not isinstance(entry, dict):
            raise ValueError(f"candidate manifest is missing {aspect}")
        relative_output = entry.get("output")
        if not isinstance(relative_output, str):
            raise ValueError(f"candidate manifest output is invalid for {aspect}")
        output_path = (dataset_root / relative_output).resolve()
        if not output_path.is_relative_to(dataset_root):
            raise ValueError("candidate paths must stay inside the dataset root")
        if output_path != (dataset_root / "candidates" / f"{aspect}.jsonl").resolve():
            raise ValueError(f"candidate manifest path does not match {aspect}")
        aspect_candidates = _read_candidate_jsonl(output_path)
        expected_count = int(entry.get("candidate_count") or 0)
        if len(aspect_candidates) != expected_count:
            raise ValueError(
                f"candidate count mismatch for {aspect}: "
                f"manifest={expected_count}, file={len(aspect_candidates)}"
            )
        for candidate in aspect_candidates:
            if candidate.model_input.aspect_id != aspect:
                raise ValueError(f"candidate aspect mismatch in {output_path}")
            if candidate.sample_id in seen_ids:
                raise ValueError(f"duplicate candidate sample_id: {candidate.sample_id}")
            seen_ids.add(candidate.sample_id)
        candidates.extend(aspect_candidates)
    return candidates


def _read_candidate_jsonl(path: Path) -> list[TeacherCandidate]:
    values: list[TeacherCandidate] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(TeacherCandidate.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid candidate at {path}:{line_number}: {exc}") from exc
    return values


def _select_candidates(
    candidates: Sequence[TeacherCandidate],
    limit_per_aspect: int | None,
) -> list[TeacherCandidate]:
    if limit_per_aspect is None:
        return list(candidates)
    selected: list[TeacherCandidate] = []
    for aspect in ASPECT_FIELDS:
        group = [item for item in candidates if item.model_input.aspect_id == aspect]
        if limit_per_aspect >= len(group):
            selected.extend(group)
            continue
        if limit_per_aspect == 1:
            selected.append(group[len(group) // 2])
            continue
        indexes = {
            round(index * (len(group) - 1) / (limit_per_aspect - 1))
            for index in range(limit_per_aspect)
        }
        selected.extend(group[index] for index in sorted(indexes))
    return selected


def _label_candidate(
    candidate: TeacherCandidate,
    *,
    worker: TeacherWorker,
    system_prompt: str,
    max_attempts: int,
) -> _CandidateResult:
    input_json = json.dumps(
        candidate.model_input.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    input_hash = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
    events: list[_AttemptEvent] = []
    for attempt in range(1, max_attempts + 1):
        try:
            label, trace, wrapper = worker.generate(
                input_json,
                TeacherLabel,
                system_prompt=system_prompt,
            )
            events.append(
                _AttemptEvent(
                    attempt=attempt,
                    status="success",
                    wrapper=wrapper,
                    trace=trace,
                    error_type=None,
                    error_message=None,
                )
            )
            return _CandidateResult(
                candidate=candidate,
                label=label,
                input_hash=input_hash,
                events=tuple(events),
            )
        except ClaudeStructuredOutputError as exc:
            events.append(
                _AttemptEvent(
                    attempt=attempt,
                    status="invalid_output",
                    wrapper=exc.wrapper,
                    trace=exc.trace,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1000],
                )
            )
        except Exception as exc:
            events.append(
                _AttemptEvent(
                    attempt=attempt,
                    status="provider_failure",
                    wrapper=None,
                    trace=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1000],
                )
            )
    return _CandidateResult(
        candidate=candidate,
        label=None,
        input_hash=input_hash,
        events=tuple(events),
    )


def _save_attempts(raw_root: Path, result: _CandidateResult) -> None:
    aspect = result.candidate.model_input.aspect_id
    for event in result.events:
        _append_jsonl(
            raw_root / f"{aspect}.jsonl",
            {
                "sample_id": result.candidate.sample_id,
                "aspect_id": aspect,
                "attempt": event.attempt,
                "status": event.status,
                "input_sha256": result.input_hash,
                "trace": event.trace.model_dump(mode="json") if event.trace else None,
                "raw_wrapper": event.wrapper,
                "error_type": event.error_type,
                "error_message": event.error_message,
                "recorded_at": _now_iso(),
            },
        )


def _rejected_record(result: _CandidateResult) -> dict[str, object]:
    last = result.events[-1]
    return {
        "sample_id": result.candidate.sample_id,
        "aspect_id": result.candidate.model_input.aspect_id,
        "attempt_count": len(result.events),
        "last_status": last.status,
        "error_type": last.error_type,
        "error_message": last.error_message,
        "recorded_at": _now_iso(),
    }


def _ensure_run_manifest(
    *,
    config: TeacherRunConfig,
    run_root: Path,
    template_path: Path,
    system_prompt: str,
    dataset_candidate_count: int,
) -> None:
    path = run_root / "run_manifest.json"
    immutable = {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "model": config.model,
        "dataset_candidate_count": dataset_candidate_count,
        "template_sha256": _sha256_file(template_path),
        "candidate_manifest_sha256": _sha256_file(
            config.dataset_root.resolve() / "candidate_manifest.json"
        ),
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "system_prompt": system_prompt,
        "max_attempts": config.max_attempts,
    }
    if path.exists():
        saved = _read_json(path)
        for key, value in immutable.items():
            if saved.get(key) != value:
                raise ValueError(f"run manifest mismatch for {key}")
        return
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(path, {**immutable, "created_at": _now_iso()})


def _load_completed_ids(labeled_root: Path) -> set[str]:
    completed: set[str] = set()
    for path in sorted(labeled_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            sample_id = value.get("sample_id")
            if not isinstance(sample_id, str):
                raise ValueError(f"labeled row has no sample_id: {path}")
            if sample_id in completed:
                raise ValueError(f"duplicate labeled sample_id: {sample_id}")
            completed.add(sample_id)
    return completed


def _write_progress(
    *,
    run_root: Path,
    dataset_total: int,
    selected_total: int,
    completed_ids: set[str],
    invocation_rejected: int,
) -> None:
    _write_json(
        run_root / "progress.json",
        {
            "dataset_total": dataset_total,
            "selected_total": selected_total,
            "success_total": len(completed_ids),
            "rejected_this_invocation": invocation_rejected,
            "remaining_in_selected": max(0, selected_total - len(completed_ids)),
            "updated_at": _now_iso(),
        },
    )


def _build_summary(
    *,
    config: TeacherRunConfig,
    run_root: Path,
    dataset_total: int,
    selected_total: int,
    completed_ids: set[str],
    pending_at_start: int,
    invocation_success: int,
    invocation_rejected: int,
    wall_latency_ms: float,
) -> dict[str, object]:
    events = list(_read_raw_events(run_root / "raw_responses"))
    traces = [
        event["trace"]
        for event in events
        if isinstance(event.get("trace"), dict)
    ]
    attempt_status_counts = {
        status: sum(event.get("status") == status for event in events)
        for status in ("success", "invalid_output", "provider_failure")
    }
    return {
        "run_id": config.run_id,
        "model": config.model,
        "dataset_total": dataset_total,
        "selected_total": selected_total,
        "completed_total": len(completed_ids),
        "pending_at_invocation_start": pending_at_start,
        "success_this_invocation": invocation_success,
        "rejected_this_invocation": invocation_rejected,
        "attempt_count_total": len(events),
        "attempt_status_counts": attempt_status_counts,
        "input_tokens_total": sum(int(item.get("input_tokens") or 0) for item in traces),
        "output_tokens_total": sum(int(item.get("output_tokens") or 0) for item in traces),
        "thinking_tokens_total": sum(int(item.get("thinking_tokens") or 0) for item in traces),
        "reported_cost_usd_total": sum(
            float(item.get("reported_cost_usd") or 0.0) for item in traces
        ),
        "model_duration_ms_total": sum(int(item.get("duration_ms") or 0) for item in traces),
        "wall_latency_ms_this_invocation": wall_latency_ms,
        "max_workers": config.max_workers,
        "max_attempts": config.max_attempts,
        "request_timeout_seconds": config.request_timeout_seconds,
        "updated_at": _now_iso(),
    }


def _read_raw_events(raw_root: Path) -> Iterable[dict[str, Any]]:
    if not raw_root.exists():
        return
    for path in sorted(raw_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _required_text(value: dict[str, Any], key: str) -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"missing nonempty {key}")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过Claude Code运行GLM教师标注")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--template-path", type=Path, required=True)
    parser.add_argument("--run-id", default="glm53flash_v1")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=int, default=120)
    parser.add_argument("--limit-per-aspect", type=int)
    parser.add_argument("--claude-executable", default="claude")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_teacher_labeling(
        TeacherRunConfig(
            dataset_root=args.dataset_root,
            template_path=args.template_path,
            run_id=args.run_id,
            model=args.model,
            claude_command=(args.claude_executable,),
            max_workers=args.max_workers,
            max_attempts=args.max_attempts,
            request_timeout_seconds=args.request_timeout_seconds,
            limit_per_aspect=args.limit_per_aspect,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

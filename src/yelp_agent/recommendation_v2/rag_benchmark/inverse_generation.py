"""从真实评论反向生成可回放的评论召回评测集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import duckdb

from yelp_agent.recommendation_v2.review_features.definitions import (
    aspect_meaning,
    preference_semantic_anchors,
)
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS, AspectField

from .claude_worker import (
    ClaudeCodeWorker,
    ClaudeStructuredOutputError,
    load_prompt,
)
from .inverse_schema import (
    InverseBenchmarkCase,
    InverseGeneratedDraft,
    InverseGenerationCandidate,
    InverseGenerationProposal,
)
from .schema import ClaudeWorkerTrace


PROMPT_VERSION = "inverse-generation-v2"
_MODULE_ROOT = Path(__file__).resolve().parent
_FIXED_PROMPT = _MODULE_ROOT / "prompts/inverse_fixed_aspect_v1.txt"
_LONG_TAIL_PROMPT = _MODULE_ROOT / "prompts/inverse_long_tail_v1.txt"

# 第一版固定使用常见需求方向；数据中仍保存方向，后续可以增加反向需求。
_TARGET_DIRECTIONS: dict[AspectField, Literal["higher", "lower"]] = {
    **{aspect: "higher" for aspect in ASPECT_FIELDS},
    "crowded": "lower",
    "queue_time": "lower",
}


@dataclass(frozen=True, slots=True)
class InverseGenerationConfig:
    """首批逆向评测集生成需要的真实数据和批量限制。"""

    source_project_root: Path
    output_root: Path
    model: str = "glm-5.3-flash[1m]"
    claude_model_alias: str = "haiku"
    claude_command: tuple[str, ...] = ("claude",)
    claude_legacy_cli: bool = False
    fixed_candidates_per_star_bucket: int = 18
    fixed_positive_count: int = 2
    fixed_negative_count: int = 2
    long_tail_case_count: int = 20
    long_tail_batch_count: int = 2
    long_tail_candidates_per_batch: int = 36
    max_attempts: int = 2
    concurrency: int = 3
    only_batch_ids: tuple[str, ...] = ()

    @property
    def feature_root(self) -> Path:
        return (
            self.source_project_root
            / "src/yelp_agent/recommendation_v2/data/review_features/v1"
        )

    @property
    def segment_path(self) -> Path:
        return (
            self.source_project_root
            / "src/yelp_agent/recommendation_v2/data/review_evidence/v1/index/review_segments.parquet"
        )

    @property
    def business_facts_path(self) -> Path:
        return (
            self.source_project_root
            / "src/yelp_agent/recommendation_v2/data/business_facts/v1/business_facts.parquet"
        )


@dataclass(frozen=True, slots=True)
class _SeedRecord:
    review_id: str
    business_id: str
    review_text: str
    review_text_sha256: str

    def public_candidate(self) -> InverseGenerationCandidate:
        return InverseGenerationCandidate(
            review_id=self.review_id,
            business_id=self.business_id,
            review_text=self.review_text,
        )


@dataclass(frozen=True, slots=True)
class _GenerationBatch:
    batch_id: str
    dataset_kind: Literal["fixed_aspect", "long_tail"]
    candidates: tuple[_SeedRecord, ...]
    expected_count: int
    expected_positive: int | None
    expected_negative: int | None
    prompt: str
    aspect: AspectField | None = None
    target_direction: Literal["higher", "lower"] | None = None


@dataclass(frozen=True, slots=True)
class _BatchOutcome:
    batch_id: str
    cases: tuple[InverseBenchmarkCase, ...]
    traces: tuple[ClaudeWorkerTrace, ...]
    rejected_reasons: tuple[str, ...]
    status: Literal["success", "failed"]


def run_inverse_benchmark_generation(
    config: InverseGenerationConfig,
    *,
    worker: ClaudeCodeWorker | None = None,
) -> dict[str, object]:
    """一个入口完成真实采样、GLM生成、硬校验、续跑和统计。"""

    _validate_config(config)
    # Claude Code只接收自身别名；用户设置再把haiku映射到实际GLM模型。
    worker = worker or ClaudeCodeWorker(
        model=config.claude_model_alias,
        executable=config.claude_command,
        legacy_cli=config.claude_legacy_cli,
    )
    started = perf_counter()
    for folder in ("inputs", "raw_outputs", "accepted", "rejected", "drafts"):
        (config.output_root / folder).mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(database=":memory:")
    try:
        _register_sources(connection, config)
        batches = _build_batches(connection, config)
        if config.only_batch_ids:
            expected_ids = {item.batch_id for item in batches}
            unknown = set(config.only_batch_ids) - expected_ids
            if unknown:
                raise ValueError(f"unknown inverse generation batches: {sorted(unknown)}")
            batches = [
                item for item in batches if item.batch_id in config.only_batch_ids
            ]
        _write_batch_inputs(config.output_root, batches)
        outcomes = _run_batches(connection, config, worker, batches)
    finally:
        connection.close()

    cases = [case for outcome in outcomes for case in outcome.cases]
    cases = _deduplicate_cases(cases)
    draft_path = config.output_root / "drafts/inverse_cases.jsonl"
    draft_path.write_text(
        "".join(
            json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    traces = [trace for outcome in outcomes for trace in outcome.traces]
    fixed_counts = Counter(
        case.aspect for case in cases if case.dataset_kind == "fixed_aspect"
    )
    direction_counts = Counter(case.expected_direction for case in cases)
    target_case_count = sum(batch.expected_count for batch in batches)
    report: dict[str, object] = {
        "status": "success" if len(cases) == target_case_count else "partial",
        "prompt_version": PROMPT_VERSION,
        "teacher_model": config.model,
        "claude_model_alias": config.claude_model_alias,
        "target_case_count": target_case_count,
        "generated_case_count": len(cases),
        "fixed_case_count": sum(fixed_counts.values()),
        "long_tail_case_count": sum(
            case.dataset_kind == "long_tail" for case in cases
        ),
        "fixed_counts": dict(sorted(fixed_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
        "batch_statuses": {
            outcome.batch_id: outcome.status for outcome in outcomes
        },
        "rejected_reasons": [
            {"batch_id": outcome.batch_id, "reason": reason}
            for outcome in outcomes
            for reason in outcome.rejected_reasons
        ],
        "model_call_count": len(traces),
        "input_tokens": sum(trace.input_tokens or 0 for trace in traces),
        "output_tokens": sum(trace.output_tokens or 0 for trace in traces),
        "thinking_tokens": sum(trace.thinking_tokens or 0 for trace in traces),
        "reported_cost_usd": sum(trace.reported_cost_usd or 0 for trace in traces),
        "model_api_time_ms_sum": sum(trace.duration_ms for trace in traces),
        "wall_latency_ms": (perf_counter() - started) * 1000,
        "draft_path": str(draft_path.resolve()),
    }
    (config.output_root / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _register_sources(
    connection: duckdb.DuckDBPyConnection,
    config: InverseGenerationConfig,
) -> None:
    paths = {
        "candidate_reviews": config.feature_root / "candidate_reviews.parquet",
        "candidate_aspects": config.feature_root / "candidate_aspects.parquet",
        "review_segments": config.segment_path,
        "business_facts": config.business_facts_path,
    }
    for name, path in paths.items():
        escaped = str(path.resolve()).replace("'", "''")
        connection.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')"
        )


def _build_batches(
    connection: duckdb.DuckDBPyConnection,
    config: InverseGenerationConfig,
) -> list[_GenerationBatch]:
    batches: list[_GenerationBatch] = []
    fixed_review_ids: set[str] = set()
    connection.execute("CREATE TEMP TABLE fixed_candidate_exclusions(review_id VARCHAR)")
    fixed_template = load_prompt(_FIXED_PROMPT)
    for aspect in ASPECT_FIELDS:
        direction = _TARGET_DIRECTIONS[aspect]
        anchors = preference_semantic_anchors(aspect, direction)
        candidates = _sample_fixed_candidates(connection, config, aspect)
        fixed_review_ids.update(item.review_id for item in candidates)
        connection.executemany(
            "INSERT INTO fixed_candidate_exclusions VALUES (?)",
            [(item.review_id,) for item in candidates],
        )
        payload = [item.public_candidate().model_dump(mode="json") for item in candidates]
        prompt = (
            fixed_template.replace("{{ASPECT}}", aspect)
            .replace("{{MEANING_ZH}}", aspect_meaning(aspect))
            .replace("{{TARGET_DIRECTION}}", direction)
            .replace(
                "{{SATISFYING_JSON}}",
                json.dumps(anchors.satisfying, ensure_ascii=False),
            )
            .replace(
                "{{CONTRADICTING_JSON}}",
                json.dumps(anchors.contradicting, ensure_ascii=False),
            )
            .replace("{{CANDIDATES_JSON}}", json.dumps(payload, ensure_ascii=False))
        )
        batches.append(
            _GenerationBatch(
                batch_id=f"fixed_{aspect}",
                dataset_kind="fixed_aspect",
                aspect=aspect,
                target_direction=direction,
                candidates=tuple(candidates),
                expected_count=config.fixed_positive_count
                + config.fixed_negative_count,
                expected_positive=config.fixed_positive_count,
                expected_negative=config.fixed_negative_count,
                prompt=prompt,
            )
        )

    long_tail_candidates = _sample_long_tail_candidates(
        connection,
        config,
        excluded_review_ids=fixed_review_ids,
    )
    long_tail_template = load_prompt(_LONG_TAIL_PROMPT)
    each_batch = config.long_tail_case_count // config.long_tail_batch_count
    remainder = config.long_tail_case_count % config.long_tail_batch_count
    for index in range(config.long_tail_batch_count):
        start = index * config.long_tail_candidates_per_batch
        end = start + config.long_tail_candidates_per_batch
        candidates = long_tail_candidates[start:end]
        expected = each_batch + (1 if index < remainder else 0)
        payload = [item.public_candidate().model_dump(mode="json") for item in candidates]
        prompt = (
            long_tail_template.replace("{{CASE_COUNT}}", str(expected)).replace(
                "{{CANDIDATES_JSON}}", json.dumps(payload, ensure_ascii=False)
            )
        )
        batches.append(
            _GenerationBatch(
                batch_id=f"long_tail_{index + 1:02d}",
                dataset_kind="long_tail",
                candidates=tuple(candidates),
                expected_count=expected,
                expected_positive=None,
                expected_negative=None,
                prompt=prompt,
            )
        )
    return batches


def _sample_fixed_candidates(
    connection: duckdb.DuckDBPyConnection,
    config: InverseGenerationConfig,
    aspect: AspectField,
) -> list[_SeedRecord]:
    rows = connection.execute(
        """
        WITH eligible AS (
            SELECT
                r.review_id,
                r.business_id,
                r.review_text,
                r.review_text_sha256,
                CASE WHEN r.stars >= 4 THEN 'high' ELSE 'low' END AS star_bucket,
                ROW_NUMBER() OVER (
                    PARTITION BY CASE WHEN r.stars >= 4 THEN 'high' ELSE 'low' END
                    ORDER BY
                        (a.keyword_hit AND a.semantic_hit) DESC,
                        a.keyword_hit DESC,
                        a.semantic_score DESC NULLS LAST,
                        r.useful DESC,
                        HASH(r.review_id || a.aspect)
                ) AS bucket_rank
            FROM candidate_aspects a
            JOIN candidate_reviews r USING (review_id, business_id)
            LEFT JOIN fixed_candidate_exclusions e USING (review_id)
            WHERE a.aspect = ?
              AND e.review_id IS NULL
              AND (r.stars <= 2 OR r.stars >= 4)
              AND LENGTH(r.review_text) BETWEEN 120 AND 1800
        )
        SELECT review_id, business_id, review_text, review_text_sha256
        FROM eligible
        WHERE bucket_rank <= ?
        ORDER BY star_bucket, bucket_rank
        """,
        [aspect, config.fixed_candidates_per_star_bucket],
    ).fetchall()
    return [_SeedRecord(*map(str, row)) for row in rows]


def _sample_long_tail_candidates(
    connection: duckdb.DuckDBPyConnection,
    config: InverseGenerationConfig,
    *,
    excluded_review_ids: set[str],
) -> list[_SeedRecord]:
    required = config.long_tail_batch_count * config.long_tail_candidates_per_batch
    # 用临时表传排除编号，不把几百个编号拼进SQL字符串。
    connection.execute("CREATE TEMP TABLE excluded_reviews(review_id VARCHAR)")
    if excluded_review_ids:
        connection.executemany(
            "INSERT INTO excluded_reviews VALUES (?)",
            [(value,) for value in sorted(excluded_review_ids)],
        )
    rows = connection.execute(
        """
        SELECT r.review_id, r.business_id, r.review_text, r.review_text_sha256
        FROM candidate_reviews r
        LEFT JOIN excluded_reviews e USING (review_id)
        WHERE e.review_id IS NULL
          AND (r.stars <= 2 OR r.stars >= 4)
          AND LENGTH(r.review_text) BETWEEN 180 AND 1800
        ORDER BY r.useful DESC, HASH(r.review_id || 'inverse-long-tail-v1')
        LIMIT ?
        """,
        [required],
    ).fetchall()
    if len(rows) != required:
        raise ValueError("not enough long-tail seed candidates")
    return [_SeedRecord(*map(str, row)) for row in rows]


def _write_batch_inputs(output_root: Path, batches: list[_GenerationBatch]) -> None:
    for batch in batches:
        payload = {
            "batch_id": batch.batch_id,
            "dataset_kind": batch.dataset_kind,
            "aspect": batch.aspect,
            "target_direction": batch.target_direction,
            "expected_count": batch.expected_count,
            "candidates": [
                item.public_candidate().model_dump(mode="json")
                for item in batch.candidates
            ],
            "prompt": batch.prompt,
        }
        (output_root / "inputs" / f"{batch.batch_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _run_batches(
    connection: duckdb.DuckDBPyConnection,
    config: InverseGenerationConfig,
    worker: ClaudeCodeWorker,
    batches: list[_GenerationBatch],
) -> list[_BatchOutcome]:
    # DuckDB连接不能跨线程共享；先把片段和商家名称读成内存映射。
    segments_by_review = _load_segment_map(connection, batches)
    business_names = _load_business_names(connection, batches)
    outcomes: list[_BatchOutcome] = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = {
            executor.submit(
                _run_one_batch,
                config,
                worker,
                batch,
                segments_by_review,
                business_names,
            ): batch.batch_id
            for batch in batches
        }
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print(
                f"{outcome.batch_id}: {outcome.status}, "
                f"cases={len(outcome.cases)}, calls={len(outcome.traces)}",
                flush=True,
            )
    return sorted(outcomes, key=lambda item: item.batch_id)


def _run_one_batch(
    config: InverseGenerationConfig,
    worker: ClaudeCodeWorker,
    batch: _GenerationBatch,
    segments_by_review: dict[str, list[tuple[str, str]]],
    business_names: dict[str, str],
) -> _BatchOutcome:
    accepted_path = config.output_root / "accepted" / f"{batch.batch_id}.json"
    if accepted_path.is_file():
        saved = json.loads(accepted_path.read_text(encoding="utf-8"))
        return _BatchOutcome(
            batch_id=batch.batch_id,
            cases=tuple(InverseBenchmarkCase.model_validate(item) for item in saved),
            traces=(),
            rejected_reasons=(),
            status="success",
        )

    traces: list[ClaudeWorkerTrace] = []
    rejected: list[str] = []
    correction = ""
    for attempt in range(1, config.max_attempts + 1):
        prompt = batch.prompt + correction
        try:
            proposal, trace, wrapper = worker.generate(
                prompt, InverseGenerationProposal
            )
            traces.append(trace)
            _write_raw(config.output_root, batch.batch_id, attempt, wrapper)
            cases = _validate_and_materialize(
                batch,
                proposal,
                segments_by_review,
                business_names,
                teacher_model=config.model,
            )
        except ClaudeStructuredOutputError as exc:
            traces.append(exc.trace)
            _write_raw(config.output_root, batch.batch_id, attempt, exc.wrapper)
            reason = f"structured output invalid: {exc}"
        except (ValueError, KeyError) as exc:
            reason = str(exc)
        except RuntimeError as exc:
            # 外部命令失败也必须留在批次报告中，不能让其余批次全部中断。
            reason = f"model command failed: {exc}"
        else:
            accepted_path.write_text(
                json.dumps(
                    [case.model_dump(mode="json") for case in cases],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return _BatchOutcome(
                batch_id=batch.batch_id,
                cases=tuple(cases),
                traces=tuple(traces),
                rejected_reasons=tuple(rejected),
                status="success",
            )
        rejected.append(f"attempt_{attempt}: {reason}"[:1000])
        correction = (
            "\n\n上一次输出没有通过程序检查。必须重新阅读原评论并完整重做。"
            f"失败原因：{reason[:700]}"
        )

    (config.output_root / "rejected" / f"{batch.batch_id}.json").write_text(
        json.dumps({"batch_id": batch.batch_id, "reasons": rejected}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return _BatchOutcome(
        batch_id=batch.batch_id,
        cases=(),
        traces=tuple(traces),
        rejected_reasons=tuple(rejected),
        status="failed",
    )


def _validate_and_materialize(
    batch: _GenerationBatch,
    proposal: InverseGenerationProposal,
    segments_by_review: dict[str, list[tuple[str, str]]],
    business_names: dict[str, str],
    *,
    teacher_model: str,
) -> list[InverseBenchmarkCase]:
    drafts = proposal.drafts
    if len(drafts) != batch.expected_count:
        raise ValueError(
            f"expected {batch.expected_count} drafts but received {len(drafts)}"
        )
    review_ids = [item.seed_review_id for item in drafts]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("one generation batch reused the same review")
    candidates = {item.review_id: item for item in batch.candidates}
    unknown = set(review_ids) - set(candidates)
    if unknown:
        raise ValueError(f"model returned unknown review ids: {sorted(unknown)}")
    directions = Counter(item.expected_direction for item in drafts)
    if batch.expected_positive is not None and (
        directions["positive"] != batch.expected_positive
        or directions["negative"] != batch.expected_negative
    ):
        raise ValueError(f"direction counts do not match: {dict(directions)}")

    result: list[InverseBenchmarkCase] = []
    per_direction_index: Counter[str] = Counter()
    for draft in drafts:
        seed = candidates[draft.seed_review_id]
        if draft.evidence_span not in seed.review_text:
            raise ValueError(
                f"evidence is not an exact substring: {draft.seed_review_id}"
            )
        business_name = business_names.get(seed.business_id, "").strip()
        if business_name and business_name.casefold() in draft.query_text.casefold():
            raise ValueError(f"query leaked business name: {draft.seed_review_id}")
        segment = _find_evidence_segment(
            segments_by_review.get(seed.review_id, []), draft.evidence_span
        )
        if segment is None:
            raise ValueError(
                f"exact evidence does not fit in one indexed segment: {seed.review_id}"
            )
        segment_id, context = segment
        per_direction_index[draft.expected_direction] += 1
        suffix = per_direction_index[draft.expected_direction]
        if batch.dataset_kind == "fixed_aspect":
            case_id = (
                f"inverse_fixed_{batch.aspect}_{draft.expected_direction}_{suffix:02d}"
            )
        else:
            case_id = f"inverse_{batch.batch_id}_{draft.expected_direction}_{suffix:02d}"
        result.append(
            InverseBenchmarkCase(
                case_id=case_id,
                dataset_kind=batch.dataset_kind,
                aspect=batch.aspect,
                target_direction=batch.target_direction,
                query_text=draft.query_text,
                requirement_text=draft.requirement_text,
                expected_direction=draft.expected_direction,
                seed_review_id=seed.review_id,
                seed_business_id=seed.business_id,
                seed_segment_id=segment_id,
                evidence_span=draft.evidence_span,
                evidence_context=context,
                review_text_sha256=seed.review_text_sha256,
                teacher_model=teacher_model,
                prompt_version=PROMPT_VERSION,
            )
        )
    return result


def _load_segment_map(
    connection: duckdb.DuckDBPyConnection,
    batches: list[_GenerationBatch],
) -> dict[str, list[tuple[str, str]]]:
    review_ids = sorted(
        {item.review_id for batch in batches for item in batch.candidates}
    )
    connection.execute("CREATE TEMP TABLE requested_reviews(review_id VARCHAR)")
    connection.executemany(
        "INSERT INTO requested_reviews VALUES (?)", [(value,) for value in review_ids]
    )
    rows = connection.execute(
        """
        SELECT s.review_id, s.segment_id, s.text
        FROM review_segments s
        JOIN requested_reviews r USING (review_id)
        ORDER BY s.review_id, s.segment_index
        """
    ).fetchall()
    result: dict[str, list[tuple[str, str]]] = {}
    for review_id, segment_id, text in rows:
        result.setdefault(str(review_id), []).append((str(segment_id), str(text)))
    return result


def _load_business_names(
    connection: duckdb.DuckDBPyConnection,
    batches: list[_GenerationBatch],
) -> dict[str, str]:
    business_ids = sorted(
        {item.business_id for batch in batches for item in batch.candidates}
    )
    connection.execute("CREATE TEMP TABLE requested_businesses(business_id VARCHAR)")
    connection.executemany(
        "INSERT INTO requested_businesses VALUES (?)",
        [(value,) for value in business_ids],
    )
    rows = connection.execute(
        """
        SELECT b.business_id, b.name
        FROM business_facts b
        JOIN requested_businesses r USING (business_id)
        """
    ).fetchall()
    return {str(business_id): str(name) for business_id, name in rows}


def _find_evidence_segment(
    segments: list[tuple[str, str]], evidence_span: str
) -> tuple[str, str] | None:
    matches = [item for item in segments if evidence_span in item[1]]
    return min(matches, key=lambda item: (len(item[1]), item[0])) if matches else None


def _deduplicate_cases(
    cases: list[InverseBenchmarkCase],
) -> list[InverseBenchmarkCase]:
    result: list[InverseBenchmarkCase] = []
    seen_reviews: set[str] = set()
    for case in sorted(cases, key=lambda item: item.case_id):
        if case.seed_review_id in seen_reviews:
            continue
        seen_reviews.add(case.seed_review_id)
        result.append(case)
    return result


def _write_raw(
    output_root: Path,
    batch_id: str,
    attempt: int,
    wrapper: dict[str, object],
) -> None:
    (output_root / "raw_outputs" / f"{batch_id}.attempt_{attempt}.json").write_text(
        json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_config(config: InverseGenerationConfig) -> None:
    for path in (
        config.feature_root / "candidate_reviews.parquet",
        config.feature_root / "candidate_aspects.parquet",
        config.segment_path,
        config.business_facts_path,
        _FIXED_PROMPT,
        _LONG_TAIL_PROMPT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        config.fixed_candidates_per_star_bucket < 4
        or config.long_tail_batch_count < 1
        or config.long_tail_candidates_per_batch < 10
        or config.max_attempts < 1
        or config.concurrency < 1
    ):
        raise ValueError("inverse generation limits are invalid")
    if config.long_tail_case_count < config.long_tail_batch_count:
        raise ValueError("long-tail case count is smaller than batch count")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-project-root",
        type=Path,
        default=Path(r"C:\Users\29072\PycharmProjects\AgentSociety"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            _MODULE_ROOT
            / "../data/rag_benchmark/v1/inverse_generation/inverse_v1_0001"
        ),
    )
    parser.add_argument("--model", default="glm-5.3-flash[1m]")
    parser.add_argument("--claude-model-alias", default="haiku")
    parser.add_argument(
        "--claude-cli-script",
        type=Path,
        help="旧版Claude Code的cli.js；提供后自动用node启动兼容模式",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--only-batch", action="append", default=[])
    args = parser.parse_args()
    claude_command = (
        ("node", str(args.claude_cli_script.resolve()))
        if args.claude_cli_script
        else ("claude",)
    )
    report = run_inverse_benchmark_generation(
        InverseGenerationConfig(
            source_project_root=args.source_project_root.resolve(),
            output_root=args.output_root.resolve(),
            model=args.model,
            claude_model_alias=args.claude_model_alias,
            claude_command=claude_command,
            claude_legacy_cli=bool(args.claude_cli_script),
            concurrency=args.concurrency,
            only_batch_ids=tuple(args.only_batch),
        )
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

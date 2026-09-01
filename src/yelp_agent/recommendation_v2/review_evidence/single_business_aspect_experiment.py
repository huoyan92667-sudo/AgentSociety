"""真实测量一家餐厅的14项评论召回、微调模型判断和特征聚合。

这个文件是离线实验入口，不接入在线推荐流程。它分成两个阶段运行：

1. ``retrieve`` 使用项目环境连接Qdrant，执行现有BM25与向量混合召回；
2. ``judge`` 使用显卡环境加载Qwen3基础模型和LoRA增量权重，判断每条候选
   评论的相关度与强度，然后聚合出该商家的14项客观程度。

分成两个阶段是因为项目环境和本地显卡环境的依赖版本不同。中间结果会保存，
模型阶段即使中断也不会重新执行评论召回。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence


DEFAULT_BUSINESS_ID = "ZYRul0i1bhOjirHED6Kd0w"
DEFAULT_BUSINESS_NAME = "SouthHouse"
DEFAULT_BASE_MODEL = Path(r"D:\model\Qwen3-4B-Instruct-2507")
DEFAULT_ADAPTER = Path(
    r"C:\Users\29072\Desktop\important\qwen3_training\outputs\baseline"
)
DEFAULT_DATA_PROJECT = Path(r"C:\Users\29072\PycharmProjects\AgentSociety")
RELEVANCE_WEIGHTS = {0: 0.0, 1: 0.25, 2: 0.65, 3: 1.0}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_output() -> Path:
    return (
        _project_root()
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "review_evidence"
        / "v1"
        / "runs"
        / "single_business_14_aspects"
        / DEFAULT_BUSINESS_ID
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieve = subparsers.add_parser("retrieve", help="执行BM25与向量混合召回")
    retrieve.add_argument("--business-id", default=DEFAULT_BUSINESS_ID)
    retrieve.add_argument("--business-name", default=DEFAULT_BUSINESS_NAME)
    retrieve.add_argument("--data-project", type=Path, default=DEFAULT_DATA_PROJECT)
    retrieve.add_argument("--output-dir", type=Path, default=_default_output())
    retrieve.add_argument("--qdrant-url", default="http://localhost:6333")
    retrieve.add_argument(
        "--embedding-model",
        type=Path,
        default=Path(r"D:\models\Qwen3-Embedding-0.6B"),
    )
    retrieve.add_argument(
        "--embedding-python",
        type=Path,
        default=Path(r"D:\anaconda3\python.exe"),
    )

    judge = subparsers.add_parser("judge", help="执行微调模型判断并聚合14项分数")
    judge.add_argument("--output-dir", type=Path, default=_default_output())
    judge.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    judge.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    judge.add_argument("--batch-size", type=int, default=1)
    judge.add_argument("--cutoff-len", type=int, default=1024)
    judge.add_argument("--max-new-tokens", type=int, default=32)
    judge.add_argument("--half-life-days", type=float, default=730.0)
    judge.add_argument("--useful-alpha", type=float, default=0.2)
    judge.add_argument("--useful-cap", type=float, default=1.2)
    args = parser.parse_args(argv)
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size必须大于0")
    return args


def _load_template() -> dict[str, Any]:
    path = Path(__file__).with_name("training_data") / "teacher_input_templates.v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve_candidates(args: argparse.Namespace) -> Path:
    """复用正式检索器，仅把候选和真实耗时保存下来。"""

    # 延迟导入项目模块，使judge阶段可在独立显卡环境中直接运行本文件。
    from yelp_agent.recommendation_v2.review_evidence.full_reviews import FullReviewStore
    from yelp_agent.recommendation_v2.review_evidence.qdrant_store import (
        QdrantReviewSegmentStore,
    )
    from yelp_agent.recommendation_v2.review_evidence.retrieval import (
        ReviewEvidenceRetriever,
    )
    from yelp_agent.recommendation_v2.review_evidence.schema import (
        PreferenceSearchDescription,
    )
    from yelp_agent.recommendation_v2.review_evidence.segment_vectors import (
        ReviewSegmentVectorStore,
    )
    from yelp_agent.recommendation_v2.review_features.definitions import (
        preference_semantic_anchors,
    )
    from yelp_agent.review_rag.config import load_review_rag_config
    from yelp_agent.semantic_embedding import (
        LocalEmbeddingEnvironment,
        LocalQwenEmbeddingEncoder,
    )

    started = perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    template = _load_template()
    aspect_templates = {item["id"]: item for item in template["aspects"]}
    requirements = []
    for priority, aspect in enumerate(aspect_templates, 1):
        anchors = preference_semantic_anchors(aspect, "higher")
        requirements.append(
            PreferenceSearchDescription(
                requirement_id=aspect,
                requirement_text=aspect_templates[aspect]["name_zh"],
                kind="long_tail",
                priority=priority,
                preference_strength=100,
                positive_descriptions=anchors.satisfying,
                negative_descriptions=anchors.contradicting,
            )
        )

    setup_started = perf_counter()
    config = load_review_rag_config(_project_root() / "configs" / "review_rag.yaml")
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        config.semantic_config().model_copy(update={"batch_size": 16}),
        LocalEmbeddingEnvironment(
            model_path=args.embedding_model,
            python_executable=args.embedding_python,
            device="cuda",
        ),
    )
    store = QdrantReviewSegmentStore.from_url(args.qdrant_url)
    index_root = (
        args.data_project
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "review_evidence"
        / "v1"
        / "index"
    )
    full_review_path = args.data_project / "data" / "processed" / "reviews.parquet"
    retriever = ReviewEvidenceRetriever(
        store=store,
        encoder=encoder,
        segment_vectors=ReviewSegmentVectorStore(index_root / "segment_embeddings.npy"),
        full_reviews=FullReviewStore(full_review_path),
        recall_threshold=0.55,
        acceptance_threshold=0.60,
        direction_margin=0.05,
        recall_each_side=15,
        initial_segment_group_size=15,
        middle_segment_group_size=30,
        final_segment_group_size=60,
        minimum_clear_evidence=5,
        search_concurrency=4,
        enable_bm25=True,
        rrf_k=60,
    )
    setup_ms = (perf_counter() - setup_started) * 1000
    reference_time = datetime.now(UTC)
    retrieval_started = perf_counter()
    try:
        batch = retriever.retrieve_many(
            requirements,
            [args.business_id],
            cutoff_time=reference_time,
        )
    finally:
        retriever.close()
    retrieval_ms = (perf_counter() - retrieval_started) * 1000

    candidates: list[dict[str, Any]] = []
    aspect_counts: dict[str, dict[str, int]] = {}
    for requirement in requirements:
        items = batch.by_requirement[requirement.requirement_id][args.business_id]
        route_counts: Counter[str] = Counter()
        for item in items:
            dense = item.positive_dense_match or item.negative_dense_match
            bm25 = item.positive_bm25_match or item.negative_bm25_match
            route = "both" if dense and bm25 else "dense_only" if dense else "bm25_only"
            route_counts[route] += 1
            # 训练输入最多900字符。短评论使用完整原文；长评论使用带相邻句
            # 上下文的命中片段，避免从开头机械截断而丢失真正命中内容。
            model_text = (
                item.review_text
                if len(item.review_text) <= 900
                else item.matched_segment_text
            )
            candidates.append(
                {
                    "aspect_id": requirement.requirement_id,
                    "aspect_name_zh": requirement.requirement_text,
                    "review_id": item.review_id,
                    "business_id": item.business_id,
                    "user_id": item.user_id,
                    "review_time": item.review_time.isoformat(),
                    "stars": item.stars,
                    "useful": item.useful,
                    "review_text": item.review_text,
                    "matched_segment_text": item.matched_segment_text,
                    "model_review_text": model_text[:900],
                    "positive_similarity": item.positive_similarity,
                    "negative_similarity": item.negative_similarity,
                    "dense_match": dense,
                    "bm25_match": bm25,
                    "retrieval_direction": item.direction,
                }
            )
        aspect_counts[requirement.requirement_id] = {
            "candidate_review_count": len(items),
            "dense_only_count": route_counts["dense_only"],
            "bm25_only_count": route_counts["bm25_only"],
            "both_count": route_counts["both"],
        }

    payload = {
        "experiment": "single_business_14_aspects",
        "business_id": args.business_id,
        "business_name": args.business_name,
        "reference_time": reference_time.isoformat(),
        "retrieval_config": {
            "routes": ["dense_embedding", "bm25"],
            "recall_threshold": 0.55,
            "acceptance_threshold": 0.60,
            "direction_margin": 0.05,
            "recall_each_side": 15,
            "initial_group_size": 15,
            "middle_group_size": 30,
            "final_group_size": 60,
            "rrf_k": 60,
            "description_count": len(requirements) * 4,
        },
        "timing_ms": {
            "retrieval_setup": setup_ms,
            "hybrid_retrieval": retrieval_ms,
            "retrieve_stage_total": (perf_counter() - started) * 1000,
        },
        "retrieval_metrics": batch.metrics.model_dump(mode="json"),
        "aspect_counts": aspect_counts,
        "candidate_relation_count": len(candidates),
        "unique_review_count": len({item["review_id"] for item in candidates}),
        "candidates": candidates,
    }
    path = output_dir / "retrieval.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RETRIEVE] business={args.business_name} candidates={len(candidates)}")
    for aspect, counts in aspect_counts.items():
        print(f"[RETRIEVE] {aspect}: {counts}")
    print(f"[RETRIEVE] elapsed={payload['timing_ms']['retrieve_stage_total']:.1f} ms")
    print(f"[RETRIEVE] saved={path}")
    return path


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _parse_label(raw: str) -> tuple[int | None, int | None]:
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None, None

    def integer(item: Any) -> int | None:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            return item
        if isinstance(item, str) and item.strip().isdigit():
            return int(item.strip())
        return None

    relevance = integer(value.get("relevance")) if isinstance(value, dict) else None
    raw_strength = value.get("strength") if isinstance(value, dict) else None
    strength = None if raw_strength is None else integer(raw_strength)
    if relevance not in {0, 1, 2, 3}:
        return None, None
    if relevance == 0:
        return (0, None) if raw_strength is None else (None, None)
    if strength not in {0, 1, 2, 3, 4}:
        return None, None
    return relevance, strength


def _load_model(base_model: Path, adapter: Path) -> tuple[Any, Any, dict[str, float]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model.eval()
    return model, tokenizer, {
        "model_load_ms": (perf_counter() - started) * 1000,
        "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
    }


def _generate(
    model: Any,
    tokenizer: Any,
    messages: list[list[dict[str, str]]],
    *,
    cutoff_len: int,
    max_new_tokens: int,
) -> tuple[list[str], int]:
    import torch

    rendered = [
        tokenizer.apply_chat_template(
            item, tokenize=False, add_generation_prompt=True
        )
        for item in messages
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cutoff_len,
    )
    input_tokens = int(inputs["attention_mask"].sum().item())
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return (
        tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        input_tokens,
    )


def _useful_weight(useful: int, *, alpha: float, cap: float) -> float:
    """本次实验值：沿用对数轻微加权，并将最高增益限制在20%。"""

    return min(cap, 1.0 + alpha * math.log1p(max(0, useful)))


def judge_and_aggregate(args: argparse.Namespace) -> Path:
    """加载最佳LoRA权重，逐条判断后计算一家商家的14项客观程度。"""

    import torch

    total_started = perf_counter()
    retrieval_path = args.output_dir.resolve() / "retrieval.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    template = _load_template()
    aspects = {item["id"]: item for item in template["aspects"]}
    system_prompt = template["system_prompt"]
    records = retrieval["candidates"]
    messages: list[list[dict[str, str]]] = []
    for record in records:
        aspect = aspects[record["aspect_id"]]
        model_input = {
            "aspect_id": aspect["id"],
            "definition": aspect["definition"],
            "relevance_scale": template["common_relevance_scale"],
            "strength_scale": aspect["strength_scale"],
            "special_rules": aspect["special_rules"],
            "review_text": record["model_review_text"],
        }
        messages.append(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        model_input, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ]
        )

    model, tokenizer, load_metrics = _load_model(args.base_model, args.adapter)
    inference_started = perf_counter()
    input_tokens = 0
    valid_outputs = 0
    judged: list[dict[str, Any]] = []
    total_batches = math.ceil(len(records) / args.batch_size)
    for batch_index, (record_batch, message_batch) in enumerate(
        zip(
            _batches(records, args.batch_size),
            _batches(messages, args.batch_size),
            strict=True,
        ),
        1,
    ):
        outputs, batch_tokens = _generate(
            model,
            tokenizer,
            list(message_batch),
            cutoff_len=args.cutoff_len,
            max_new_tokens=args.max_new_tokens,
        )
        input_tokens += batch_tokens
        for record, raw in zip(record_batch, outputs, strict=True):
            relevance, strength = _parse_label(raw)
            if relevance is not None:
                valid_outputs += 1
            judged.append(
                {
                    **record,
                    "raw_model_output": raw,
                    "relevance": relevance,
                    "strength": strength,
                }
            )
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
            print(f"[JUDGE] batch {batch_index}/{total_batches}")
    inference_ms = (perf_counter() - inference_started) * 1000

    aggregation_started = perf_counter()
    reference_time = datetime.fromisoformat(retrieval["reference_time"])
    aspect_results: dict[str, dict[str, Any]] = {}
    for aspect_id, aspect in aspects.items():
        items = [item for item in judged if item["aspect_id"] == aspect_id]
        effective: list[dict[str, Any]] = []
        for item in items:
            relevance = item["relevance"]
            strength = item["strength"]
            if relevance is None or relevance == 0 or strength is None:
                continue
            review_time = datetime.fromisoformat(item["review_time"])
            age_days = max(
                0.0, (reference_time - review_time).total_seconds() / 86400.0
            )
            time_weight = 2 ** (-age_days / args.half_life_days)
            useful_weight = _useful_weight(
                int(item["useful"]), alpha=args.useful_alpha, cap=args.useful_cap
            )
            evidence_weight = (
                RELEVANCE_WEIGHTS[relevance] * time_weight * useful_weight
            )
            effective.append(
                {
                    **item,
                    "degree_value": strength / 4.0,
                    "time_weight": time_weight,
                    "useful_weight": useful_weight,
                    "evidence_weight": evidence_weight,
                }
            )
        total_weight = sum(item["evidence_weight"] for item in effective)
        degree = (
            sum(item["degree_value"] * item["evidence_weight"] for item in effective)
            / total_weight
            if total_weight > 0
            else None
        )
        representative = sorted(
            effective,
            key=lambda item: (-item["evidence_weight"], item["review_id"]),
        )[:5]
        aspect_results[aspect_id] = {
            "name_zh": aspect["name_zh"],
            "candidate_review_count": len(items),
            "model_relevance_counts": {
                str(level): sum(item["relevance"] == level for item in items)
                for level in range(4)
            },
            "invalid_model_output_count": sum(
                item["relevance"] is None for item in items
            ),
            "effective_review_count": len(effective),
            "evidence_weight_sum": total_weight,
            "degree": degree,
            "score_0_to_100": None if degree is None else degree * 100,
            "representative_review_ids": [
                item["review_id"] for item in representative
            ],
        }
    aggregation_ms = (perf_counter() - aggregation_started) * 1000
    total_ms = (perf_counter() - total_started) * 1000
    retrieve_ms = float(retrieval["timing_ms"]["retrieve_stage_total"])
    result = {
        "experiment": retrieval["experiment"],
        "business_id": retrieval["business_id"],
        "business_name": retrieval["business_name"],
        "reference_time": retrieval["reference_time"],
        "formula": {
            "degree_value": "strength / 4",
            "relevance_weights": RELEVANCE_WEIGHTS,
            "time_weight": f"2 ** (-age_days / {args.half_life_days})",
            "useful_weight": (
                f"min({args.useful_cap}, 1 + {args.useful_alpha} * ln(1 + useful))"
            ),
            "business_aspect_degree": (
                "sum(degree_value * evidence_weight) / sum(evidence_weight)"
            ),
            "unknown_rule": "没有有效评论时degree为null，不使用0.5",
            "useful_rule_status": "本次实验参数，尚未成为正式系统定值",
        },
        "complexity": {
            "query_description_count": retrieval["retrieval_config"][
                "description_count"
            ],
            "candidate_relation_count": len(records),
            "unique_review_count": retrieval["unique_review_count"],
            "model_batch_size": args.batch_size,
            "model_batch_count": total_batches,
            "model_input_tokens": input_tokens,
            "valid_model_output_count": valid_outputs,
        },
        "timing_ms": {
            **retrieval["timing_ms"],
            **load_metrics,
            "model_inference": inference_ms,
            "aggregation": aggregation_ms,
            "judge_stage_total": total_ms,
            "end_to_end_sum": retrieve_ms + total_ms,
        },
        "retrieval_metrics": retrieval["retrieval_metrics"],
        "aspect_results": aspect_results,
        "judgments": judged,
    }
    output_path = args.output_dir.resolve() / "result.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[RESULT] valid={valid_outputs}/{len(records)}")
    for aspect, item in aspect_results.items():
        print(
            f"[RESULT] {aspect}: candidates={item['candidate_review_count']} "
            f"effective={item['effective_review_count']} "
            f"score={item['score_0_to_100']}"
        )
    print(f"[RESULT] inference={inference_ms / 1000:.2f}s")
    print(f"[RESULT] end_to_end={result['timing_ms']['end_to_end_sum'] / 1000:.2f}s")
    print(f"[RESULT] saved={output_path}")
    del model
    torch.cuda.empty_cache()
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "retrieve":
        retrieve_candidates(args)
    else:
        judge_and_aggregate(args)


if __name__ == "__main__":
    main()

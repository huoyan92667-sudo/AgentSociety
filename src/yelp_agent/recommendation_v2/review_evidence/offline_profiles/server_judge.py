"""在4090服务器上批量判断评论，并聚合成商家14项软偏好画像。

本文件会被本地准备程序复制到数据目录，因此不能依赖项目源码。服务器只需
安装 PyTorch、Transformers、PEFT 和 bitsandbytes，然后把整个数据目录上传。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

RELEVANCE_WEIGHTS = {0: 0.0, 1: 0.25, 2: 0.65, 3: 1.0}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--half-life-days", type=float, default=730.0)
    parser.add_argument("--useful-alpha", type=float, default=0.2)
    parser.add_argument("--useful-cap", type=float, default=1.2)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="允许Transformers联网下载基础模型；默认只读服务器本地文件",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.input_dir / "server_output"
    if args.batch_size < 1 or args.cutoff_len < 1 or args.max_new_tokens < 1:
        parser.error("batch size and token limits must be positive")
    if args.half_life_days <= 0 or args.useful_alpha < 0 or args.useful_cap < 1:
        parser.error("aggregation parameters are invalid")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_label(raw: str) -> tuple[int | None, int | None]:
    """接受整数或数字字符串，但仍严格执行训练时的字段约束。"""

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

    if not isinstance(value, dict) or set(value) != {"relevance", "strength"}:
        return None, None
    relevance = integer(value.get("relevance"))
    raw_strength = value.get("strength")
    strength = None if raw_strength is None else integer(raw_strength)
    if relevance not in RELEVANCE_WEIGHTS:
        return None, None
    if relevance == 0:
        return (0, None) if raw_strength is None else (None, None)
    if strength not in {0, 1, 2, 3, 4}:
        return None, None
    return relevance, strength


def _load_previous(path: Path) -> dict[str, dict[str, Any]]:
    """读取断点文件；若最后一行因断电只写了一半，只忽略该残行。"""

    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise ValueError(f"corrupted checkpoint line {index + 1}: {path}")
        if isinstance(item, dict) and item.get("sample_id"):
            result[str(item["sample_id"])] = item
    return result


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = perf_counter()
    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=local_only
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
        args.base_model,
        trust_remote_code=True,
        local_files_only=local_only,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model.eval()
    return (
        model,
        tokenizer,
        {
            "model_load_ms": (perf_counter() - started) * 1000,
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        },
    )


def _generate(
    model: Any,
    tokenizer: Any,
    messages: list[list[dict[str, str]]],
    *,
    cutoff_len: int,
    max_new_tokens: int,
) -> tuple[list[str], int, int]:
    import torch

    rendered = [
        tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
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
    generated_only = generated[:, prompt_length:]
    output_tokens = int((generated_only != tokenizer.pad_token_id).sum().item())
    return (
        tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        input_tokens,
        output_tokens,
    )


def _append_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _useful_weight(useful: int, *, alpha: float, cap: float) -> float:
    # useful只能提供轻微且有上限的增益，不能让热门旧评论垄断画像。
    return min(cap, 1.0 + alpha * math.log1p(max(0, useful)))


def _weighted_variance(values: list[float], weights: list[float], mean: float) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    return (
        sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights))
        / total
    )


def _level(value: float | None, strength_scale: dict[str, str]) -> dict[str, Any]:
    if value is None:
        return {"code": "unknown", "name_zh": "未知", "meaning": "没有有效证据"}
    strength = min(4, max(0, math.floor(value * 4 + 0.5)))
    return {
        "code": str(strength),
        "name_zh": ["很低", "偏低", "一般", "偏高", "很高"][strength],
        "meaning": strength_scale[str(strength)],
    }


def _evidence_view(item: dict[str, Any]) -> dict[str, Any]:
    """只保留后续排序和回答真正需要的证据字段。"""

    return {
        "review_id": item["review_id"],
        "user_id": item["user_id"],
        "review_time": item["review_time"],
        "stars": item["stars"],
        "useful": item["useful"],
        "text": item["full_review_text"],
        "relevance": item["relevance"],
        "strength": item["strength"],
        "evidence_weight": item["evidence_weight"],
    }


def _aggregate(
    *,
    inputs: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    contract: dict[str, Any],
    prepare_manifest: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference_time = datetime.fromisoformat(prepare_manifest["reference_time"])
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=UTC)
    aspects = {item["id"]: item for item in contract["aspects"]}
    businesses = {item["business_id"]: item for item in selection["businesses"]}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in inputs:
        judged = judgments.get(source["sample_id"])
        if judged is None or judged.get("relevance") is None:
            continue
        grouped[(source["business_id"], source["aspect_id"])].append(
            {**source, **judged}
        )

    profiles: list[dict[str, Any]] = []
    counts = prepare_manifest["counts_by_business_and_aspect"]
    for business_id, business in businesses.items():
        aspect_profiles: list[dict[str, Any]] = []
        for aspect_id, aspect in aspects.items():
            candidates = grouped.get((business_id, aspect_id), [])
            effective: list[dict[str, Any]] = []
            for item in candidates:
                relevance, strength = item["relevance"], item["strength"]
                if relevance == 0 or strength is None:
                    continue
                review_time = datetime.fromisoformat(item["review_time"])
                if review_time.tzinfo is None:
                    review_time = review_time.replace(tzinfo=UTC)
                age_days = max(
                    0.0, (reference_time - review_time).total_seconds() / 86400
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

            # 同一用户针对同一商家同一特征只保留最有力的一条，防止重复刷评。
            best_by_user: dict[str, dict[str, Any]] = {}
            for item in effective:
                previous = best_by_user.get(item["user_id"])
                if (
                    previous is None
                    or item["evidence_weight"] > previous["evidence_weight"]
                ):
                    best_by_user[item["user_id"]] = item
            effective = list(best_by_user.values())
            weights = [item["evidence_weight"] for item in effective]
            degrees = [item["degree_value"] for item in effective]
            total_weight = sum(weights)
            degree = (
                sum(value * weight for value, weight in zip(degrees, weights))
                / total_weight
                if total_weight > 0
                else None
            )
            ess = (
                total_weight**2 / sum(weight**2 for weight in weights)
                if weights
                else 0.0
            )
            diversity = 0.8 + 0.2 * min(1.0, len(best_by_user) / 5.0)
            sufficiency = (1.0 - math.exp(-ess / 5.0)) * diversity
            if degree is None or ess < 3:
                controversy = None
            else:
                controversy = min(
                    1.0, 4.0 * _weighted_variance(degrees, weights, degree)
                )
            ordered = sorted(
                effective,
                key=lambda item: (-item["evidence_weight"], item["review_id"]),
            )
            high = [_evidence_view(item) for item in ordered if item["strength"] >= 3][
                :5
            ]
            low = [_evidence_view(item) for item in ordered if item["strength"] <= 1][
                :5
            ]
            middle = [
                _evidence_view(item) for item in ordered if item["strength"] == 2
            ][:3]
            strong_evidence = [item for item in effective if item["relevance"] >= 2]
            strong_user_count = len({item["user_id"] for item in strong_evidence})
            unusable_reasons: list[str] = []
            if degree is None:
                unusable_reasons.append("没有有效评论")
            if len(strong_evidence) < 3:
                unusable_reasons.append("相关程度为2或3的评论少于3条")
            if strong_user_count < 3:
                unusable_reasons.append("强相关证据来自少于3个不同用户")
            if sufficiency <= 0.3:
                unusable_reasons.append("证据充分程度不足")
            usable_for_ranking = not unusable_reasons
            retrieve_count = counts[business_id][aspect_id]
            aspect_profiles.append(
                {
                    "aspect_id": aspect_id,
                    "aspect_name_zh": aspect["name_zh"],
                    "degree": degree,
                    "degree_0_to_100": None if degree is None else degree * 100,
                    "degree_level": _level(degree, aspect["strength_scale"]),
                    "evidence_sufficiency": sufficiency,
                    "evidence_sufficiency_level": (
                        "不足"
                        if sufficiency <= 0.3
                        else "一般"
                        if sufficiency <= 0.7
                        else "充分"
                    ),
                    "controversy": controversy,
                    "controversy_level": (
                        "未知"
                        if controversy is None
                        else "低"
                        if controversy <= 0.25
                        else "中"
                        if controversy <= 0.6
                        else "高"
                    ),
                    "business_total_review_count": business["actual_review_count"],
                    "retrieved_candidate_count": retrieve_count[
                        "unique_candidate_count"
                    ],
                    "model_related_review_count": len(effective),
                    "unique_evidence_user_count": len(best_by_user),
                    "strong_evidence_count": len(strong_evidence),
                    "unique_strong_user_count": strong_user_count,
                    "usable_for_ranking": usable_for_ranking,
                    "ranking_degree": degree if usable_for_ranking else None,
                    "unusable_reasons": unusable_reasons,
                    "effective_sample_size": ess,
                    "evidence_weight_sum": total_weight,
                    "retrieval_limit_reached": {
                        "high": retrieve_count["high_limit_reached"],
                        "low": retrieve_count["low_limit_reached"],
                    },
                    "evidence": {
                        "high_degree": high,
                        "low_degree": low,
                        "middle_degree": middle,
                        "conditional": [],
                        "conditional_status": "not_available_in_current_model",
                    },
                }
            )
        profiles.append({"business": business, "aspects": aspect_profiles})

    return {
        "schema_version": "1.0",
        "reference_time": reference_time.isoformat(),
        "formula": {
            "degree_value": "strength / 4",
            "relevance_weights": RELEVANCE_WEIGHTS,
            "time_weight": f"2 ** (-age_days / {args.half_life_days})",
            "useful_weight": f"min({args.useful_cap}, 1 + {args.useful_alpha} * ln(1 + useful))",
            "business_aspect_degree": "sum(degree_value * evidence_weight) / sum(evidence_weight)",
            "effective_sample_size": "sum(weight)^2 / sum(weight^2)",
            "sufficiency": "(1 - exp(-ESS / 5)) * user_diversity_factor",
            "controversy": "min(1, 4 * weighted_variance(degree_value)); ESS<3时未知",
            "usable_for_ranking": (
                "degree非空、至少3条relevance>=2证据、至少3个不同用户、sufficiency>0.3"
            ),
        },
        "business_profiles": profiles,
    }


def _run_flat(args: argparse.Namespace) -> Path:
    total_started = perf_counter()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "model_inputs.jsonl"
    prepare_manifest = _read_json(input_dir / "prepare_manifest.json")
    expected_hash = prepare_manifest.get("sha256", {}).get("model_inputs")
    if expected_hash and _sha256(input_path) != expected_hash:
        raise ValueError("model_inputs.jsonl hash does not match prepare_manifest.json")
    inputs = _read_jsonl(input_path)
    if len({item["sample_id"] for item in inputs}) != len(inputs):
        raise ValueError("model_inputs.jsonl contains duplicate sample_id values")

    checkpoint_path = output_dir / "predictions.partial.jsonl"
    previous = _load_previous(checkpoint_path)
    # 之前已经得到合法JSON的样本不再计算；非法输出会在本次重新尝试一次。
    pending = [
        item
        for item in inputs
        if previous.get(item["sample_id"], {}).get("relevance") is None
    ]
    pending.sort(key=lambda item: (item.get("input_char_count", 0), item["sample_id"]))
    print(
        f"[JUDGE] total={len(inputs)} resumed_valid={len(inputs) - len(pending)} pending={len(pending)}"
    )

    load_metrics: dict[str, Any] = {"model_load_ms": 0.0}
    inference_ms = 0.0
    input_tokens = 0
    output_tokens = 0
    if pending:
        model, tokenizer, load_metrics = _load_model(args)
        started = perf_counter()
        total_batches = math.ceil(len(pending) / args.batch_size)
        for batch_index, batch in enumerate(_batches(pending, args.batch_size), 1):
            outputs, batch_input_tokens, batch_output_tokens = _generate(
                model,
                tokenizer,
                [item["messages"] for item in batch],
                cutoff_len=args.cutoff_len,
                max_new_tokens=args.max_new_tokens,
            )
            input_tokens += batch_input_tokens
            output_tokens += batch_output_tokens
            checkpoint_rows: list[dict[str, Any]] = []
            for item, raw in zip(batch, outputs, strict=True):
                relevance, strength = _parse_label(raw)
                checkpoint_rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "raw_model_output": raw,
                        "relevance": relevance,
                        "strength": strength,
                        "judged_at": datetime.now(UTC).isoformat(),
                    }
                )
            _append_checkpoint(checkpoint_path, checkpoint_rows)
            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == total_batches
            ):
                print(f"[JUDGE] batch={batch_index}/{total_batches}")
        inference_ms = (perf_counter() - started) * 1000

    latest = _load_previous(checkpoint_path)
    ordered_judgments = [
        latest.get(
            item["sample_id"],
            {
                "sample_id": item["sample_id"],
                "raw_model_output": None,
                "relevance": None,
                "strength": None,
            },
        )
        for item in inputs
    ]
    final_judgments_path = output_dir / "model_judgments.jsonl"
    with final_judgments_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in ordered_judgments:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    aggregation_started = perf_counter()
    profile = _aggregate(
        inputs=inputs,
        judgments={item["sample_id"]: item for item in ordered_judgments},
        selection=_read_json(input_dir / "selection.json"),
        contract=_read_json(input_dir / "model_contract.v1.json"),
        prepare_manifest=prepare_manifest,
        args=args,
    )
    profile_path = output_dir / "business_aspect_profiles.json"
    _write_json(profile_path, profile)
    aggregation_ms = (perf_counter() - aggregation_started) * 1000
    invalid = [item for item in ordered_judgments if item.get("relevance") is None]
    _write_json(output_dir / "invalid_outputs.json", invalid)
    manifest = {
        "schema_version": "1.0",
        "completed_at": datetime.now(UTC).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "input_count": len(inputs),
        "valid_output_count": len(inputs) - len(invalid),
        "invalid_output_count": len(invalid),
        "newly_inferred_count": len(pending),
        "batch_size": args.batch_size,
        "input_tokens_this_run": input_tokens,
        "output_tokens_this_run": output_tokens,
        "timing_ms": {
            **load_metrics,
            "inference_this_run": inference_ms,
            "aggregation": aggregation_ms,
            "total_this_run": (perf_counter() - total_started) * 1000,
        },
        "throughput_relations_per_second": (
            len(pending) / (inference_ms / 1000) if inference_ms > 0 else None
        ),
        "files": {
            "checkpoint": checkpoint_path.name,
            "judgments": final_judgments_path.name,
            "profiles": profile_path.name,
            "invalid_outputs": "invalid_outputs.json",
        },
    }
    manifest_path = output_dir / "judge_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"[DONE] valid={manifest['valid_output_count']} invalid={len(invalid)}")
    print(f"[DONE] profiles={profile_path}")
    return manifest_path


def _messages_for_compact_input(
    record: dict[str, Any],
    *,
    contract: dict[str, Any],
    aspects: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """根据固定合同还原与训练时完全相同的两条模型消息。"""

    aspect = aspects[record["aspect_id"]]
    model_input = {
        "aspect_id": aspect["id"],
        "definition": aspect["definition"],
        "relevance_scale": contract["common_relevance_scale"],
        "strength_scale": aspect["strength_scale"],
        "special_rules": aspect["special_rules"],
        "review_text": record["model_review_text"],
    }
    return [
        {"role": "system", "content": contract["system_prompt"]},
        {
            "role": "user",
            "content": json.dumps(
                model_input, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]


def _shard_directories(input_dir: Path) -> list[Path]:
    values = sorted(
        path for path in (input_dir / "input_shards").glob("shard_*") if path.is_dir()
    )
    if not values:
        raise FileNotFoundError("input_shards does not contain any shard directories")
    return values


def _validated_shard_inputs(
    shard_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(shard_dir / "shard_manifest.json")
    input_path = shard_dir / manifest["files"]["model_inputs"]
    if _sha256(input_path) != manifest["sha256"]["model_inputs"]:
        raise ValueError(f"model input hash mismatch: {shard_dir.name}")
    inputs = _read_jsonl(input_path)
    if len(inputs) != manifest["candidate_relation_count"]:
        raise ValueError(f"model input count mismatch: {shard_dir.name}")
    return inputs, manifest


def _run_sharded(args: argparse.Namespace) -> Path:
    """一次加载模型后依次处理多个商家分片，并逐分片聚合画像。"""

    total_started = perf_counter()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_manifest = _read_json(input_dir / "prepare_manifest.json")
    contract = _read_json(input_dir / "model_contract.v1.json")
    aspects = {item["id"]: item for item in contract["aspects"]}
    shard_dirs = _shard_directories(input_dir)
    if len(shard_dirs) != prepare_manifest["shard_count"]:
        raise ValueError("shard count does not match prepare manifest")

    checkpoint_path = output_dir / "predictions.partial.jsonl"
    previous = _load_previous(checkpoint_path)
    total_count = int(prepare_manifest["candidate_relation_count"])
    print(
        f"[JUDGE] shards={len(shard_dirs)} total={total_count} "
        f"checkpoint_rows={len(previous)}"
    )
    model = None
    tokenizer = None
    load_metrics: dict[str, Any] = {"model_load_ms": 0.0}
    inference_ms = 0.0
    input_tokens = 0
    output_tokens = 0
    newly_inferred = 0
    for shard_index, shard_dir in enumerate(shard_dirs, 1):
        inputs, _ = _validated_shard_inputs(shard_dir)
        pending = [
            item
            for item in inputs
            if previous.get(item["sample_id"], {}).get("relevance") is None
        ]
        pending.sort(
            key=lambda item: (item.get("input_char_count", 0), item["sample_id"])
        )
        print(
            f"[SHARD {shard_index}/{len(shard_dirs)}] total={len(inputs)} "
            f"pending={len(pending)}"
        )
        if not pending:
            continue
        if model is None or tokenizer is None:
            model, tokenizer, load_metrics = _load_model(args)
        shard_started = perf_counter()
        total_batches = math.ceil(len(pending) / args.batch_size)
        for batch_index, batch in enumerate(_batches(pending, args.batch_size), 1):
            outputs, batch_input_tokens, batch_output_tokens = _generate(
                model,
                tokenizer,
                [
                    _messages_for_compact_input(
                        item, contract=contract, aspects=aspects
                    )
                    for item in batch
                ],
                cutoff_len=args.cutoff_len,
                max_new_tokens=args.max_new_tokens,
            )
            input_tokens += batch_input_tokens
            output_tokens += batch_output_tokens
            checkpoint_rows: list[dict[str, Any]] = []
            for item, raw in zip(batch, outputs, strict=True):
                relevance, strength = _parse_label(raw)
                checkpoint_rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "raw_model_output": raw,
                        "relevance": relevance,
                        "strength": strength,
                        "judged_at": datetime.now(UTC).isoformat(),
                    }
                )
            _append_checkpoint(checkpoint_path, checkpoint_rows)
            previous.update({item["sample_id"]: item for item in checkpoint_rows})
            newly_inferred += len(batch)
            if (
                batch_index == 1
                or batch_index % 50 == 0
                or batch_index == total_batches
            ):
                print(f"[SHARD {shard_index}] batch={batch_index}/{total_batches}")
        shard_ms = (perf_counter() - shard_started) * 1000
        inference_ms += shard_ms
        print(f"[SHARD {shard_index}] inference_ms={shard_ms:.1f}")

    latest = _load_previous(checkpoint_path)
    final_judgments_path = output_dir / "model_judgments.jsonl"
    invalid: list[dict[str, Any]] = []
    ordered_judgment_count = 0
    with final_judgments_path.open("w", encoding="utf-8", newline="\n") as handle:
        for shard_dir in shard_dirs:
            inputs, _ = _validated_shard_inputs(shard_dir)
            for item in inputs:
                judged = latest.get(
                    item["sample_id"],
                    {
                        "sample_id": item["sample_id"],
                        "raw_model_output": None,
                        "relevance": None,
                        "strength": None,
                    },
                )
                handle.write(
                    json.dumps(judged, ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
                ordered_judgment_count += 1
                if judged.get("relevance") is None:
                    invalid.append(judged)
    if ordered_judgment_count != total_count:
        raise ValueError("final judgment count does not match prepared input count")

    aggregation_started = perf_counter()
    profile_shard_dir = output_dir / "profile_shards"
    profile_shard_dir.mkdir(parents=True, exist_ok=True)
    all_profiles: list[dict[str, Any]] = []
    formula: dict[str, Any] | None = None
    for shard_dir in shard_dirs:
        inputs, shard_manifest = _validated_shard_inputs(shard_dir)
        reviews_path = shard_dir / shard_manifest["files"]["reviews"]
        if _sha256(reviews_path) != shard_manifest["sha256"]["reviews"]:
            raise ValueError(f"review text hash mismatch: {shard_dir.name}")
        review_texts = {
            item["review_id"]: item["full_review_text"]
            for item in _read_jsonl(reviews_path)
        }
        enriched = [
            {**item, "full_review_text": review_texts[item["review_id"]]}
            for item in inputs
        ]
        shard_judgments = {
            item["sample_id"]: latest[item["sample_id"]]
            for item in inputs
            if item["sample_id"] in latest
        }
        profile = _aggregate(
            inputs=enriched,
            judgments=shard_judgments,
            selection=_read_json(shard_dir / shard_manifest["files"]["selection"]),
            contract=contract,
            prepare_manifest={
                "reference_time": prepare_manifest["reference_time"],
                "counts_by_business_and_aspect": shard_manifest[
                    "counts_by_business_and_aspect"
                ],
            },
            args=args,
        )
        formula = profile["formula"]
        all_profiles.extend(profile["business_profiles"])
        _write_json(profile_shard_dir / f"{shard_dir.name}.json", profile)
        print(
            f"[AGGREGATE] {shard_dir.name} "
            f"businesses={len(profile['business_profiles'])}"
        )

    profile_path = output_dir / "business_aspect_profiles.json"
    _write_json(
        profile_path,
        {
            "schema_version": "2.0",
            "reference_time": prepare_manifest["reference_time"],
            "formula": formula,
            "business_profiles": all_profiles,
        },
    )
    aggregation_ms = (perf_counter() - aggregation_started) * 1000
    _write_json(output_dir / "invalid_outputs.json", invalid)
    manifest = {
        "schema_version": "2.0",
        "completed_at": datetime.now(UTC).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "input_count": total_count,
        "valid_output_count": total_count - len(invalid),
        "invalid_output_count": len(invalid),
        "newly_inferred_count": newly_inferred,
        "shard_count": len(shard_dirs),
        "business_count": len(all_profiles),
        "batch_size": args.batch_size,
        "input_tokens_this_run": input_tokens,
        "output_tokens_this_run": output_tokens,
        "timing_ms": {
            **load_metrics,
            "inference_this_run": inference_ms,
            "aggregation": aggregation_ms,
            "total_this_run": (perf_counter() - total_started) * 1000,
        },
        "throughput_relations_per_second": (
            newly_inferred / (inference_ms / 1000) if inference_ms > 0 else None
        ),
        "files": {
            "checkpoint": checkpoint_path.name,
            "judgments": final_judgments_path.name,
            "profiles": profile_path.name,
            "profile_shards": profile_shard_dir.name,
            "invalid_outputs": "invalid_outputs.json",
        },
    }
    manifest_path = output_dir / "judge_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        f"[DONE] businesses={len(all_profiles)} valid={manifest['valid_output_count']} "
        f"invalid={len(invalid)}"
    )
    print(f"[DONE] profiles={profile_path}")
    return manifest_path


def run(args: argparse.Namespace) -> Path:
    if (args.input_dir.resolve() / "input_shards").is_dir():
        return _run_sharded(args)
    return _run_flat(args)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

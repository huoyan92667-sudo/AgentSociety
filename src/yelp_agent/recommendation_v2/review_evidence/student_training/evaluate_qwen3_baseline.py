"""在本地用原始Qwen3 4B评测同一份validation和test数据。

这个脚本由服务器上的微调模型评测脚本改造而来。两者使用完全相同的：

* validation/test数据；
* 对话模板渲染方式；
* 确定性生成参数；
* JSON、相关程度、强度和联合准确率计算方式。

唯一差别是本脚本不加载LoRA增量文件，只加载本地基础模型，因此得到的是
微调前基线。模型使用bitsandbytes NF4四位加载，以适应8GB显存设备。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

RELEVANCE_LABELS = [0, 1, 2, 3]
STRENGTH_LABELS = [None, 0, 1, 2, 3, 4]
INVALID = "__INVALID__"


def _default_paths() -> tuple[Path, Path, Path]:
    recommendation_root = Path(__file__).resolve().parents[2]
    data_dir = (
        recommendation_root
        / "data"
        / "student_training"
        / "qwen3_4b_teacher_v2"
        / "v1"
    )
    output_dir = data_dir / "evaluation" / "original_model"
    base_model = Path(r"D:\model\Qwen3-4B-Instruct-2507")
    return base_model, data_dir, output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base_model, data_dir, output_dir = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=base_model)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--split",
        choices=["validation", "test", "both"],
        default="both",
        help="要评测的数据；默认依次评测validation和test。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="8GB显存默认一次处理1条；确认余量后可以手动提高。",
    )
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="只评测前N条，用于正式全量评测前试跑。",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size必须大于0")
    if args.cutoff_len < 1 or args.max_new_tokens < 1:
        parser.error("--cutoff-len和--max-new-tokens必须大于0")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples必须大于0")
    return args


def load_jsonl(path: Path, max_samples: Optional[int] = None) -> list[dict[str, Any]]:
    """读取训练格式数据，并确认最后一条消息确实是教师答案。"""

    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}不是合法JSON: {exc}") from exc
            messages = value.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"{path}:{line_number}的messages结构不正确")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_number}最后一条消息不是教师答案")
            samples.append(value)
            if max_samples is not None and len(samples) >= max_samples:
                break
    if not samples:
        raise ValueError(f"评测文件为空: {path}")
    return samples


def parse_gold(sample: dict[str, Any]) -> dict[str, Any]:
    """从最后一条assistant消息读取教师标签。"""

    gold = json.loads(sample["messages"][-1]["content"])
    return {"relevance": gold["relevance"], "strength": gold["strength"]}


def strict_parse_prediction(
    text: str,
) -> tuple[Optional[dict[str, Any]], bool, bool, bool]:
    """严格解析模型原文，不提取代码块，也不自动修复JSON。"""

    try:
        parsed = json.loads(text.strip())
    except Exception:
        return None, False, False, False
    if not isinstance(parsed, dict):
        return parsed, True, False, False

    exact_keys = set(parsed) == {"relevance", "strength"}
    relevance = parsed.get("relevance")
    strength = parsed.get("strength")
    relevance_valid = (
        isinstance(relevance, int)
        and not isinstance(relevance, bool)
        and relevance in RELEVANCE_LABELS
    )
    strength_valid = strength is None or (
        isinstance(strength, int)
        and not isinstance(strength, bool)
        and strength in [0, 1, 2, 3, 4]
    )
    rule_violation = bool(relevance == 0 and strength is not None)
    schema_valid = bool(
        exact_keys and relevance_valid and strength_valid and not rule_violation
    )
    return parsed, True, schema_valid, rule_violation


def normalize_prediction_for_scoring(parsed: Any) -> tuple[Any, Any]:
    """将可明确理解的字符串标签转成数值，只用于内容正确率。

    格式判断仍然使用strict_parse_prediction。例如模型返回字符串"3"时，
    schema_valid依然为False；但它表达的类别没有歧义，所以相关程度、强度和
    联合正确率使用数值3参与计算。无法明确转换的内容继续记为INVALID。
    """

    if not isinstance(parsed, dict):
        return INVALID, INVALID

    def normalize(value: Any, allowed: set[int], *, allow_null: bool) -> Any:
        if value is None:
            return None if allow_null else INVALID
        if isinstance(value, bool):
            return INVALID
        if isinstance(value, int):
            return value if value in allowed else INVALID
        if isinstance(value, str):
            stripped = value.strip().lower()
            if allow_null and stripped == "null":
                return None
            if stripped in {str(item) for item in allowed}:
                return int(stripped)
        return INVALID

    return (
        normalize(parsed.get("relevance"), set(RELEVANCE_LABELS), allow_null=False),
        normalize(parsed.get("strength"), {0, 1, 2, 3, 4}, allow_null=True),
    )


def macro_f1(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    labels: Sequence[Any],
) -> tuple[float, dict[str, dict[str, float | int]]]:
    """逐类别计算准确与召回，再取各类别F1的简单平均。"""

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(
            1 for truth, pred in zip(gold, predicted) if truth == label and pred == label
        )
        false_positive = sum(
            1 for truth, pred in zip(gold, predicted) if truth != label and pred == label
        )
        false_negative = sum(
            1 for truth, pred in zip(gold, predicted) if truth == label and pred != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        key = "null" if label is None else str(label)
        per_class[key] = {
            "support": sum(1 for truth in gold if truth == label),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
    return sum(f1_values) / len(f1_values), per_class


def accuracy(gold: Sequence[Any], predicted: Sequence[Any]) -> float:
    if not gold:
        return 0.0
    return sum(truth == pred for truth, pred in zip(gold, predicted)) / len(gold)


def confusion_matrix(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    gold_labels: Sequence[Any],
) -> dict[str, dict[str, int]]:
    """输出以真实类别为行、预测类别为列的混淆数量。"""

    predicted_labels = list(gold_labels)
    if any(value == INVALID for value in predicted):
        predicted_labels.append(INVALID)
    matrix: dict[str, dict[str, int]] = {}
    for gold_label in gold_labels:
        gold_key = "null" if gold_label is None else str(gold_label)
        matrix[gold_key] = {}
        for predicted_label in predicted_labels:
            predicted_key = "null" if predicted_label is None else str(predicted_label)
            matrix[gold_key][predicted_key] = sum(
                1
                for truth, pred in zip(gold, predicted)
                if truth == gold_label and pred == predicted_label
            )
    return matrix


def build_model_and_tokenizer(base_model: Path) -> tuple[Any, Any]:
    """只从D盘加载原始模型，不访问网络，也不加载任何LoRA文件。"""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "缺少本地推理依赖。请安装transformers、accelerate、bitsandbytes和torch。"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用CUDA显卡；本脚本的四位加载需要NVIDIA GPU")
    base_model = base_model.resolve()
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"没有找到本地基础模型: {base_model}")

    print(f"[INFO] Local base model: {base_model}")
    print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
    print("[INFO] LoRA adapter: disabled (original-model baseline)")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(
        "[INFO] GPU memory after load: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated, "
        f"{torch.cuda.memory_reserved() / 1024**3:.2f} GiB reserved"
    )
    return model, tokenizer


def batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate_batch(
    model: Any,
    tokenizer: Any,
    samples: Sequence[dict[str, Any]],
    cutoff_len: int,
    max_new_tokens: int,
) -> list[str]:
    """删除教师答案后，以贪心方式生成一批原始模型预测。"""

    import torch

    rendered: list[str] = []
    for sample in samples:
        rendered.append(
            tokenizer.apply_chat_template(
                sample["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cutoff_len,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
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
    return tokenizer.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def evaluate_split(
    *,
    split: str,
    model: Any,
    tokenizer: Any,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    cutoff_len: int,
    max_new_tokens: int,
    max_samples: Optional[int],
) -> dict[str, Any]:
    """运行一个集合并保存逐条预测、汇总指标和混淆矩阵。"""

    input_path = data_dir / f"{split}.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(f"缺少评测数据: {input_path}")
    samples = load_jsonl(input_path, max_samples=max_samples)
    print(f"\n[INFO] Evaluating {split}: {len(samples)} samples")
    records: list[dict[str, Any]] = []
    total_batches = math.ceil(len(samples) / batch_size)
    for batch_index, batch_samples in enumerate(batches(samples, batch_size), 1):
        outputs = generate_batch(
            model,
            tokenizer,
            batch_samples,
            cutoff_len,
            max_new_tokens,
        )
        for sample, raw_output in zip(batch_samples, outputs):
            gold = parse_gold(sample)
            parsed, json_valid, schema_valid, rule_violation = strict_parse_prediction(
                raw_output
            )
            predicted_relevance, predicted_strength = normalize_prediction_for_scoring(
                parsed
            )
            value_parseable = (
                predicted_relevance != INVALID and predicted_strength != INVALID
            )
            records.append(
                {
                    "gold": gold,
                    "prediction_raw": raw_output,
                    "prediction_parsed": parsed,
                    "json_valid": json_valid,
                    "schema_valid": schema_valid,
                    "rule_violation": rule_violation,
                    "value_parseable": value_parseable,
                    "prediction_for_scoring": {
                        "relevance": predicted_relevance,
                        "strength": predicted_strength,
                    },
                    "relevance_correct": predicted_relevance == gold["relevance"],
                    "strength_correct": predicted_strength == gold["strength"],
                    "joint_correct": (
                        predicted_relevance == gold["relevance"]
                        and predicted_strength == gold["strength"]
                    ),
                    "messages": sample["messages"][:-1],
                }
            )
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
            print(
                f"[{split}] batch {batch_index}/{total_batches} "
                f"({min(batch_index * batch_size, len(samples))}/{len(samples)})"
            )

    gold_relevance = [record["gold"]["relevance"] for record in records]
    predicted_relevance = [
        record["prediction_for_scoring"]["relevance"]
        for record in records
    ]
    gold_strength = [record["gold"]["strength"] for record in records]
    predicted_strength = [
        record["prediction_for_scoring"]["strength"]
        for record in records
    ]
    relevance_macro_f1, relevance_per_class = macro_f1(
        gold_relevance,
        predicted_relevance,
        RELEVANCE_LABELS,
    )
    strength_macro_f1, strength_per_class = macro_f1(
        gold_strength,
        predicted_strength,
        STRENGTH_LABELS,
    )
    strength_errors = [
        abs(truth - predicted)
        for truth, predicted in zip(gold_strength, predicted_strength)
        if isinstance(truth, int)
        and not isinstance(truth, bool)
        and isinstance(predicted, int)
        and not isinstance(predicted, bool)
    ]
    metrics = {
        "split": split,
        "num_samples": len(records),
        "json_valid_rate": sum(record["json_valid"] for record in records)
        / len(records),
        "schema_valid_rate": sum(record["schema_valid"] for record in records)
        / len(records),
        "value_parseable_rate": sum(record["value_parseable"] for record in records)
        / len(records),
        "rule_violation_rate": sum(record["rule_violation"] for record in records)
        / len(records),
        "relevance_accuracy": accuracy(gold_relevance, predicted_relevance),
        "relevance_macro_f1": relevance_macro_f1,
        "strength_accuracy": accuracy(gold_strength, predicted_strength),
        "strength_macro_f1": strength_macro_f1,
        "joint_accuracy": sum(record["joint_correct"] for record in records)
        / len(records),
        "strength_mae_numeric_pairs": (
            sum(strength_errors) / len(strength_errors) if strength_errors else None
        ),
        "strength_mae_num_pairs": len(strength_errors),
        "relevance_per_class": relevance_per_class,
        "strength_per_class": strength_per_class,
        "relevance_confusion_matrix": confusion_matrix(
            gold_relevance,
            predicted_relevance,
            RELEVANCE_LABELS,
        ),
        "strength_confusion_matrix": confusion_matrix(
            gold_strength,
            predicted_strength,
            STRENGTH_LABELS,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split}_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    metrics_path = output_dir / f"{split}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_metrics(metrics)
    print(f"[INFO] Predictions saved to: {predictions_path}")
    print(f"[INFO] Metrics saved to: {metrics_path}")
    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print(f"{metrics['split'].upper()} RESULTS")
    print("=" * 64)
    print(f"Samples                : {metrics['num_samples']}")
    print(f"JSON valid rate        : {metrics['json_valid_rate']:.4f}")
    print(f"Schema valid rate      : {metrics['schema_valid_rate']:.4f}")
    print(f"Value parseable rate   : {metrics['value_parseable_rate']:.4f}")
    print(f"Rule violation rate    : {metrics['rule_violation_rate']:.4f}")
    print(f"Relevance accuracy     : {metrics['relevance_accuracy']:.4f}")
    print(f"Relevance macro-F1     : {metrics['relevance_macro_f1']:.4f}")
    print(f"Strength accuracy      : {metrics['strength_accuracy']:.4f}")
    print(f"Strength macro-F1      : {metrics['strength_macro_f1']:.4f}")
    print(f"Joint accuracy         : {metrics['joint_accuracy']:.4f}")
    mae = metrics["strength_mae_numeric_pairs"]
    if mae is None:
        print("Strength MAE            : N/A")
    else:
        print(
            f"Strength MAE            : {mae:.4f} "
            f"(on {metrics['strength_mae_num_pairs']} numeric pairs)"
        )
    print("=" * 64 + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少PyTorch，无法运行本地模型评测") from exc
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = build_model_and_tokenizer(args.base_model)
    splits = ["validation", "test"] if args.split == "both" else [args.split]
    summary: dict[str, Any] = {
        "evaluation_kind": "original_model_baseline",
        "base_model": str(args.base_model.resolve()),
        "adapter": None,
        "cutoff_len": args.cutoff_len,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "splits": {},
    }
    for split in splits:
        summary["splits"][split] = evaluate_split(
            split=split,
            model=model,
            tokenizer=tokenizer,
            data_dir=args.data_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            batch_size=args.batch_size,
            cutoff_len=args.cutoff_len,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir.resolve() / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

"""Smoke-test a local Qwen3 Reranker without starting the Agent runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


INSTRUCTION = (
    "Given a Yelp user request, judge whether the candidate business matches "
    "the requested category, occasion, atmosphere, price, and experience "
    "preferences. Use only facts present in the business document."
)
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements '
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cross_encoder_v1/smoke.json"),
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - started) * 1000.0

    false_token_id = tokenizer.convert_tokens_to_ids("no")
    true_token_id = tokenizer.convert_tokens_to_ids("yes")
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    def format_pair(query: str, document: str) -> str:
        return (
            f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def prepare(
        pairs: list[tuple[str, str]],
    ) -> tuple[dict[str, object], list[int], list[int]]:
        values = [format_pair(query, document) for query, document in pairs]
        raw = tokenizer(values, padding=False, truncation=False)["input_ids"]
        raw_lengths = [
            len(value) + len(prefix_tokens) + len(suffix_tokens) for value in raw
        ]
        encoded = tokenizer(
            values,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=args.max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for index, value in enumerate(encoded["input_ids"]):
            encoded["input_ids"][index] = prefix_tokens + value + suffix_tokens
        batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
        token_counts = [
            int(value) for value in batch["attention_mask"].sum(dim=1).tolist()
        ]
        return (
            {key: value.to(args.device) for key, value in batch.items()},
            token_counts,
            raw_lengths,
        )

    @torch.inference_mode()
    def score(
        pairs: list[tuple[str, str]],
    ) -> tuple[list[float], list[int], list[int], float, float]:
        inputs, token_counts, raw_lengths = prepare(pairs)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        outputs = model(**inputs, logits_to_keep=1)
        logits = outputs.logits[:, -1, :]
        pair_logits = torch.stack(
            [logits[:, false_token_id], logits[:, true_token_id]],
            dim=1,
        )
        probabilities = torch.softmax(pair_logits.float(), dim=1)[:, 1]
        if args.device == "cuda":
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_mb = 0.0
        latency_ms = (time.perf_counter() - inference_started) * 1000.0
        values = [float(value) for value in probabilities.cpu()]
        return values, token_counts, raw_lengths, latency_ms, peak_mb

    query = "I want a quiet romantic steakhouse for a date, with reservations."
    documents = [
        (
            "Business name: Candlelight Steak House\n"
            "Categories: Restaurants, Steakhouses\n"
            "Structured attributes: Ambience romantic=True; Noise Level=quiet; "
            "Restaurants Reservations=True"
        ),
        (
            "Business name: Stadium Sports Bar\n"
            "Categories: Sports Bars, Nightlife, Burgers\n"
            "Structured attributes: Ambience casual=True; Noise Level=loud; "
            "Restaurants Reservations=False"
        ),
        (
            "Business name: Central Cafe\n"
            "Categories: Coffee & Tea, Cafes\n"
            "Structured attributes: Outdoor Seating=True; Wi Fi=free"
        ),
    ]
    base_pairs = [(query, document) for document in documents]
    scores, tokens, _, initial_latency, initial_peak = score(base_pairs)
    benchmark_rows = []
    for batch_size in (1, 4, 8, 20):
        pairs = [base_pairs[index % len(base_pairs)] for index in range(batch_size)]
        values, token_counts, raw_lengths, latency_ms, peak_mb = score(pairs)
        benchmark_rows.append(
            {
                "batch_size": batch_size,
                "latency_ms": latency_ms,
                "pairs_per_second": batch_size / (latency_ms / 1000.0),
                "total_tokens": sum(token_counts),
                "maximum_tokens": max(token_counts),
                "truncated_count": sum(
                    length > args.max_length for length in raw_lengths
                ),
                "peak_allocated_mb": peak_mb,
                "minimum_score": min(values),
                "maximum_score": max(values),
            }
        )

    reranker_only_mb = 0.0
    both_models_mb = 0.0
    embedding_load_ms = 0.0
    free_mb = 0.0
    total_mb = 0.0
    if args.device == "cuda":
        torch.cuda.empty_cache()
        reranker_only_mb = torch.cuda.memory_allocated() / 1024 / 1024
        embedding_started = time.perf_counter()
        embedding_model = AutoModel.from_pretrained(
            args.embedding_model_path,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(args.device).eval()
        torch.cuda.synchronize()
        embedding_load_ms = (time.perf_counter() - embedding_started) * 1000.0
        both_models_mb = torch.cuda.memory_allocated() / 1024 / 1024
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / 1024 / 1024
        total_mb = total_bytes / 1024 / 1024
        del embedding_model

    payload = {
        "schema_version": 1,
        "status": "success",
        "reranker_model": args.model_path.name,
        "embedding_model": args.embedding_model_path.name,
        "max_length": args.max_length,
        "load_ms": load_ms,
        "true_token_id": true_token_id,
        "false_token_id": false_token_id,
        "scores": [
            {"label": label, "score": value, "tokens": token_count}
            for label, value, token_count in zip(
                ("strong_match", "clear_mismatch", "ambiguous"),
                scores,
                tokens,
                strict=True,
            )
        ],
        "initial_batch_latency_ms": initial_latency,
        "initial_peak_allocated_mb": initial_peak,
        "batch_benchmarks": benchmark_rows,
        "reranker_only_allocated_mb": reranker_only_mb,
        "embedding_second_model_load_ms": embedding_load_ms,
        "both_models_allocated_mb": both_models_mb,
        "gpu_free_mb_after_both_loaded": free_mb,
        "gpu_total_mb": total_mb,
        "external_api_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(args.output)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()

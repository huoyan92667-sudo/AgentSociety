"""Persistent JSON-lines worker for local Qwen3 Cross-Encoder inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements '
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def _write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
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
    false_token_id = int(tokenizer.convert_tokens_to_ids("no"))
    true_token_id = int(tokenizer.convert_tokens_to_ids("yes"))
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body_limit = args.max_sequence_length - len(prefix_tokens) - len(suffix_tokens)
    if body_limit <= 0:
        raise ValueError("max sequence length is too small for the fixed prompt")
    _write(
        {
            "status": "ready",
            "device": args.device,
            "model": args.model_path.name,
            "true_token_id": true_token_id,
            "false_token_id": false_token_id,
        }
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = str(request["request_id"])
            instruction = request["instruction"]
            query = request["query"]
            documents = request["documents"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be nonempty text")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be nonempty text")
            if not isinstance(documents, list) or not documents or any(
                not isinstance(document, str) or not document.strip()
                for document in documents
            ):
                raise ValueError("documents must be a nonempty list of strings")
            bodies = [
                f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
                for document in documents
            ]
            raw = tokenizer(bodies, padding=False, truncation=False)["input_ids"]
            raw_lengths = [
                len(tokens) + len(prefix_tokens) + len(suffix_tokens)
                for tokens in raw
            ]
            encoded = tokenizer(
                bodies,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=body_limit,
            )
            for index, token_ids in enumerate(encoded["input_ids"]):
                encoded["input_ids"][index] = (
                    prefix_tokens + token_ids + suffix_tokens
                )
            batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            per_pair_tokens = [
                int(value) for value in batch["attention_mask"].sum(dim=1).tolist()
            ]
            batch = {key: value.to(args.device) for key, value in batch.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
                pair_logits = torch.stack(
                    [logits[:, false_token_id], logits[:, true_token_id]], dim=1
                )
                scores = torch.softmax(pair_logits.float(), dim=1)[:, 1]
            if args.device == "cuda":
                torch.cuda.synchronize()
            _write(
                {
                    "status": "success",
                    "request_id": request_id,
                    "scores": [float(value) for value in scores.cpu()],
                    "per_pair_input_tokens": per_pair_tokens,
                    "truncated_pair_count": sum(
                        length > args.max_sequence_length for length in raw_lengths
                    ),
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
        except Exception as exc:
            _write(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )


if __name__ == "__main__":
    main()

"""在独立Python进程中常驻Qwen2.5，并只返回A到E中的一个标签。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=768)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    label_ids = []
    for label in "ABCDE":
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f"label {label} is not one tokenizer token")
        label_ids.append(int(token_ids[0]))
    _write(
        {
            "status": "ready",
            "model": args.model_path.name,
            "device": args.device,
            "batch_size": args.batch_size,
            "label_token_ids": label_ids,
        }
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            system_prompt = str(request["system_prompt"])
            user_prompts = request["user_prompts"]
            if not isinstance(user_prompts, list) or not user_prompts:
                raise ValueError("user_prompts must be a nonempty list")
            if len(user_prompts) > args.batch_size:
                raise ValueError("classification batch exceeds configured limit")
            chat_texts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(user_prompt)},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for user_prompt in user_prompts
            ]
            raw_lengths = [
                len(tokenizer.encode(text, add_special_tokens=False))
                for text in chat_texts
            ]
            encoded = tokenizer(
                chat_texts,
                padding=True,
                truncation=True,
                max_length=args.max_sequence_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_token_count = int(encoded["attention_mask"].sum().item())
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**encoded, logits_to_keep=1).logits[:, -1, :]
                allowed = logits[:, label_ids]
                choices = allowed.argmax(dim=1).tolist()
            if args.device == "cuda":
                torch.cuda.synchronize()
            labels = ["ABCDE"[int(index)] for index in choices]
            _write(
                {
                    "status": "success",
                    "request_id": request.get("request_id"),
                    "labels": labels,
                    "input_token_count": input_token_count,
                    "truncated_candidate_count": sum(
                        length > args.max_sequence_length for length in raw_lengths
                    ),
                    "model_latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
        except Exception as exc:
            _write(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                }
            )


if __name__ == "__main__":
    main()

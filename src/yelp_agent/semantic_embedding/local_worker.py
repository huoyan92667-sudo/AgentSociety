"""Persistent JSON-lines worker for local Qwen3 embedding inference."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path


def _write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _pool(last_hidden_state: object, attention_mask: object) -> object:
    import torch

    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[rows, sequence_lengths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        local_files_only=True,
    ).to(args.device).eval()
    dimension = int(model.config.hidden_size)
    _write(
        {
            "status": "ready",
            "dimension": dimension,
            "device": args.device,
            "model": args.model_path.name,
        }
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = str(request["request_id"])
            texts = request["texts"]
            if not isinstance(texts, list) or not texts or any(
                not isinstance(text, str) or not text.strip() for text in texts
            ):
                raise ValueError("texts must be a nonempty list of strings")
            # Count one text at a time. Some fast-tokenizer versions interpret a
            # large list as a pre-tokenized sequence instead of a text batch.
            raw = []
            for index, text in enumerate(texts):
                try:
                    raw.append(tokenizer.encode(text, add_special_tokens=True))
                except Exception as exc:
                    raise ValueError(
                        f"tokenization_failed_at_index={index}"
                    ) from exc
            truncated_text_count = sum(
                len(tokens) > args.max_sequence_length for tokens in raw
            )
            per_text_tokens = [
                min(len(tokens), args.max_sequence_length) for tokens in raw
            ]
            if request.get("operation") == "count_tokens":
                _write(
                    {
                        "status": "success",
                        "request_id": request_id,
                        "per_text_input_tokens": per_text_tokens,
                        "truncated_text_count": truncated_text_count,
                    }
                )
                continue
            batch = tokenizer(
                texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=args.max_sequence_length,
                return_tensors="pt",
            )
            per_text_tokens = [
                int(value) for value in batch["attention_mask"].sum(dim=1).tolist()
            ]
            batch = {key: value.to(args.device) for key, value in batch.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
                vectors = functional.normalize(
                    _pool(hidden, batch["attention_mask"]),
                    p=2,
                    dim=1,
                )
            if args.device == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            array = vectors.float().cpu().numpy().astype("<f4", copy=False)
            _write(
                {
                    "status": "success",
                    "request_id": request_id,
                    "rows": int(array.shape[0]),
                    "dimension": int(array.shape[1]),
                    "vectors_b64": base64.b64encode(array.tobytes()).decode("ascii"),
                    "per_text_input_tokens": per_text_tokens,
                    "truncated_text_count": truncated_text_count,
                    "latency_ms": latency_ms,
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

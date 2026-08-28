"""用Qwen2.5一次五选一判断评论证据方向，不生成解释或置信度。"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal, Protocol

from pydantic import Field

from yelp_agent.models import StrictModel

from .schema import PreferenceSearchDescription


type FiveClassEvidenceLabel = Literal[
    "irrelevant",
    "positive",
    "negative",
    "mixed",
    "ambiguous",
]

_LETTER_TO_LABEL: dict[str, FiveClassEvidenceLabel] = {
    "A": "irrelevant",
    "B": "positive",
    "C": "negative",
    "D": "mixed",
    "E": "ambiguous",
}

_SYSTEM_PROMPT = """
You classify one Yelp review passage against one restaurant requirement.
Use only the passage. Judge the reviewed restaurant, not another restaurant mentioned in a comparison.
The label is relative to the requirement, not to the review stars or general sentiment.

Choose exactly one label:
A = irrelevant: the passage gives no evidence about the requirement. Merely naming a cuisine or giving general praise/criticism is not enough.
B = positive: the passage clearly says the reviewed restaurant satisfies the requirement.
C = negative: the passage clearly says the reviewed restaurant fails the requirement.
D = mixed: the passage contains clear evidence on both sides, possibly for different dishes, times, or conditions.
E = ambiguous: the passage discusses the requirement but does not provide enough information to decide a direction.

Important examples:
- Requirement: quiet conversation. Passage: "The music was so loud we could barely hear each other." => C
- Requirement: quiet conversation. Passage: "We could talk comfortably without raising our voices." => B
- Requirement: quiet conversation. Passage: "Lunch was peaceful, but dinner became extremely noisy." => D
- Requirement: authentic regional food. Passage: "I do not know authentic cooking, but the food was good." => E
- Requirement: authentic regional food. Passage: "The server was friendly and the room was clean." => A

Answer with only A, B, C, D, or E.
""".strip()


class QwenEvidenceInput(StrictModel):
    """第三版只需要评论编号和命中的上下文片段。"""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    segment_text: str = Field(min_length=1)


class QwenEvidenceJudgment(StrictModel):
    """正式输出只保留唯一分类，不保存看似精确的模型概率。"""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    label: FiveClassEvidenceLabel


class QwenEvidenceMetrics(StrictModel):
    candidate_count: int = Field(ge=0)
    batch_count: int = Field(ge=0)
    input_token_count: int = Field(ge=0)
    truncated_candidate_count: int = Field(ge=0)
    model_latency_ms: float = Field(ge=0)
    wall_latency_ms: float = Field(ge=0)


class QwenEvidenceBatch(StrictModel):
    judgments: list[QwenEvidenceJudgment]
    metrics: QwenEvidenceMetrics


@dataclass(frozen=True, slots=True)
class LabelBatch:
    """本地模型只返回标签和真实运行统计。"""

    labels: tuple[str, ...]
    input_token_count: int
    truncated_candidate_count: int
    latency_ms: float


class LabelClassifier(Protocol):
    batch_size: int

    def classify(self, prompts: Sequence[str]) -> LabelBatch: ...


class QwenFiveClassEvidenceJudge:
    """隐藏提示词拼装和批处理，调用方只提交一条要求及评论片段。"""

    def __init__(self, classifier: LabelClassifier) -> None:
        self._classifier = classifier

    def judge(
        self,
        requirement: PreferenceSearchDescription,
        candidates: Sequence[QwenEvidenceInput],
    ) -> QwenEvidenceBatch:
        started = perf_counter()
        prompts = [_user_prompt(requirement, item.segment_text) for item in candidates]
        judgments: list[QwenEvidenceJudgment] = []
        batch_count = 0
        input_tokens = 0
        truncated = 0
        model_latency_ms = 0.0
        for offset in range(0, len(prompts), self._classifier.batch_size):
            prompt_batch = prompts[offset : offset + self._classifier.batch_size]
            candidate_batch = candidates[offset : offset + self._classifier.batch_size]
            if not prompt_batch:
                continue
            result = self._classifier.classify(prompt_batch)
            if len(result.labels) != len(candidate_batch):
                raise RuntimeError("Qwen label result does not match input batch")
            batch_count += 1
            input_tokens += result.input_token_count
            truncated += result.truncated_candidate_count
            model_latency_ms += result.latency_ms
            for candidate, letter in zip(candidate_batch, result.labels, strict=True):
                label = _LETTER_TO_LABEL.get(letter)
                if label is None:
                    raise RuntimeError(f"Qwen returned an unsupported label: {letter}")
                judgments.append(
                    QwenEvidenceJudgment(
                        review_id=candidate.review_id,
                        business_id=candidate.business_id,
                        label=label,
                    )
                )
        return QwenEvidenceBatch(
            judgments=judgments,
            metrics=QwenEvidenceMetrics(
                candidate_count=len(candidates),
                batch_count=batch_count,
                input_token_count=input_tokens,
                truncated_candidate_count=truncated,
                model_latency_ms=model_latency_ms,
                wall_latency_ms=(perf_counter() - started) * 1000,
            ),
        )


class LocalQwenFiveClassClassifier:
    """常驻子进程适配器；模型加载一次后可反复批量判断。"""

    def __init__(
        self,
        *,
        process: subprocess.Popen[str],
        batch_size: int,
        model_name: str,
    ) -> None:
        self._process = process
        self.batch_size = batch_size
        self.model_name = model_name
        self._lock = Lock()
        self._request_index = 0

    @classmethod
    def start(
        cls,
        *,
        model_path: str | Path,
        python_executable: str | Path,
        device: Literal["cuda", "cpu"] = "cuda",
        batch_size: int = 4,
        max_sequence_length: int = 768,
    ) -> LocalQwenFiveClassClassifier:
        model = Path(model_path)
        python = Path(python_executable)
        if not model.is_dir() or not (model / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"Qwen model is incomplete: {model}")
        if not python.is_file():
            raise FileNotFoundError(f"Qwen Python executable does not exist: {python}")
        worker = Path(__file__).with_name("qwen_five_class_worker.py")
        process = subprocess.Popen(
            [
                str(python),
                str(worker),
                "--model-path",
                str(model),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--max-sequence-length",
                str(max_sequence_length),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert process.stdout is not None
        ready_line = process.stdout.readline()
        if not ready_line:
            detail = process.stderr.read()[-2000:] if process.stderr is not None else ""
            process.kill()
            raise RuntimeError(f"Qwen five-class worker failed to start: {detail}")
        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            process.kill()
            raise RuntimeError(f"Qwen five-class worker failed: {ready}")
        return cls(
            process=process,
            batch_size=int(ready["batch_size"]),
            model_name=str(ready["model"]),
        )

    def classify(self, prompts: Sequence[str]) -> LabelBatch:
        if not prompts or len(prompts) > self.batch_size:
            raise ValueError("Qwen classification batch size is invalid")
        payload = {"system_prompt": _SYSTEM_PROMPT, "user_prompts": list(prompts)}
        response, wall_ms = self._send(payload)
        labels = tuple(str(item) for item in response["labels"])
        return LabelBatch(
            labels=labels,
            input_token_count=int(response["input_token_count"]),
            truncated_candidate_count=int(response["truncated_candidate_count"]),
            latency_ms=wall_ms,
        )

    def _send(self, payload: dict[str, object]) -> tuple[dict[str, object], float]:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("Qwen five-class worker has stopped")
            self._request_index += 1
            request = {"request_id": self._request_index, **payload}
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            started = perf_counter()
            self._process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            wall_ms = (perf_counter() - started) * 1000
        if not line:
            detail = self._process.stderr.read()[-2000:] if self._process.stderr is not None else ""
            raise RuntimeError(f"Qwen five-class worker stopped: {detail}")
        response = json.loads(line)
        if response.get("status") != "success":
            raise RuntimeError(
                f"Qwen five-class inference failed: {response.get('error_message')}"
            )
        return response, wall_ms

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)


def _user_prompt(
    requirement: PreferenceSearchDescription,
    segment_text: str,
) -> str:
    return (
        f"Restaurant requirement: {requirement.requirement_text}\n"
        "Positive evidence meanings: "
        + "; ".join(requirement.positive_descriptions)
        + "\nNegative evidence meanings: "
        + "; ".join(requirement.negative_descriptions)
        + f"\nReview passage:\n{segment_text.strip()}"
    )

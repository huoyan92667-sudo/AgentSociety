"""Enforce development-data rules and deterministic user-level folds."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import random
from typing import Sequence

from pydantic import Field

from yelp_agent.config import EvaluationDataUsageConfig
from yelp_agent.experiments import (
    TaskFileError,
    read_recommendation_tasks,
    write_json_artifact,
)
from yelp_agent.models import RecommendationTask, StrictModel


class DataUsageViolation(RuntimeError):
    """Raised when test data is used for a development-time decision."""


class FoldSize(StrictModel):
    user_count: int = Field(ge=0)
    task_count: int = Field(ge=0)


class UserFoldSummary(StrictModel):
    format_version: int
    purpose: str
    development_split: str
    strategy: str
    seed: int
    fold_count: int = Field(ge=2)
    task_count: int = Field(ge=0)
    user_count: int = Field(ge=0)
    multi_task_user_count: int = Field(ge=0)
    folds: dict[str, FoldSize]
    assignment_sha256: str
    assignment_hash_algorithm: str
    source_tasks_sha256: str
    user_ids_included: bool
    task_ids_included: bool
    legacy_test_name: str
    legacy_test_status: str
    legacy_test_usage: str
    strict_blind_holdout: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_user_fold(
    user_id: str,
    policy: EvaluationDataUsageConfig,
) -> int:
    """Return a stable, one-based fold number for one user."""

    if not user_id or user_id != user_id.strip():
        raise ValueError("user_id must be nonempty with no surrounding whitespace")
    cross_validation = policy.cross_validation
    payload = f"{cross_validation.seed}:{user_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, "big") % cross_validation.folds + 1


def assign_development_task_folds(
    tasks: Sequence[RecommendationTask],
    policy: EvaluationDataUsageConfig,
) -> dict[str, int]:
    """Assign validation tasks by user and reject any test task."""

    if not tasks:
        raise ValueError("development tasks cannot be empty")
    assignments: dict[str, int] = {}
    user_folds: dict[str, int] = {}
    expected_prefix = f"{policy.development_split}:"
    for task in tasks:
        if not task.task_id.startswith(expected_prefix):
            raise DataUsageViolation(
                f"Model selection accepts {policy.development_split} tasks only; "
                f"received {task.task_id!r}"
            )
        if task.task_id in assignments:
            raise DataUsageViolation(
                f"Duplicate development task_id: {task.task_id!r}"
            )
        fold = assign_user_fold(task.user_id, policy)
        previous = user_folds.setdefault(task.user_id, fold)
        if previous != fold:
            raise DataUsageViolation(
                f"User {task.user_id!r} was assigned to multiple folds"
            )
        assignments[task.task_id] = fold
    return assignments


def deterministic_bootstrap_users(
    user_ids: Sequence[str],
    *,
    replicate_index: int,
    policy: EvaluationDataUsageConfig,
) -> list[str]:
    """Sample users with replacement for one deterministic bootstrap replicate."""

    if replicate_index < 0:
        raise ValueError("replicate_index cannot be negative")
    if not user_ids:
        raise ValueError("user_ids cannot be empty")
    if any(not user_id or user_id != user_id.strip() for user_id in user_ids):
        raise ValueError("user_ids contain an invalid value")
    if len(set(user_ids)) != len(user_ids):
        raise ValueError("user_ids must be unique before bootstrap sampling")
    ordered_users = sorted(user_ids)
    seed_payload = (
        f"{policy.bootstrap.seed}:bootstrap:{replicate_index}".encode("utf-8")
    )
    seed = int.from_bytes(hashlib.sha256(seed_payload).digest(), "big")
    generator = random.Random(seed)
    return generator.choices(ordered_users, k=len(ordered_users))


def build_validation_user_fold_summary(
    validation_tasks_path: str | Path,
    policy: EvaluationDataUsageConfig,
) -> UserFoldSummary:
    """Summarize folds without publishing any user or task identifier."""

    path = Path(validation_tasks_path)
    try:
        tasks = read_recommendation_tasks(path)
    except TaskFileError as exc:
        raise DataUsageViolation(str(exc)) from exc
    task_assignments = assign_development_task_folds(tasks, policy)
    tasks_per_user = Counter(task.user_id for task in tasks)
    users_per_fold: Counter[int] = Counter()
    tasks_per_fold: Counter[int] = Counter(task_assignments.values())
    user_assignments: dict[str, int] = {}
    for task in tasks:
        user_assignments.setdefault(task.user_id, task_assignments[task.task_id])
    users_per_fold.update(user_assignments.values())

    assignment_digest = hashlib.sha256()
    for user_id, fold in sorted(user_assignments.items()):
        assignment_digest.update(user_id.encode("utf-8"))
        assignment_digest.update(b"\0")
        assignment_digest.update(str(fold).encode("ascii"))
        assignment_digest.update(b"\n")

    return UserFoldSummary(
        format_version=1,
        purpose="deterministic user-level development cross-validation",
        development_split=policy.development_split,
        strategy=policy.cross_validation.strategy,
        seed=policy.cross_validation.seed,
        fold_count=policy.cross_validation.folds,
        task_count=len(tasks),
        user_count=len(user_assignments),
        multi_task_user_count=sum(
            count > 1 for count in tasks_per_user.values()
        ),
        folds={
            str(fold): FoldSize(
                user_count=users_per_fold[fold],
                task_count=tasks_per_fold[fold],
            )
            for fold in range(1, policy.cross_validation.folds + 1)
        },
        assignment_sha256=assignment_digest.hexdigest(),
        assignment_hash_algorithm=(
            "SHA256 over user_id, NUL, one-based fold, LF in sorted user order"
        ),
        source_tasks_sha256=_sha256_file(path),
        user_ids_included=False,
        task_ids_included=False,
        legacy_test_name=policy.legacy_test.name,
        legacy_test_status=policy.legacy_test.status,
        legacy_test_usage=policy.legacy_test.usage,
        strict_blind_holdout=policy.strict_blind_holdout,
    )


def write_user_fold_summary(
    summary: UserFoldSummary,
    output_path: str | Path,
) -> None:
    """Atomically write a byte-stable, identifier-free fold summary."""

    write_json_artifact(output_path, summary)

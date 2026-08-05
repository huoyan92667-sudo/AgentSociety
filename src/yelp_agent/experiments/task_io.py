"""Read frozen recommendation tasks through one validated boundary."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from yelp_agent.models import RecommendationTask


class TaskFileError(RuntimeError):
    """Raised when a task artifact is empty, malformed, or ambiguous."""


class TaskSplitError(TaskFileError):
    """Raised when a task belongs to a forbidden experiment split."""


def read_recommendation_tasks(
    path: str | Path,
    *,
    required_split: str | None = None,
    limit: int | None = None,
) -> list[RecommendationTask]:
    """Load unique tasks in file order, with optional split and count guards."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Task JSONL does not exist: {source}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if required_split is not None and (
        not required_split or required_split != required_split.strip()
    ):
        raise ValueError("required_split must be nonempty without outer whitespace")

    tasks: list[RecommendationTask] = []
    seen_task_ids: set[str] = set()
    expected_prefix = (
        None if required_split is None else f"{required_split}:"
    )
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    task = RecommendationTask.model_validate_json(line)
                except ValidationError as exc:
                    raise TaskFileError(
                        f"Invalid task at {source}:{line_number}: {exc}"
                    ) from exc
                if expected_prefix is not None and not task.task_id.startswith(
                    expected_prefix
                ):
                    raise TaskSplitError(
                        f"Expected {required_split} tasks only; "
                        f"received {task.task_id!r} at line {line_number}"
                    )
                if task.task_id in seen_task_ids:
                    raise TaskFileError(
                        f"Duplicate task_id at {source}:{line_number}: "
                        f"{task.task_id!r}"
                    )
                seen_task_ids.add(task.task_id)
                tasks.append(task)
                if limit is not None and len(tasks) == limit:
                    break
    except TaskFileError:
        raise
    except OSError as exc:
        raise TaskFileError(f"Could not read task JSONL: {source}") from exc

    if not tasks:
        raise TaskFileError(f"Task JSONL contains no tasks: {source}")
    return tasks

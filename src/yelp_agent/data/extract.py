"""Safe, selective extraction of the three Yelp JSONL inputs."""

from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from yelp_agent.data.archive import (
    REQUIRED_ARCHIVE_MEMBERS,
    inspect_archive,
)
from yelp_agent.models import StrictModel


class ExtractionError(RuntimeError):
    """Raised when a required member cannot be extracted safely."""


class ExtractedMemberResult(StrictModel):
    name: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    status: Literal["extracted", "skipped"]
    size_bytes: int = Field(ge=0)


class ExtractionResult(StrictModel):
    archive_path: str
    output_dir: str
    files: dict[str, ExtractedMemberResult]


def extract_required_members(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> ExtractionResult:
    archive_file = Path(archive_path)
    output = Path(output_dir)
    inspection = inspect_archive(archive_file)
    output.mkdir(parents=True, exist_ok=True)
    output_resolved = output.resolve()
    results: dict[str, ExtractedMemberResult] = {}
    plans: list[tuple[str, str, int, Path, Literal["extracted", "skipped"]]] = []
    bytes_needed = 0
    for kind, member_name in REQUIRED_ARCHIVE_MEMBERS.items():
        expected_size = inspection.required_members[kind].size_bytes
        destination = output / member_name
        if destination.parent.resolve() != output_resolved:
            raise ExtractionError(
                f"refusing extraction outside output directory: {member_name}"
            )
        if destination.exists() and not force:
            if not destination.is_file() or destination.stat().st_size != expected_size:
                raise ExtractionError(
                    f"existing output does not match archive member: {destination}"
                )
            action: Literal["extracted", "skipped"] = "skipped"
        else:
            action = "extracted"
            bytes_needed += expected_size
        plans.append((kind, member_name, expected_size, destination, action))

    free_bytes = shutil.disk_usage(output).free
    if free_bytes < bytes_needed:
        raise ExtractionError(
            f"insufficient free disk space: need {bytes_needed} bytes, "
            f"have {free_bytes} bytes"
        )

    with tarfile.open(archive_file, "r:*") as archive:
        for kind, member_name, expected_size, destination, action in plans:
            if action == "skipped":
                results[kind] = ExtractedMemberResult(
                    name=member_name,
                    output_path=str(destination.resolve()),
                    status="skipped",
                    size_bytes=expected_size,
                )
                continue

            temporary = destination.with_name(destination.name + ".partial")
            try:
                temporary.unlink(missing_ok=True)
                member = archive.getmember(member_name)
                source = archive.extractfile(member)
                if source is None:
                    raise ExtractionError(
                        f"cannot read required archive member: {member_name}"
                    )
                with source, temporary.open("xb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                actual_size = temporary.stat().st_size
                if actual_size != expected_size:
                    raise ExtractionError(
                        f"extracted size mismatch for {member_name}: "
                        f"expected {expected_size}, got {actual_size}"
                    )
                os.replace(temporary, destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

            results[kind] = ExtractedMemberResult(
                name=member_name,
                output_path=str(destination.resolve()),
                status="extracted",
                size_bytes=expected_size,
            )

    return ExtractionResult(
        archive_path=str(archive_file.resolve()),
        output_dir=str(output_resolved),
        files=results,
    )


def write_extraction_report(
    result: ExtractionResult,
    output_path: str | Path,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

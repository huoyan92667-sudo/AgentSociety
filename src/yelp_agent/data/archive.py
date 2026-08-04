"""Read-only validation of the Yelp dataset TAR archive."""

from __future__ import annotations

import tarfile
from pathlib import Path

from pydantic import Field

from yelp_agent.models import StrictModel


REQUIRED_ARCHIVE_MEMBERS = {
    "business": "yelp_academic_dataset_business.json",
    "review": "yelp_academic_dataset_review.json",
    "user": "yelp_academic_dataset_user.json",
}


class ArchiveValidationError(ValueError):
    """Raised when an archive cannot satisfy the Yelp input contract."""


class ArchiveMemberInfo(StrictModel):
    name: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    is_regular_file: bool


class ArchiveInspection(StrictModel):
    archive_path: str
    archive_size_bytes: int = Field(ge=0)
    member_count: int = Field(ge=0)
    uncompressed_size_bytes: int = Field(ge=0)
    required_files_present: bool
    required_members: dict[str, ArchiveMemberInfo]
    extra_members: list[ArchiveMemberInfo]


def _member_info(member: tarfile.TarInfo) -> ArchiveMemberInfo:
    return ArchiveMemberInfo(
        name=member.name,
        size_bytes=member.size,
        is_regular_file=member.isfile(),
    )


def inspect_archive(archive_path: str | Path) -> ArchiveInspection:
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(f"archive does not exist: {path}")
    try:
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
    except (tarfile.TarError, OSError) as exc:
        raise ArchiveValidationError(
            f"cannot open TAR archive: {path}"
        ) from exc

    seen_names: set[str] = set()
    for member in members:
        if member.name in seen_names:
            raise ArchiveValidationError(
                f"duplicate archive member: {member.name}"
            )
        seen_names.add(member.name)
    by_name = {member.name: member for member in members}
    for required_name in REQUIRED_ARCHIVE_MEMBERS.values():
        if required_name not in by_name:
            raise ArchiveValidationError(
                f"missing required member: {required_name}"
            )
        if not by_name[required_name].isfile():
            raise ArchiveValidationError(
                f"required member is not a regular file: {required_name}"
            )
    required = {
        kind: _member_info(by_name[name])
        for kind, name in REQUIRED_ARCHIVE_MEMBERS.items()
    }
    required_names = set(REQUIRED_ARCHIVE_MEMBERS.values())
    extras = [
        _member_info(member)
        for member in members
        if member.name not in required_names
    ]
    return ArchiveInspection(
        archive_path=str(path.resolve()),
        archive_size_bytes=path.stat().st_size,
        member_count=len(members),
        uncompressed_size_bytes=sum(member.size for member in members if member.isfile()),
        required_files_present=True,
        required_members=required,
        extra_members=extras,
    )


def write_inspection_report(
    inspection: ArchiveInspection,
    output_path: str | Path,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        inspection.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

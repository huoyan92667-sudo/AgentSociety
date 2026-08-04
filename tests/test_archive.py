import io
import json
import tarfile
from pathlib import Path

import pytest

from yelp_agent.data.archive import (
    ArchiveValidationError,
    inspect_archive,
    write_inspection_report,
)


REQUIRED_FILES = {
    "yelp_academic_dataset_business.json": b'{"business_id":"b1"}\n',
    "yelp_academic_dataset_review.json": b'{"review_id":"r1"}\n',
    "yelp_academic_dataset_user.json": b'{"user_id":"u1"}\n',
}


def write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name=name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_complete_archive_returns_structured_read_only_inspection(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    files = {**REQUIRED_FILES, "yelp_academic_dataset_tip.json": b"{}\n"}
    write_tar(archive_path, files)

    inspection = inspect_archive(archive_path)

    assert inspection.required_files_present is True
    assert inspection.member_count == 4
    assert inspection.uncompressed_size_bytes == sum(map(len, files.values()))
    assert inspection.required_members["business"].name.endswith("business.json")
    assert inspection.required_members["review"].size_bytes == len(
        files["yelp_academic_dataset_review.json"]
    )
    assert inspection.extra_members[0].name == "yelp_academic_dataset_tip.json"
    assert list(tmp_path.iterdir()) == [archive_path]


def test_archive_missing_required_member_fails_with_clear_error(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    files = dict(REQUIRED_FILES)
    del files["yelp_academic_dataset_review.json"]
    write_tar(archive_path, files)

    with pytest.raises(
        ArchiveValidationError,
        match="missing required member: yelp_academic_dataset_review.json",
    ):
        inspect_archive(archive_path)


def test_required_member_must_be_a_regular_file(tmp_path: Path) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, content in REQUIRED_FILES.items():
            member = tarfile.TarInfo(name=name)
            if name == "yelp_academic_dataset_user.json":
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

    with pytest.raises(
        ArchiveValidationError,
        match="required member is not a regular file: yelp_academic_dataset_user.json",
    ):
        inspect_archive(archive_path)


def test_duplicate_member_names_are_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, content in REQUIRED_FILES.items():
            member = tarfile.TarInfo(name=name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        duplicate = tarfile.TarInfo(
            name="yelp_academic_dataset_business.json"
        )
        duplicate.size = 3
        archive.addfile(duplicate, io.BytesIO(b"{}\n"))

    with pytest.raises(
        ArchiveValidationError,
        match="duplicate archive member: yelp_academic_dataset_business.json",
    ):
        inspect_archive(archive_path)


def test_corrupt_archive_fails_with_clear_error(tmp_path: Path) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    archive_path.write_bytes(b"this is not a tar archive")

    with pytest.raises(ArchiveValidationError, match="cannot open TAR archive"):
        inspect_archive(archive_path)


def test_inspection_report_can_be_saved_as_json(tmp_path: Path) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    report_path = tmp_path / "runs" / "data_inspection.json"
    write_tar(archive_path, REQUIRED_FILES)

    inspection = inspect_archive(archive_path)
    write_inspection_report(inspection, report_path)

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["required_files_present"] is True
    assert saved["required_members"]["user"]["size_bytes"] == len(
        REQUIRED_FILES["yelp_academic_dataset_user.json"]
    )

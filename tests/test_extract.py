import io
import json
import tarfile
from collections import namedtuple
from pathlib import Path

import pytest

from yelp_agent.data.extract import (
    ExtractionError,
    extract_required_members,
    write_extraction_report,
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


def test_extracts_only_required_members_and_preserves_archive(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    output_dir = tmp_path / "raw"
    files = {
        **REQUIRED_FILES,
        "yelp_academic_dataset_tip.json": b'{"text":"ignore"}\n',
    }
    write_tar(archive_path, files)
    archive_before = archive_path.read_bytes()

    result = extract_required_members(archive_path, output_dir)

    assert {
        path.name for path in output_dir.iterdir()
    } == set(REQUIRED_FILES)
    for name, expected in REQUIRED_FILES.items():
        assert (output_dir / name).read_bytes() == expected
    assert {item.status for item in result.files.values()} == {"extracted"}
    assert archive_path.read_bytes() == archive_before


def test_matching_existing_files_are_skipped_without_rewrite(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    output_dir = tmp_path / "raw"
    write_tar(archive_path, REQUIRED_FILES)
    extract_required_members(archive_path, output_dir)
    mtimes_before = {
        path.name: path.stat().st_mtime_ns for path in output_dir.iterdir()
    }

    result = extract_required_members(archive_path, output_dir)

    assert {item.status for item in result.files.values()} == {"skipped"}
    assert {
        path.name: path.stat().st_mtime_ns for path in output_dir.iterdir()
    } == mtimes_before


def test_mismatched_existing_file_is_never_silently_overwritten(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    output_dir = tmp_path / "raw"
    write_tar(archive_path, REQUIRED_FILES)
    output_dir.mkdir()
    business_path = output_dir / "yelp_academic_dataset_business.json"
    business_path.write_bytes(b"incomplete")

    with pytest.raises(
        ExtractionError,
        match="existing output does not match archive member",
    ):
        extract_required_members(archive_path, output_dir)

    assert business_path.read_bytes() == b"incomplete"


def test_all_existing_outputs_are_preflighted_before_any_new_file_is_written(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    output_dir = tmp_path / "raw"
    write_tar(archive_path, REQUIRED_FILES)
    output_dir.mkdir()
    review_path = output_dir / "yelp_academic_dataset_review.json"
    review_path.write_bytes(b"incomplete")

    with pytest.raises(ExtractionError):
        extract_required_members(archive_path, output_dir)

    assert not (
        output_dir / "yelp_academic_dataset_business.json"
    ).exists()
    assert review_path.read_bytes() == b"incomplete"


def test_force_replaces_mismatched_file_via_partial_path(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    output_dir = tmp_path / "raw"
    write_tar(archive_path, REQUIRED_FILES)
    output_dir.mkdir()
    business_path = output_dir / "yelp_academic_dataset_business.json"
    partial_path = output_dir / "yelp_academic_dataset_business.json.partial"
    business_path.write_bytes(b"incomplete")
    partial_path.write_bytes(b"stale partial data")

    result = extract_required_members(
        archive_path,
        output_dir,
        force=True,
    )

    assert business_path.read_bytes() == REQUIRED_FILES[business_path.name]
    assert not partial_path.exists()
    assert result.files["business"].status == "extracted"


def test_extraction_fails_before_writing_when_disk_space_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    output_dir = tmp_path / "raw"
    write_tar(archive_path, REQUIRED_FILES)
    disk_usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        "yelp_agent.data.extract.shutil.disk_usage",
        lambda _: disk_usage(total=100, used=99, free=1),
    )

    with pytest.raises(ExtractionError, match="insufficient free disk space"):
        extract_required_members(archive_path, output_dir)

    assert list(output_dir.iterdir()) == []


def test_extraction_report_can_be_saved_as_json(tmp_path: Path) -> None:
    archive_path = tmp_path / "yelp_dataset.tar"
    output_dir = tmp_path / "raw"
    report_path = tmp_path / "runs" / "extraction_report.json"
    write_tar(archive_path, REQUIRED_FILES)

    result = extract_required_members(archive_path, output_dir)
    write_extraction_report(result, report_path)

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["files"]["business"]["status"] == "extracted"
    assert saved["files"]["review"]["size_bytes"] == len(
        REQUIRED_FILES["yelp_academic_dataset_review.json"]
    )

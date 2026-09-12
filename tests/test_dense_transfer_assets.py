# Copyright (c) 2026 QLab contributors.

from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path  # noqa: TC003

import pytest

from scripts import prepare_dense_transfer_assets as assets


def _write_coco(root: Path, *, train: int = 2, val: int = 1, missing: bool = False) -> None:
    (root / "annotations").mkdir(parents=True)
    categories = [{"id": index} for index in range(80)]
    for split, count in (("train", train), ("val", val)):
        image_dir = root / f"{split}2017"
        image_dir.mkdir()
        names = [f"{index:012d}.jpg" for index in range(count)]
        for name in names:
            (image_dir / name).write_bytes(b"fixture")
        references = names[:-1] + (["missing.jpg"] if missing and names else [])
        payload = {
            "images": [{"id": index, "file_name": name} for index, name in enumerate(references)],
            "categories": categories,
            "annotations": [
                {"id": index, "image_id": index, "segmentation": [[0, 0, 1, 1, 0, 1]]}
                for index in range(max(1, len(references)))
            ],
        }
        (root / "annotations" / f"instances_{split}2017.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def _write_ade(root: Path, *, training: int = 2, validation: int = 1) -> None:
    for split, count in (("training", training), ("validation", validation)):
        image_dir = root / "images" / split
        label_dir = root / "annotations" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(count):
            stem = f"scene_{index:05d}"
            (image_dir / f"{stem}.jpg").write_bytes(b"jpg")
            (label_dir / f"{stem}.png").write_bytes(b"png")


def test_coco_fixture_override_and_missing_reference_are_explicit(tmp_path: Path) -> None:
    _write_coco(tmp_path, missing=True)

    result = assets.verify_coco(tmp_path, {"train": 2, "val": 1})

    assert result["ready"] is False
    assert result["state"] == "incomplete"
    assert result["splits"]["train"]["missing_referenced_files"] == ["missing.jpg"]
    assert result["splits"]["val"]["categories_count"] == 80


def test_ade_fixture_requires_jpg_png_stem_pairs(tmp_path: Path) -> None:
    _write_ade(tmp_path)
    (tmp_path / "annotations" / "validation" / "scene_00000.png").unlink()

    result = assets.verify_ade(tmp_path, {"training": 2, "validation": 1})

    assert result["ready"] is False
    assert result["splits"]["validation"]["missing_labels"] == ["scene_00000"]
    assert result["ignore_index"] == 0
    assert result["ignore_mapping"] == "handled_by_dataloader"


def test_checkpoints_report_observed_hashes_without_upstream_claim(tmp_path: Path) -> None:
    for name in assets.CHECKPOINT_NAMES:
        (tmp_path / name).write_bytes(name.encode())

    result = assets.verify_checkpoints(tmp_path)

    assert result["ready"] is True
    for row in result["files"]:
        assert row["sha256"] == row["sha256_observed"]
        assert row["sha256_verified_upstream"] is False


def test_downloader_resumes_part_without_replacing_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "train2017.zip"
    partial = tmp_path / "train2017.zip.part"
    partial.write_bytes(b"abc")

    class Response:
        status = 206
        url = assets.COCO_TRAIN_URL
        headers: dict[str, str]

        def __init__(self) -> None:
            self.headers = {"Content-Length": "3"}
            self.read_once = False

        def read(self, _size: int) -> bytes:
            if self.read_once:
                return b""
            self.read_once = True
            return b"def"

        def close(self) -> None:
            return None

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    result = assets.download_archive(assets.ARCHIVE_SPECS[0], destination, reserve_bytes=0)

    assert result["status"] == "resumed"
    assert destination.read_bytes() == b"abcdef"
    assert not partial.exists()

    # A subsequent call observes the final file and cannot replace it.
    destination.write_bytes(b"user")
    observed = assets.download_archive(assets.ARCHIVE_SPECS[0], destination, reserve_bytes=0)
    assert observed["status"] == "existing"
    assert destination.read_bytes() == b"user"


def test_manifest_hash_excludes_dynamic_observations() -> None:
    first = {
        "manifest": {"assets": ["a"], "free_space": {"free_bytes": 1}},
        "generated_at_utc": "2026-01-01T00:00:00Z",
    }
    second = {
        "manifest": {"assets": ["a"], "free_space": {"free_bytes": 2}},
        "generated_at_utc": "2027-01-01T00:00:00Z",
    }

    assert assets.manifest_sha256(first) == assets.manifest_sha256(second)


def test_unsafe_zip_is_rejected_before_writing_outside_target(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../outside.txt", b"no")
    target = tmp_path / "extract"

    with pytest.raises(assets.UnsafeArchiveError, match="traversal"):
        assets.safe_extract_zip(archive, target, reserve_bytes=0)
    assert not (tmp_path / "outside.txt").exists()
    assert not target.exists()


def test_zip_symlink_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(info, "../../outside")

    with pytest.raises(assets.UnsafeArchiveError, match="symbolic-link"):
        assets.safe_extract_zip(archive, tmp_path / "extract", reserve_bytes=0)


def test_extraction_does_not_replace_existing_material_file(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data/file.txt", b"new")
    target = tmp_path / "target"
    (target / "data").mkdir(parents=True)
    (target / "data" / "file.txt").write_bytes(b"user")

    with pytest.raises(assets.ArchiveCollisionError):
        assets.safe_extract_zip(archive, target, reserve_bytes=0)
    assert (target / "data" / "file.txt").read_bytes() == b"user"


def test_safe_extraction_of_normal_archive(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("folder/file.txt", b"payload")

    result = assets.safe_extract_zip(archive, tmp_path / "target", reserve_bytes=0)

    assert result["state"] == "complete"
    assert (tmp_path / "target" / "folder" / "file.txt").read_bytes() == b"payload"


def test_verify_report_is_atomic_and_incomplete_without_false_readiness(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "assets.json"
    result = assets.prepare(
        coco_root=tmp_path / "coco",
        ade_root=tmp_path / "ade",
        checkpoint_root=tmp_path / "checkpoints",
        source_root=tmp_path / "sources",
        report=report,
    )

    assert result["state"] == "incomplete"
    assert result["ready"] is False
    assert json.loads(report.read_text(encoding="utf-8"))["manifest_sha256"] == result[
        "manifest_sha256"
    ]
    assert not (tmp_path / "coco").exists()
    assert not (tmp_path / "ade").exists()

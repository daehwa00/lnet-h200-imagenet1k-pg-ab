from __future__ import annotations

import json
from pathlib import Path

import pytest

from h200.validate_imagenet1k import build_manifest, persist_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dataset(root: Path) -> Path:
    for split in ("train", "val"):
        for class_name in ("n00000001", "n00000002"):
            (root / split / class_name).mkdir(parents=True)
    (root / "train/n00000001/a.JPEG").write_bytes(b"a")
    (root / "train/n00000002/b.JPEG").write_bytes(b"bb")
    (root / "val/n00000001/c.JPEG").write_bytes(b"ccc")
    (root / "val/n00000002/d.JPEG").write_bytes(b"dddd")
    return root


def _manifest(root: Path) -> dict[str, object]:
    return build_manifest(
        root,
        expected_train=2,
        expected_val=2,
        expected_classes=2,
    )


def test_dataset_identity_is_stable_and_sensitive_to_same_size_content(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "imagenet")
    first = _manifest(dataset)
    second = _manifest(dataset)
    assert first["identity_sha256"] == second["identity_sha256"]
    assert first["splits"] == second["splits"]

    (dataset / "val/n00000002/d.JPEG").write_bytes(b"zzzz")
    changed = _manifest(dataset)
    assert changed["identity_sha256"] != first["identity_sha256"]


def test_persisted_dataset_identity_is_immutable(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "imagenet")
    output = tmp_path / "run/dataset_manifest.json"
    first = persist_manifest(_manifest(dataset), output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["identity_sha256"] == first["identity_sha256"]

    (dataset / "train/n00000001/a.JPEG").write_bytes(b"different")
    with pytest.raises(RuntimeError, match="persisted dataset identity differs"):
        persist_manifest(_manifest(dataset), output)


def test_dataset_validator_rejects_class_or_count_drift(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "imagenet")
    (dataset / "val/n00000002").rename(dataset / "val/n00000003")
    with pytest.raises(RuntimeError, match="class directory sets differ"):
        _manifest(dataset)

    dataset = _dataset(tmp_path / "other-imagenet")
    (dataset / "train/n00000002/b.JPEG").unlink()
    with pytest.raises(RuntimeError, match="image count mismatch"):
        _manifest(dataset)


def test_entrypoint_freezes_commit_campaign_credentials_and_batch_shape() -> None:
    script = (PROJECT_ROOT / "h200/run.sh").read_text(encoding="utf-8")
    assert 'readonly PYTHON_VERSION="3.13.11"' in script
    assert "uv_bootstrap_version" in script
    assert ".incomplete-$$" in script
    assert "--require-hashes" in script
    assert "H200_EXPECTED_COMMIT" in script
    assert "git rev-parse --verify HEAD" in script
    assert 'export WANDB_API_KEY="${DUMMY_WANDB_API_KEY}"' in script
    assert "${WANDB_API_KEY:-" not in script
    assert 'export WANDB_BASE_URL="${CAMPAIGN_WANDB_BASE_URL}"' in script
    assert 'export WANDB_ENTITY="${CAMPAIGN_ENTITY}"' in script
    assert 'export WANDB_PROJECT="${CAMPAIGN_PROJECT}"' in script
    assert 'export WANDB_GROUP="${CAMPAIGN_GROUP}"' in script
    assert "export LNET_PERSISTENT_WORKERS=0" in script
    assert script.count("--batch-size 256") == 2
    assert script.count('--workers "${WORKERS}"') == 2
    assert "WORKERS > 8" in script
    assert script.index('CPU_COUNT="$(nproc)"') < script.index("export OMP_NUM_THREADS=1")
    assert "${OUTPUT_NAMESPACE}-${MANIFEST_SHA256:0:16}" in script


def test_h200_requirements_and_lock_are_exact_and_non_yanked() -> None:
    requirements = (PROJECT_ROOT / "h200/requirements.txt").read_text(encoding="utf-8")
    package_lines = [
        line
        for raw in requirements.splitlines()
        if (line := raw.strip()) and not line.startswith(("#", "--"))
    ]
    assert package_lines
    assert all("==" in line for line in package_lines)
    assert "numpy==2.4.6" in package_lines
    assert "numpy==2.4.0" not in requirements

    lock = (PROJECT_ROOT / "h200/requirements.lock").read_text(encoding="utf-8")
    assert "numpy==2.4.6" in lock
    assert "numpy==2.4.0" not in lock
    assert "--hash=sha256:" in lock
    assert "torch==2.9.1+cu128" in lock
    assert "wandb==0.22.3" in lock

    uv_lock = (PROJECT_ROOT / "h200/uv-bootstrap.requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "uv==0.9.26" in uv_lock
    assert "--hash=sha256:" in uv_lock

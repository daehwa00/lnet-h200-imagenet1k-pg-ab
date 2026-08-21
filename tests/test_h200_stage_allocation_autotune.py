from __future__ import annotations

# ruff: noqa: SLF001
import json
from pathlib import Path

import pytest
import torch

from scripts import benchmark_h200_stage_allocation_autotune as autotune


def _candidate(
    batch_size: int,
    lanes: int,
    *,
    elapsed_seconds: float,
    peak_reserved_bytes: int = 10 * 2**30,
    free_memory_bytes: int = 100 * 2**30,
    allocated_drift_bytes: int = 0,
    finite: bool = True,
) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "lanes": lanes,
        "status": "complete",
        "warmup_steps": 5,
        "measured_steps": 20,
        "loader_batches": 130000 // batch_size,
        "elapsed_seconds": elapsed_seconds,
        "host_input_wait_seconds": 0.1,
        "model_images_per_second": batch_size * 13 * 20 / elapsed_seconds,
        "peak_allocated_bytes": peak_reserved_bytes - 2**30,
        "peak_reserved_bytes": peak_reserved_bytes,
        "free_memory_bytes": free_memory_bytes,
        "total_memory_bytes": 140 * 2**30,
        "allocated_drift_bytes": allocated_drift_bytes,
        "finite": finite,
        "variants": list(autotune.stage.VARIANTS),
    }


def test_candidate_matrix_and_partitions_are_exact() -> None:
    assert autotune.CANDIDATE_MATRIX == (
        (128, 1),
        (128, 2),
        (128, 4),
        (128, 6),
        (128, 7),
        (128, 8),
        (256, 1),
        (256, 2),
        (256, 4),
        (512, 1),
        (512, 2),
    )
    assert len(set(autotune.CANDIDATE_MATRIX)) == len(autotune.CANDIDATE_MATRIX)
    assert autotune._partitions(13, 4) == [4, 4, 4, 1]
    assert autotune._partitions(13, 6) == [6, 6, 1]
    assert autotune._partitions(13, 7) == [7, 6]
    assert autotune._partitions(13, 8) == [8, 5]


def test_selection_rejects_unsafe_rows_and_selects_large_gain() -> None:
    total = 140 * 2**30
    candidates = [
        _candidate(128, 1, elapsed_seconds=20.0),
        _candidate(128, 8, elapsed_seconds=5.0, peak_reserved_bytes=int(0.85 * total)),
        _candidate(256, 4, elapsed_seconds=10.0),
        _candidate(512, 2, elapsed_seconds=8.0, free_memory_bytes=23 * 2**30),
        _candidate(128, 2, elapsed_seconds=18.0, finite=False),
        _candidate(256, 2, elapsed_seconds=12.0, allocated_drift_bytes=800 * 2**20),
    ]

    recommendation = autotune._select(candidates, total)

    assert recommendation["status"] == "selected"
    assert recommendation["selected"]["batch_size"] == 256
    assert recommendation["selected"]["lanes"] == 4
    assert recommendation["selected"]["partition"] == [4, 4, 4, 1]
    assert recommendation["gain_over_baseline"] > 0.1


def test_minimum_gain_gate_keeps_baseline() -> None:
    total = 140 * 2**30
    baseline = _candidate(128, 1, elapsed_seconds=20.0)
    marginal = _candidate(128, 2, elapsed_seconds=18.5)

    recommendation = autotune._select([baseline, marginal], total)

    assert recommendation["selected"]["batch_size"] == 128
    assert recommendation["selected"]["lanes"] == 1
    assert recommendation["minimum_gain_gate_applied"] is True


def test_epoch_estimate_uses_drop_last_loader_length() -> None:
    candidate = _candidate(512, 1, elapsed_seconds=20.0)

    annotated = autotune._annotate_candidate(candidate, 140 * 2**30)

    assert annotated["loader_batches"] == 130000 // 512
    assert annotated["estimated_full_cohort_seconds_per_epoch"] == 130000 // 512


def test_oom_is_recoverable_but_other_runtime_errors_propagate() -> None:
    recovered: list[bool] = []

    result = autotune._attempt(
        512,
        2,
        lambda: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("oom")),
        lambda: recovered.append(True),
    )

    assert result["status"] == "oom"
    assert recovered == [True]
    with pytest.raises(RuntimeError, match="code bug"):
        autotune._attempt(
            128,
            1,
            lambda: (_ for _ in ()).throw(RuntimeError("code bug")),
            lambda: recovered.append(False),
        )
    assert recovered == [True]


def test_final_and_batch_cache_require_complete_exact_identity(tmp_path: Path) -> None:
    final = tmp_path / "result.json"
    final.write_text(
        json.dumps(
            {
                "schema": autotune.SCHEMA,
                "status": "complete",
                "identity_sha256": "a" * 64,
                "candidates": [],
                "recommendation": {},
            }
        )
    )
    assert autotune._cache_hit(final, "a" * 64) is not None
    assert autotune._cache_hit(final, "b" * 64) is None
    final.write_text("{")
    assert autotune._cache_hit(final, "a" * 64) is None

    batch = tmp_path / "batch-128.json"
    rows = [
        {"batch_size": 128, "lanes": lanes, "status": "complete"} for lanes in (1, 2, 4, 6, 7, 8)
    ]
    batch.write_text(
        json.dumps(
            {
                "schema": autotune.SCHEMA,
                "status": "batch_complete",
                "identity_sha256": "a" * 64,
                "batch_size": 128,
                "candidates": rows,
            }
        )
    )
    assert (
        autotune._batch_cache_hit(
            batch,
            identity_sha256="a" * 64,
            batch_size=128,
        )
        == rows
    )
    assert (
        autotune._batch_cache_hit(
            batch,
            identity_sha256="a" * 64,
            batch_size=256,
        )
        is None
    )


def test_owner_stop_marker_is_commit_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "stop.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "lnet.h200.owner_stop.v1",
                "target_commit": "a" * 40,
                "generation": 4,
                "reason": "owner stop",
            }
        )
    )
    monkeypatch.setenv("H200_EXPECTED_COMMIT", "a" * 40)
    monkeypatch.setenv("H200_CONTROL_FAST_STOP_MARKER", str(marker))

    with pytest.raises(autotune.OwnerStopRequestedError, match="owner stop"):
        autotune._raise_if_owner_stopped()

    monkeypatch.setenv("H200_EXPECTED_COMMIT", "b" * 40)
    with pytest.raises(RuntimeError, match="identity changed"):
        autotune._raise_if_owner_stopped()


def test_h200_shell_routes_autotune_after_dataset_staging() -> None:
    script = Path("h200/run_imagenet100_stage_allocation.sh").read_text()

    assert 'H200_AUTOTUNE_ONLY="${H200_AUTOTUNE_ONLY:-0}"' in script
    assert "H200_AUTOTUNE_ONLY must be 0 or 1" in script
    assert script.count("scripts/benchmark_h200_stage_allocation_autotune.py") == 1
    assert "--kill-after=5m 3h" in script
    assert "env WANDB_MODE=disabled" in script
    assert 'if [[ "${H200_AUTOTUNE_ONLY}" == "0" ]]; then' in script
    assert script.index('if [[ "${H200_AUTOTUNE_ONLY}" == "0" ]]; then') < script.index(
        "cloudflare/stage-allocation-relay/canary.py"
    )
    assert script.index("H200_TRAIN_DATA_ROOT=") < script.index(
        "scripts/benchmark_h200_stage_allocation_autotune.py"
    )
    assert script.index("scripts/benchmark_h200_stage_allocation_autotune.py") < script.index(
        "scripts/run_h200_imagenet100_stage_allocation.py"
    )

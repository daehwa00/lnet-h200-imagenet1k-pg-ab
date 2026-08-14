# pyright: reportExplicitAny=false

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _seed_metrics(directory: Path, pattern: str) -> dict[str, Any]:
    rows = [json.loads(path.read_text()) for path in sorted(directory.glob(pattern))]
    values = [float(row["test"]["balanced_accuracy"]) for row in rows]
    if len(values) != 5:
        message = f"expected five seeds in {directory}, found {len(values)}"
        raise ValueError(message)
    return {
        "seeds": [int(row["config"]["seed"]) for row in rows],
        "balanced_accuracy": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "parameter_count": sorted({int(row["parameter_count"]) for row in rows}),
    }


def _audit_summary(directory: Path, scope: str) -> dict[str, Any]:
    payloads = [
        json.loads(path.read_text())
        for path in sorted(directory.glob("pole-audit-seed*.json"))
    ]
    result: dict[str, Any] = {}
    for target in ("16", "92"):
        rows = [payload[scope][target] for payload in payloads]
        values = [
            float(row["statistic"])
            for row in rows
            if math.isfinite(float(row["statistic"]))
        ]
        result[target] = {
            "spearman": values,
            "finite_seed_mean": statistics.mean(values) if values else None,
            "significant_positive_seeds": sum(
                float(row["pvalue"]) < 0.05 and float(row["statistic"]) > 0
                for row in rows
            ),
        }
    return result


def _fixed_ndft(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "modes": payload["modes"],
        "objects": sum(
            int(value["count"]) for value in payload["class_spearman"].values()
        ),
        "class_spearman": payload["class_spearman"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--original-phase0-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root
    original = _seed_metrics(args.original_phase0_root, "alphabet-seed*.json")
    variants = {
        "periodic_zoh_actual": _seed_metrics(root / "periodic-actual", "alphabet-seed*.json"),
        "periodic_zoh_unit": _seed_metrics(root / "periodic-unit", "alphabet-seed*.json"),
        "periodic_zoh_grid": _seed_metrics(root / "periodic-grid", "alphabet-seed*.json"),
        "periodic_impulse": _seed_metrics(
            root / "periodic-impulse", "alphabet-seed*.json"
        ),
        "periodic_impulse_u8": _seed_metrics(
            root / "periodic-impulse-u8", "alphabet-seed*.json"
        ),
        "periodic_impulse_point": _seed_metrics(
            root / "periodic-impulse-point", "alphabet-seed*.json"
        ),
        "periodic_impulse_point_u8": _seed_metrics(
            root / "periodic-impulse-point-u8", "alphabet-seed*.json"
        ),
        "full_impulse": _seed_metrics(root / "full-impulse", "alphabet-seed*.json"),
        "full_impulse_point_u8": _seed_metrics(
            root / "full-impulse-point-u8", "alphabet-seed*.json"
        ),
        "dls_multiband_weighted": _seed_metrics(
            root / "periodic-dls-multiband-m64-weighted",
            "dls-seed*.json",
        ),
        "dls_multiband_u32_weighted": _seed_metrics(
            root / "periodic-dls-multiband-m64-u32-weighted",
            "dls-seed*.json",
        ),
        "dls_fixedw_m512_weighted": _seed_metrics(
            root / "periodic-dls-fixedw-m512-weighted",
            "dls-seed*.json",
        ),
    }
    periodic_actual = variants["periodic_zoh_actual"]["mean"]
    full_impulse = variants["full_impulse"]["mean"]
    pole_rows = json.loads((root / "pole-distribution.json").read_text())["rows"]
    payload = {
        "schema": "lnet.astronomy.zoh_impulse_diagnosis.v1",
        "execution_host": "local_gpu",
        "gpu": "2x NVIDIA GeForce RTX 4090",
        "original_full_zoh": original,
        "variants": variants,
        "effects": {
            "periodic_actual_minus_unit_pp": 100.0
            * (periodic_actual - variants["periodic_zoh_unit"]["mean"]),
            "periodic_actual_minus_grid_pp": 100.0
            * (periodic_actual - variants["periodic_zoh_grid"]["mean"]),
            "periodic_best_impulse_minus_zoh_pp": 100.0
            * (variants["periodic_impulse_point_u8"]["mean"] - periodic_actual),
            "full_impulse_minus_zoh_pp": 100.0 * (full_impulse - original["mean"]),
        },
        "learned_pole_diagnosis": {
            "pole_count": len(pole_rows),
            "abs_omega_below_0_1": sum(
                abs(float(row["omega_rad_per_day"])) < 0.1 for row in pole_rows
            ),
            "memory_over_200_days": sum(
                float(row["memory_days"]) > 200.0 for row in pole_rows
            ),
            "memory_over_1000_days": sum(
                float(row["memory_days"]) > 1000.0 for row in pole_rows
            ),
        },
        "alphabet_g4": {
            "periodic_zoh": _audit_summary(root / "periodic-actual", "class_spearman"),
            "periodic_impulse": _audit_summary(
                root / "periodic-impulse", "writer_class_spearman"
            ),
            "periodic_impulse_point_u8": _audit_summary(
                root / "periodic-impulse-point-u8",
                "writer_class_spearman",
            ),
            "decision": "FAIL",
        },
        "fixed_grid_ndft_vs_lomb_scargle": {
            "m512_all": _fixed_ndft(
                root
                / "fixed-ndft-all-m512"
                / "spectrum-audit-initialized-m512-all.json"
            ),
            "m4096_all": _fixed_ndft(
                root
                / "fixed-ndft-all-m4096"
                / "spectrum-audit-initialized-m4096-all.json"
            ),
        },
        "decisions": {
            "time_information_unused": "REJECTED",
            "native_irregular_beats_interpolation": "REJECTED",
            "zoh_mismatch_hurts_classification": "SUPPORTED",
            "impulse_alphabet_restores_physical_poles": "REJECTED",
            "ndft_equals_lomb_scargle": "REJECTED",
            "new_paper_thesis_ready": "NO_GO",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

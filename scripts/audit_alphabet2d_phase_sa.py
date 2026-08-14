#!/usr/bin/env python3
"""Combine the primary and mirrored S-A results into the valid gate decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--mirror-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    full = json.loads(args.full_summary.read_text())
    mirror = json.loads(args.mirror_summary.read_text())
    audited = {}
    for epsilon, row in full["by_epsilon"].items():
        means = row["mean_balanced_accuracy"]
        audited[epsilon] = {
            **row,
            "mean_product_four_minus_axial_pp": 100.0
            * (means["product_four"] - means["axial2d"]),
        }
    gates = {
        "A1_raw_covariance_within_2pp": full["gates"][
            "A1_raw_covariance_within_2pp"
        ],
        "A2_four_scan_product_beats_axial_5pp_epsilon_ge_0_4": all(
            row["mean_product_four_minus_axial_pp"] >= 5.0
            for epsilon, row in audited.items()
            if float(epsilon) >= 0.4
        ),
        "A3_four_scan_product_reaches_90pct_oracle_at_0_8": (
            (
                audited["0.8"]["mean_balanced_accuracy"]["product_four"] - 0.5
            )
            / (audited["0.8"]["oracle_mean_balanced_accuracy"] - 0.5)
            >= 0.9
        ),
        "A4_single_within_2pp_of_four_both_slopes": (
            full["gates"]["A4_single_within_2pp_of_four"]
            and mirror["A4_single_within_2pp_of_four_on_mirror"]
        ),
    }
    payload = {
        "schema": "lnet.alphabet2d.phase_sa.audited.v1",
        "primary": str(args.full_summary),
        "mirror": str(args.mirror_summary),
        "by_epsilon": audited,
        "mirror_control": mirror,
        "gates": gates,
        "decision": {
            "joint_2d_spectral_hypothesis": (
                "pass"
                if gates[
                    "A2_four_scan_product_beats_axial_5pp_epsilon_ge_0_4"
                ]
                else "kill"
            ),
            "default_scan": (
                "single"
                if gates["A4_single_within_2pp_of_four_both_slopes"]
                else "four"
            ),
            "P1_single_scan_hypothesis": (
                "pass"
                if gates["A4_single_within_2pp_of_four_both_slopes"]
                else "rejected"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

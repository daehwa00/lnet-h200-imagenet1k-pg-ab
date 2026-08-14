from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pole-distribution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    pole_rows = json.loads(args.pole_distribution.read_text())["rows"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        "ZOH actual",
        "h=1",
        "1-day grid",
        "Impulse",
        "Impulse+point",
        "Impulse+point+u8",
    ]
    keys = [
        "periodic_zoh_actual",
        "periodic_zoh_unit",
        "periodic_zoh_grid",
        "periodic_impulse",
        "periodic_impulse_point",
        "periodic_impulse_point_u8",
    ]
    variants = summary["variants"]
    means = [variants[key]["mean"] for key in keys]
    errors = [variants[key]["sample_std"] for key in keys]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    positions = np.arange(len(labels))
    axis.bar(positions, means, yerr=errors, capsize=4, color="#3b82f6")
    for position, key in zip(positions, keys, strict=True):
        axis.scatter(
            np.full(5, position),
            variants[key]["balanced_accuracy"],
            color="#172554",
            s=18,
            zorder=3,
        )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Periodic-only balanced accuracy")
    axis.set_ylim(0.75, 0.96)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "periodic-a2-impulse.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    omega = np.abs([float(row["omega_rad_per_day"]) for row in pole_rows])
    memory = [float(row["memory_days"]) for row in pole_rows]
    axes[0].hist(omega, bins=24, color="#0f766e")
    axes[0].axvline(0.1, color="#dc2626", linestyle="--", linewidth=1)
    axes[0].set_xlabel(r"Learned $|\omega|$ [rad/day]")
    axes[0].set_ylabel("Pole count")
    axes[1].hist(np.log10(memory), bins=24, color="#7c3aed")
    axes[1].axvline(np.log10(200.0), color="#dc2626", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"$\log_{10}$ memory time [day]")
    figure.tight_layout()
    figure.savefig(args.output_dir / "learned-pole-distribution.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

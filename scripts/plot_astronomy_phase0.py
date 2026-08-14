from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot as plt


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phase0 = cast(
        "dict[str, object]",
        json.loads((args.results_dir / "phase0-summary.json").read_text()),
    )
    mechanism = cast(
        "dict[str, object]",
        json.loads((args.results_dir / "mechanism-summary.json").read_text()),
    )
    metrics = cast("dict[str, dict[str, object]]", phase0["metrics"])
    controls = cast(
        "dict[str, dict[str, float]]",
        mechanism["readout_and_time_controls"],
    )
    early = cast("list[dict[str, object]]", mechanism["early_classification"])

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    model_names = ("alphabet", "gru")
    means = [
        float(
            cast("dict[str, dict[str, float]]", metrics[name]["test"])[
                "balanced_accuracy"
            ]["mean"]
        )
        for name in model_names
    ]
    errors = [
        float(
            cast("dict[str, dict[str, float]]", metrics[name]["test"])[
                "balanced_accuracy"
            ]["sample_std"]
        )
        for name in model_names
    ]
    axes[0, 0].bar(("ALPHABET", "Δt-GRU"), means, yerr=errors, capsize=5)
    axes[0, 0].axhline(0.85, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(0.75, 1.0)
    axes[0, 0].set_ylabel("Balanced accuracy")
    axes[0, 0].set_title("Five-seed Phase 0")

    control_labels = (
        "physical",
        "unit Δt",
        "token lag",
        "energy only",
    )
    control_keys = (
        "actual_intervals_physical_lag",
        "unit_intervals_physical_lag",
        "actual_intervals_token_lag",
        "actual_intervals_energy_only",
    )
    control_means = [controls[key]["mean"] for key in control_keys]
    control_errors = [controls[key]["sample_std"] for key in control_keys]
    axes[0, 1].bar(control_labels, control_means, yerr=control_errors, capsize=4)
    axes[0, 1].set_ylim(0.75, 1.0)
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].set_ylabel("Balanced accuracy")
    axes[0, 1].set_title("Time/readout controls")

    for model, label in (("alphabet", "ALPHABET"), ("gru", "Δt-GRU")):
        rows = sorted(
            (row for row in early if row["model"] == model),
            key=lambda row: float(cast("float", row["days"])),
        )
        axes[1, 0].errorbar(
            [float(cast("float", row["days"])) for row in rows],
            [float(cast("float", row["balanced_accuracy_mean"])) for row in rows],
            yerr=[
                float(cast("float", row["balanced_accuracy_sample_std"]))
                for row in rows
            ],
            marker="o",
            capsize=3,
            label=label,
        )
    axes[1, 0].set_xlabel("Observed days")
    axes[1, 0].set_ylabel("Balanced accuracy")
    axes[1, 0].set_title("Early classification")
    axes[1, 0].legend()

    attributed: list[float] = []
    lomb_scargle: list[float] = []
    for seed in (7, 11, 19, 23, 31):
        audit = cast(
            "dict[str, object]",
            json.loads(
                (args.results_dir / f"pole-audit-seed{seed}.json").read_text()
            ),
        )
        for row in cast("list[dict[str, object]]", audit["objects"]):
            if int(cast("int", row["target"])) == 2:
                attributed.append(float(cast("float", row["attributed_period_days"])))
                lomb_scargle.append(
                    float(cast("float", row["lomb_scargle_period_days"]))
                )
    axes[1, 1].scatter(lomb_scargle, attributed, alpha=0.35, s=18)
    lower = min(lomb_scargle)
    upper = max(lomb_scargle)
    axes[1, 1].plot((lower, upper), (lower, upper), color="black", linewidth=1)
    axes[1, 1].plot(
        (lower, upper),
        (0.5 * lower, 0.5 * upper),
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    axes[1, 1].plot(
        (lower, upper),
        (2.0 * lower, 2.0 * upper),
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Lomb-Scargle period (days)")
    axes[1, 1].set_ylabel("Attributed pole period (days)")
    axes[1, 1].set_title("RR Lyrae pole attribution (G4 FAIL)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()

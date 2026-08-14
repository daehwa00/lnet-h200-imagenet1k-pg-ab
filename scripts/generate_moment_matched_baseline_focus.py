"""Generate the full-width, baseline-focused moment-matched diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTENDED_ROOT = ROOT / ".omx/results/pac-moment-matched-extended-20260727"
OUTPUT_ROOT = EXTENDED_ROOT
ORIGINAL_ROOT = ROOT / ".omx/results/pac-moment-matched-l4-20260727"
LONG_ROOT = ROOT / ".omx/results/pac-moment-matched-null-long-t-20260727"

COLORS = {
    "control_dark": "#46515d",
    "control_mid": "#7d8995",
    "control_light": "#aab2ba",
    "alphabet": "#b83d31",
    "s4d": "#168c8c",
    "lru": "#e07a12",
    "gru": "#a93e58",
    "transformer": "#506f91",
}
GRID = "#d7dce0"
DATA_LINEWIDTH = 1.5


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(color=GRID, linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _mean_sd(payload: dict, epsilon_keys: tuple[str, ...], metric: str) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray([payload[key][metric]["mean"] for key in epsilon_keys], dtype=float)
    sds = np.asarray([payload[key][metric]["sample_sd"] for key in epsilon_keys], dtype=float)
    return means, sds


def _efficiency_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_length: dict[int, list[float]] = {length: [] for length in (128, 256, 512, 1024, 2048)}
    for path in sorted((ORIGINAL_ROOT / "completed").glob("eps0p400__seed*__full.json")):
        row = json.loads(path.read_text())
        numerator = float(row["validation_balanced_accuracy"]) - 0.5
        denominator = float(row["bayes_balanced_accuracy"]) - 0.5
        by_length[128].append(numerator / denominator)
    for path in sorted((LONG_ROOT / "completed").glob("long_t__T*__seed*.json")):
        row = json.loads(path.read_text())
        length = int(row["length"])
        numerator = float(row["validation_balanced_accuracy"]) - 0.5
        denominator = float(row["bayes_balanced_accuracy"]) - 0.5
        by_length[length].append(numerator / denominator)
    lengths = np.asarray(sorted(by_length), dtype=int)
    means = np.asarray([np.mean(by_length[length]) for length in lengths])
    ci95 = np.asarray(
        [
            2.776 * np.std(by_length[length], ddof=1) / math.sqrt(len(by_length[length]))
            for length in lengths
        ]
    )
    return lengths, means, ci95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("wide", "column"), default="wide")
    args = parser.parse_args()

    extended = json.loads((EXTENDED_ROOT / "summary.json").read_text())
    original = json.loads((ORIGINAL_ROOT / "summary.json").read_text())
    epsilon_keys = ("0.100", "0.200", "0.400", "0.800")
    eps = np.asarray([float(key) for key in epsilon_keys])
    baseline = extended["baseline"]
    primary = original["summary"]

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Bayes ceiling": _mean_sd(baseline, epsilon_keys, "bayes"),
        r"raw $\widehat\Gamma(0{:}4)$": _mean_sd(primary, epsilon_keys, "raw_moment_0_4"),
        r"raw $\widehat\Gamma(0{:}8)$": _mean_sd(baseline, epsilon_keys, "raw_moment_0_8"),
        "ALPHABET": _mean_sd(primary, epsilon_keys, "full"),
        "S4D": _mean_sd(baseline, epsilon_keys, "s4d"),
        "LRU": _mean_sd(baseline, epsilon_keys, "lru"),
        "GRU": _mean_sd(baseline, epsilon_keys, "gru"),
        "Transformer": _mean_sd(baseline, epsilon_keys, "transformer"),
    }
    styles = {
        "Bayes ceiling": (COLORS["control_dark"], (0, (6, 3)), "o", DATA_LINEWIDTH),
        r"raw $\widehat\Gamma(0{:}4)$": (
            COLORS["control_light"],
            (0, (2, 2)),
            "X",
            DATA_LINEWIDTH,
        ),
        r"raw $\widehat\Gamma(0{:}8)$": (
            COLORS["control_mid"],
            (0, (4, 2, 1, 2)),
            "o",
            DATA_LINEWIDTH,
        ),
        "ALPHABET": (COLORS["alphabet"], "-", "s", DATA_LINEWIDTH),
        "S4D": (COLORS["s4d"], "-", "^", DATA_LINEWIDTH),
        "LRU": (COLORS["lru"], "-", "D", DATA_LINEWIDTH),
        "GRU": (COLORS["gru"], "-", "v", DATA_LINEWIDTH),
        "Transformer": (COLORS["transformer"], "-", "P", DATA_LINEWIDTH),
    }
    parameters = {
        "ALPHABET": 5_698,
        "S4D": 21_634,
        "LRU": 25_666,
        "Transformer": 67_202,
        "GRU": 200_136,
    }
    parameter_labels = {
        "ALPHABET": "5.7k",
        "S4D": "21.6k",
        "LRU": "25.7k",
        "Transformer": "67.2k",
        "GRU": "200k",
    }
    accuracy_04 = {
        "ALPHABET": primary["0.400"]["full"]["mean"],
        "S4D": baseline["0.400"]["s4d"]["mean"],
        "LRU": baseline["0.400"]["lru"]["mean"],
        "Transformer": baseline["0.400"]["transformer"]["mean"],
        "GRU": baseline["0.400"]["gru"]["mean"],
    }

    if args.layout == "wide":
        typography = {
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
        subplot_shape = (1, 3)
        figure_size = (14.2, 3.65)
        output_stem = "moment_matched_baseline_focus"
    else:
        typography = {
            "font.size": 8.2,
            "axes.titlesize": 7.0,
            "axes.labelsize": 8.2,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
        }
        subplot_shape = (1, 2)
        figure_size = (3.45, 2.70)
        output_stem = "moment_matched_baseline_focus_column"
    plt.rcParams.update(
        {
            **typography,
            "figure.dpi": 200,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        *subplot_shape,
        figsize=figure_size,
        constrained_layout=args.layout == "wide",
    )

    ax = axes[0]
    for label, (means, sds) in curves.items():
        color, linestyle, marker, width = styles[label]
        ax.fill_between(eps, means - sds, means + sds, color=color, alpha=0.055, linewidth=0)
        ax.plot(
            eps,
            means,
            color=color,
            linestyle=linestyle,
            marker=marker,
            lw=width,
            ms=3.7,
            label=(
                f"{label} ({parameter_labels[label]})"
                if args.layout == "column" and label in parameter_labels
                else label
            ),
            zorder=4 if label == "ALPHABET" else 3,
        )
    ax.axhline(0.5, color=COLORS["control_light"], ls=":", lw=1)
    ax.text(0.79, 0.507, "chance = .50", color=COLORS["control_mid"], ha="right", va="bottom")
    ax.set(
        title=(
            r"a  Moment-matched separation across $\epsilon$"
            if args.layout == "wide"
            else None
        ),
        xlabel=r"spectral separation $\epsilon$",
        ylabel="validation balanced accuracy",
        ylim=(0.46, 1.025),
    )
    if args.layout == "column":
        ax.set_xticks((0.2, 0.4, 0.6, 0.8))
        ax.set_xlabel(r"spectral separation $\epsilon$", fontsize=8.2)
        ax.set_ylabel("validation balanced\naccuracy", fontsize=8.2)
    if args.layout == "wide":
        ax.legend(ncol=2, frameon=False, loc="upper left", columnspacing=0.8, handlelength=2.5)
    _style_axis(ax)

    if args.layout == "wide":
        ax = axes[1]
        label_offsets = {
            "ALPHABET": (7, 8),
            "S4D": (7, -17),
            "LRU": (7, 8),
            "Transformer": (-7, 6),
            "GRU": (-7, 20),
        }
        for model in ("ALPHABET", "S4D", "LRU", "Transformer", "GRU"):
            key = model.lower()
            ax.scatter(
                parameters[model],
                accuracy_04[model],
                s=82 if model == "ALPHABET" else 52,
                marker="s" if model == "ALPHABET" else styles[model][2],
                color=COLORS[key],
                zorder=3,
            )
            ax.annotate(
                f"{model}  {parameters[model] / 1000:.1f}k\nBA {accuracy_04[model]:.3f}",
                (parameters[model], accuracy_04[model]),
                xytext=label_offsets[model],
                textcoords="offset points",
                ha="right" if model in {"Transformer", "GRU"} else "left",
            )
        bayes_04 = baseline["0.400"]["bayes"]["mean"]
        ax.axhline(bayes_04, color=COLORS["control_dark"], ls="--", lw=1.4)
        ax.text(
            190_000,
            bayes_04 - 0.008,
            f"Bayes ceiling = {bayes_04:.3f}",
            color=COLORS["control_dark"],
            ha="right",
            va="top",
        )
        ax.set_xscale("log")
        ax.set(
            title=r"b  Compactness at $\epsilon=.4$",
            xlabel="trainable parameters (log scale)",
            ylabel="validation balanced accuracy",
            ylim=(0.47, 0.92),
        )
        _style_axis(ax)

    ax = axes[2] if args.layout == "wide" else axes[1]
    lengths, efficiency, ci95 = _efficiency_rows()
    ax.errorbar(
        lengths,
        efficiency,
        yerr=ci95,
        color=COLORS["alphabet"],
        marker="s",
        lw=DATA_LINEWIDTH,
        elinewidth=1.0,
        ms=4,
        capsize=3,
    )
    ax.axhline(1.0, color=COLORS["control_dark"], ls="--", lw=1.4)
    ax.text(
        132,
        1.002,
        "Bayes-normalized ceiling",
        color=COLORS["control_dark"],
        ha="left",
        va="bottom",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, [str(length) for length in lengths])
    ax.set(
        title=(
            r"c  Bayes-normalized efficiency at $\epsilon=.4$"
            if args.layout == "wide"
            else None
        ),
        xlabel="sequence length T",
        ylabel=r"$(\mathrm{BA}-.5)/(\mathrm{Bayes}-.5)$",
        ylim=(0.86, 1.015),
    )
    if args.layout == "column":
        ax.set_xlabel("sequence length T", fontsize=8.2)
        ax.set_ylabel(
            r"$(\mathrm{BA}-.5)/(\mathrm{Bayes}-.5)$",
            fontsize=8.2,
            labelpad=1,
        )
    _style_axis(ax)

    if args.layout == "column":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
            columnspacing=0.75,
            handlelength=1.7,
        )
        fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.46, wspace=0.34)

    output = OUTPUT_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{output_stem}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output / f"{output_stem}.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()

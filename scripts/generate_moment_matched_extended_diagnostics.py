"""Generate standalone plots for the extended moment-matched diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / ".omx/results/pac-moment-matched-extended-20260727"
ORIGINAL = ROOT / ".omx/results/pac-moment-matched-l4-20260727/summary.json"


def series(block: dict, keys: list[str], metric: str, field: str = "mean") -> np.ndarray:
    return np.asarray([block[key][metric][field] for key in keys], dtype=float)


def main() -> None:
    payload = json.loads((RESULT_ROOT / "summary.json").read_text())
    original = json.loads(ORIGINAL.read_text())
    eps_keys = ["0.100", "0.200", "0.400", "0.800"]
    eps = np.asarray([float(key) for key in eps_keys])
    baseline = payload["baseline"]
    poles = payload["poles"]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.3), constrained_layout=True)

    ax = axes[0, 0]
    styles = {
        "Bayes": ("#202938", "o", 2.5),
        r"raw $\widehat\Gamma(0{:}8)$": ("#8b98a5", "o", 2.0),
        "ALPHABET full (5.7k)": ("#6a49a5", "s", 2.7),
        "S4D (21.6k)": ("#168c8c", "^", 1.6),
        "LRU (25.7k)": ("#d47517", "D", 1.6),
        "GRU (200.1k)": ("#b14c54", "v", 1.6),
        "Transformer (67.2k)": ("#5f748d", "P", 1.6),
    }
    curves = {
        "Bayes": series(baseline, eps_keys, "bayes"),
        r"raw $\widehat\Gamma(0{:}8)$": series(baseline, eps_keys, "raw_moment_0_8"),
        "ALPHABET full (5.7k)": np.asarray(
            [original["summary"][key]["full"]["mean"] for key in eps_keys]
        ),
        "S4D (21.6k)": series(baseline, eps_keys, "s4d"),
        "LRU (25.7k)": series(baseline, eps_keys, "lru"),
        "GRU (200.1k)": series(baseline, eps_keys, "gru"),
        "Transformer (67.2k)": series(baseline, eps_keys, "transformer"),
    }
    for label, values in curves.items():
        color, marker, width = styles[label]
        ax.plot(eps, values, color=color, marker=marker, lw=width, ms=5, label=label)
    ax.axhline(0.5, color="#aab2bd", ls="--", lw=1)
    ax.set(xlabel=r"spectral separation $\epsilon$", ylabel="validation balanced accuracy")
    ax.set_title("a  Family comparison")
    ax.set_ylim(0.46, 1.025)
    ax.legend(ncol=2, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    learned = series(poles, eps_keys, "full")
    fixed = series(poles, eps_keys, "fixed_random_poles")
    learned_sd = np.asarray([poles[key]["full"]["sample_sd"] for key in eps_keys])
    fixed_sd = np.asarray([poles[key]["fixed_random_poles"]["sample_sd"] for key in eps_keys])
    ax.errorbar(eps, learned, learned_sd, color="#6a49a5", marker="s", lw=2.4, label="learned poles")
    ax.errorbar(eps, fixed, fixed_sd, color="#d47517", marker="o", lw=2.0, label="fixed random poles")
    ax.set(xlabel=r"spectral separation $\epsilon$", ylabel="validation balanced accuracy")
    ax.set_title("b  Learned versus fixed poles")
    ax.set_ylim(0.5, 1.025)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    initial = np.asarray(
        [poles[key]["full"]["initial_phase_extremum_alignment"]["mean"] for key in eps_keys]
    )
    final = np.asarray(
        [poles[key]["full"]["final_phase_extremum_alignment"]["mean"] for key in eps_keys]
    )
    fixed_alignment = np.asarray(
        [
            poles[key]["fixed_random_poles"]["final_phase_extremum_alignment"]["mean"]
            for key in eps_keys
        ]
    )
    ax.plot(eps, initial, color="#9ca6b2", marker="o", ls="--", label="learned-bank initialization")
    ax.plot(eps, final, color="#6a49a5", marker="s", lw=2.4, label="learned-bank final")
    ax.plot(eps, fixed_alignment, color="#d47517", marker="o", lw=1.8, label="fixed random bank")
    ax.set(
        xlabel=r"spectral separation $\epsilon$",
        ylabel=r"mean phase alignment $|\cos(5\omega)|$",
    )
    ax.set_title("c  Pole-phase alignment audit")
    ax.set_ylim(0.5, 0.75)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    lengths = np.asarray([64, 128, 256, 512], dtype=float)
    points = payload["length"]["points"]
    full_error = np.asarray([points[str(int(t))]["full_error"]["mean"] for t in lengths])
    bayes_error = np.asarray([points[str(int(t))]["bayes_error"]["mean"] for t in lengths])
    full_sd = np.asarray([points[str(int(t))]["full_error"]["sample_sd"] for t in lengths])
    bayes_sd = np.asarray([points[str(int(t))]["bayes_error"]["sample_sd"] for t in lengths])
    ax.errorbar(lengths, full_error, full_sd, color="#6a49a5", marker="s", lw=2.4, label="ALPHABET full")
    ax.errorbar(lengths, bayes_error, bayes_sd, color="#202938", marker="o", lw=2.4, label="Bayes")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lengths, [str(int(t)) for t in lengths])
    ax.set(xlabel="sequence length T", ylabel="validation error")
    ax.set_title(
        "d  Length sweep at "
        + r"$\epsilon=.1$"
        + f"  (slopes {payload['length']['full_log_error_log_T_slope']:.2f}/"
        + f"{payload['length']['bayes_log_error_log_T_slope']:.2f})"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")

    output = RESULT_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "moment_matched_extended_diagnostics.png", bbox_inches="tight")
    fig.savefig(output / "moment_matched_extended_diagnostics.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

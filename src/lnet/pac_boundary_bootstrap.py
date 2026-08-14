# ruff: noqa: EM101, T201, TRY003
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_OUTPUT: Final = Path(
    ".omx/results/pac-boundary-bootstrap-20260713/reports/BOUNDARY_BOOTSTRAP.json"
)
ALPHABET_ROOT: Final = Path(".omx/results/pac-pa2wp-boundary-20260713/shards")
PUBLIC_ROOT: Final = Path(".omx/results/pac-tf-p1p2-confirmatory-20260711/results")
CAPACITY_ROOTS: Final = (
    Path(
        ".omx/results/pac-endpoint-ood-capacity-matched-pro6000-20260713/"
        "results/pac_tf_synthetic_ood.csv"
    ),
    Path(
        ".omx/results/pac-endpoint-ood-capacity-s4d-pro6000-20260713/"
        "results/pac_tf_synthetic_ood.csv"
    ),
)
RETUNED_ROOT: Final = Path(
    ".omx/results/pac-endpoint-ood-retuned-pro6000-20260713/"
    "final/results/synthetic_ood.csv"
)

ObservationMap = dict[str, dict[tuple[str, ...], float]]


def build_report(*, replicates: int = 2_000, seed: int = 20_260_713) -> dict[str, object]:
    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    suites = _collect_suites()
    specifications = {
        "low_data_balanced_accuracy": {
            "expected_per_complete_model": 125,
            "hierarchy": "dataset -> seed; ratios retained as repeated fixed conditions",
            "higher_is_better": True,
        },
        "corruption_retained_accuracy": {
            "expected_per_complete_model": 175,
            "hierarchy": "dataset -> seed; seven corruptions retained as repeated fixed conditions",
            "higher_is_better": True,
        },
        "corruption_accuracy_drop": {
            "expected_per_complete_model": 175,
            "hierarchy": "dataset -> seed; seven corruptions retained as repeated fixed conditions",
            "higher_is_better": False,
        },
        "mit_bih_ood_balanced_accuracy": {
            "expected_per_complete_model": 5,
            "hierarchy": "seed cluster bootstrap",
            "higher_is_better": True,
        },
        "synthetic_absolute_nrmse_increase": {
            "expected_per_complete_model": 95,
            "hierarchy": (
                "seed cluster bootstrap; 19 OOD conditions retained as repeated "
                "fixed conditions"
            ),
            "higher_is_better": False,
        },
    }
    output: dict[str, object] = {}
    for suite, models in suites.items():
        spec = specifications[suite]
        model_rows: dict[str, object] = {}
        for index, (model, values) in enumerate(sorted(models.items())):
            estimate = _cluster_ci(
                values,
                suite=suite,
                replicates=replicates,
                seed=seed + index,
            )
            estimate["complete"] = len(values) == spec["expected_per_complete_model"]
            estimate["expected"] = spec["expected_per_complete_model"]
            model_rows[model] = estimate
        pairs: dict[str, object] = {}
        alphabet = models.get("alphabet", {})
        for index, (model, values) in enumerate(sorted(models.items())):
            if model == "alphabet":
                continue
            common = sorted(set(alphabet) & set(values))
            difference = {
                key: (
                    alphabet[key] - values[key]
                    if spec["higher_is_better"]
                    else values[key] - alphabet[key]
                )
                for key in common
            }
            paired = _cluster_ci(
                difference,
                suite=suite,
                replicates=replicates,
                seed=seed + 10_000 + index,
            )
            paired.update(
                {
                    "definition": "positive favors ALPHABET",
                    "paired_common_observations": len(common),
                    "complete_pair": len(common)
                    == spec["expected_per_complete_model"],
                }
            )
            pairs[model] = paired
        output[suite] = {
            "metric_direction": "higher" if spec["higher_is_better"] else "lower",
            "hierarchy": spec["hierarchy"],
            "models": model_rows,
            "paired_alphabet_advantage": pairs,
        }
    retuned = suites["synthetic_absolute_nrmse_increase"]
    return {
        "schema": "pac_boundary_hierarchical_bootstrap.v1",
        "replicates": replicates,
        "random_seed": seed,
        "partial_safe": True,
        "sources": {
            "alphabet_boundary": str(ALPHABET_ROOT),
            "public_low_data_corruption_mit": str(PUBLIC_ROOT),
            "synthetic_historical_hyperparameters": [
                str(path) for path in CAPACITY_ROOTS
            ],
            "synthetic_id_retuned": str(RETUNED_ROOT),
        },
        "pairing_policy": (
            "pair exact dataset/seed/condition keys before resampling; sample the same "
            "outer and inner clusters for both models"
        ),
        "retuned_synthetic_status": {
            "models_present": sorted(
                model for model in retuned if model.endswith("@id_retuned")
            ),
            "expected_models": 8,
            "complete_models": sum(
                len(values) == 95
                for model, values in retuned.items()
                if model.endswith("@id_retuned")
            ),
        },
        "suites": output,
    }


def write_report(
    output: Path = DEFAULT_OUTPUT,
    *,
    replicates: int = 2_000,
    seed: int = 20_260_713,
) -> dict[str, object]:
    payload = build_report(replicates=replicates, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _collect_suites() -> dict[str, ObservationMap]:
    suites: dict[str, ObservationMap] = {
        "low_data_balanced_accuracy": defaultdict(dict),
        "corruption_retained_accuracy": defaultdict(dict),
        "corruption_accuracy_drop": defaultdict(dict),
        "mit_bih_ood_balanced_accuracy": defaultdict(dict),
        "synthetic_absolute_nrmse_increase": defaultdict(dict),
    }
    low_paths = [PUBLIC_ROOT / "pac_tf_low_data.csv"]
    low_paths.extend(ALPHABET_ROOT.glob("*/results/pac_tf_low_data.csv"))
    for row in _done_rows(low_paths):
        model = _model_label(row)
        key = (
            str(row.get("dataset_or_task") or row.get("dataset")),
            str(row["seed"]),
            f"{float(row.get('requested_ratio') or row.get('data_ratio') or row['ratio']):g}",
        )
        suites["low_data_balanced_accuracy"][model][key] = float(row["balanced_accuracy"])

    corruption_paths = [PUBLIC_ROOT / "pac_tf_real_diagnostics.csv"]
    corruption_paths.extend(ALPHABET_ROOT.glob("*/results/pac_tf_real_diagnostics.csv"))
    for row in _done_rows(corruption_paths):
        model = _model_label(row)
        dataset = str(row.get("dataset_or_task") or row.get("dataset"))
        for condition in json.loads(row["real_corruption_ood_json"]):
            if condition["shift"] == "id":
                continue
            key = (dataset, str(row["seed"]), str(condition["shift"]))
            suites["corruption_retained_accuracy"][model][key] = float(
                condition["accuracy"]
            )
            suites["corruption_accuracy_drop"][model][key] = float(
                condition["absolute_accuracy_drop"]
            )

    mit_paths = [PUBLIC_ROOT / "pac_tf_real_domain_ood.csv"]
    mit_paths.extend(ALPHABET_ROOT.glob("*/results/pac_tf_real_domain_ood.csv"))
    for row in _done_rows(mit_paths):
        suites["mit_bih_ood_balanced_accuracy"][_model_label(row)][
            (str(row["seed"]),)
        ] = float(row["ood_common_balanced_accuracy"])

    alpha_synthetic = ALPHABET_ROOT.glob("*/results/pac_tf_synthetic_ood.csv")
    for row in _done_rows(alpha_synthetic):
        _add_synthetic_row(
            suites["synthetic_absolute_nrmse_increase"], row, "alphabet"
        )
    for row in _done_rows(CAPACITY_ROOTS):
        _add_synthetic_row(
            suites["synthetic_absolute_nrmse_increase"],
            row,
            f"{row['model']}@historical_hp",
        )
    for row in _done_rows((RETUNED_ROOT,)):
        _add_synthetic_row(
            suites["synthetic_absolute_nrmse_increase"],
            row,
            f"{row['model']}@id_retuned",
        )
    return {suite: dict(models) for suite, models in suites.items()}


def _add_synthetic_row(target: ObservationMap, row: dict[str, str], model: str) -> None:
    for condition in json.loads(row["ood_sweep_json"]):
        key = (str(row["seed"]), str(condition["family"]), str(condition["level"]))
        target[model][key] = float(condition["absolute_nrmse_increase"])


def _done_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = str(row.get("job_key") or row.get("key") or "")
                if row.get("status") == "done" and key:
                    latest[key] = dict(row)
    return list(latest.values())


def _model_label(row: dict[str, str]) -> str:
    return "alphabet" if row.get("model") == "pa2wp_pac" else str(row["model"])


def _cluster_ci(
    values: dict[tuple[str, ...], float],
    *,
    suite: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "observations": 0}
    generator = random.Random(seed)  # noqa: S311 - deterministic scientific bootstrap
    samples: list[float] = []
    if suite.startswith(("low_data", "corruption")):
        outer = sorted({key[0] for key in values})
        inner = {item: sorted({key[1] for key in values if key[0] == item}) for item in outer}
        for _ in range(replicates):
            selected: list[float] = []
            for sampled_outer in generator.choices(outer, k=len(outer)):
                seeds = inner[sampled_outer]
                for sampled_seed in generator.choices(seeds, k=len(seeds)):
                    selected.extend(
                        value
                        for key, value in values.items()
                        if key[0] == sampled_outer and key[1] == sampled_seed
                    )
            samples.append(mean(selected))
        outer_count = len(outer)
    else:
        outer = sorted({key[0] for key in values})
        for _ in range(replicates):
            selected = []
            for sampled_outer in generator.choices(outer, k=len(outer)):
                selected.extend(value for key, value in values.items() if key[0] == sampled_outer)
            samples.append(mean(selected))
        outer_count = len(outer)
    ordered = sorted(samples)
    return {
        "mean": mean(values.values()),
        "ci95_low": _percentile(ordered, 0.025),
        "ci95_high": _percentile(ordered, 0.975),
        "observations": len(values),
        "outer_clusters": outer_count,
    }


def _percentile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    args = parser.parse_args()
    payload = write_report(args.output, replicates=args.replicates, seed=args.seed)
    print(json.dumps({"output": str(args.output), "retuned": payload["retuned_synthetic_status"]}))


if __name__ == "__main__":
    main()

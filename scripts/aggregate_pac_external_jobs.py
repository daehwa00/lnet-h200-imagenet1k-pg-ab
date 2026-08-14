# ruff: noqa: EM102, TRY003
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-results", type=Path)
    arguments = parser.parse_args()
    rows = _collect(arguments.jobs_root, "external_comparisons.csv")
    matches = _collect(arguments.jobs_root, "external_parameter_matches.csv")
    _write_bundle(arguments.output_root, rows, matches, "PAC Additional External Tasks")
    if arguments.base_results is not None:
        combined = [*_read(arguments.base_results), *rows]
        combined_root = arguments.output_root.parent / f"{arguments.output_root.name}-combined"
        _write_bundle(combined_root, combined, [], "PAC Combined External Tasks")


def _collect(root: Path, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob(f"*/results/{filename}")):
        rows.extend(_read(path))
    key_fields = ("dataset", "model", "seed") if "comparisons" in filename else ("dataset", "model")
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        if key in unique and unique[key] != row:
            raise RuntimeError(f"conflicting rows for {key}")
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_bundle(
    root: Path,
    rows: list[dict[str, str]],
    matches: list[dict[str, str]],
    title: str,
) -> None:
    results = root / "results"
    reports = root / "reports"
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "external_comparisons.csv", rows)
    if matches:
        _write_csv(results / "external_parameter_matches.csv", matches)
    (reports / "external_comparisons.md").write_text(
        _report(title, rows), encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(title: str, rows: list[dict[str, str]]) -> str:
    done = [row for row in rows if row.get("status") == "done"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in done:
        grouped[(row["dataset"], row["model"])].append(row)
    lines = [
        f"# {title}",
        "",
        f"- completed rows: `{len(done)}/{len(rows)}`",
        f"- datasets: `{len({row['dataset'] for row in rows})}`",
        f"- failed rows: `{sum(row.get('status') == 'failed' for row in rows)}`",
        f"- unavailable rows: `{sum(row.get('status') == 'unavailable' for row in rows)}`",
        "",
        "| Dataset | Model | Metric | Mean | Seed SD | Params | Runs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (dataset, model), group in sorted(grouped.items()):
        metric = _primary_metric(dataset, group[0]["objective"])
        values = [float(row[metric]) for row in group]
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(
            f"| {dataset} | {model} | {metric} | {statistics.mean(values):.6f} | "
            f"{deviation:.6f} | {group[0]['params_trainable']} | {len(group)} |"
        )
    return "\n".join(lines) + "\n"


def _primary_metric(dataset: str, objective: str) -> str:
    if objective == "multiclass":
        return "accuracy"
    if objective == "forecasting":
        return "mse"
    if dataset == "audioset-balanced":
        return "macro_auprc"
    return "macro_auroc"


if __name__ == "__main__":
    main()

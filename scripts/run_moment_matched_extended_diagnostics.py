"""Extended diagnostics for the L=4 moment-matched spectral experiment.

This script is intentionally separate from the paper-generating experiment. It
adds four sequence-model baselines, a fixed-random-pole control, raw lag-0:8
statistics, learned-pole phase audits, and a sequence-length sweep. All scores
use synthetic TRAIN-derived validation samples; no benchmark TEST split exists.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Literal, cast

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_moment_matched_spectral_experiment import (  # noqa: E402
    EPSILONS,
    EPOCHS,
    FIRST_DIFFERING_LAG,
    MODEL_DIM,
    MODES,
    SEEDS,
    TRAIN_COUNT,
    TRIAL,
    VALIDATION_COUNT,
    balanced_accuracy,
    bayes_balanced_accuracy,
    empirical_autocovariances,
    make_task,
    moment_matched_samples,
)
from lnet.pac_confirmatory_baselines import (  # noqa: E402
    ConfirmatoryFamily,
    build_confirmatory_family,
    confirmatory_trial_spec,
)
from lnet.pac_final_two_scan_ablation import FinalTwoScanAblation  # noqa: E402
from lnet.pac_metrics import count_parameters  # noqa: E402
from lnet.pac_training import classification_metric_bundle, train_classifier  # noqa: E402
from lnet.pac_types import PACClassificationTask, PACDevice, PACExperimentConfig  # noqa: E402

Section = Literal["baseline", "poles", "length"]
BASELINES: tuple[ConfirmatoryFamily, ...] = ("s4d", "lru", "gru", "transformer")
BASELINE_TRIALS: dict[ConfirmatoryFamily, int] = {
    "s4d": 5,
    "lru": 4,
    "gru": 6,
    "transformer": 6,
}
POLE_VARIANTS = ("full", "fixed_random_poles")
LENGTHS = (64, 128, 256, 512)
DEFAULT_ROOT = Path(".omx/results/pac-moment-matched-extended-20260727")


@dataclass(frozen=True, slots=True)
class Job:
    section: Section
    epsilon: float
    seed: int
    variant: str
    length: int = 128

    @property
    def key(self) -> str:
        eps = f"{self.epsilon:.3f}".replace(".", "p")
        return (
            f"{self.section}__eps{eps}__T{self.length}__seed{self.seed}"
            f"__{self.variant}"
        )


def jobs() -> list[Job]:
    baseline = [
        Job("baseline", epsilon, seed, family)
        for epsilon in EPSILONS
        for seed in SEEDS
        for family in BASELINES
    ]
    poles = [
        Job("poles", epsilon, seed, variant)
        for epsilon in EPSILONS
        for seed in SEEDS
        for variant in POLE_VARIANTS
    ]
    length = [
        Job("length", 0.1, seed, "full", length)
        for length in LENGTHS
        if length != 128
        for seed in SEEDS
    ]
    return baseline + poles + length


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{random.randrange(1 << 30)}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_length_task(seed: int, epsilon: float, length: int) -> PACClassificationTask:
    generator = torch.Generator().manual_seed(seed)
    train_x, train_y = moment_matched_samples(TRAIN_COUNT, length, epsilon, generator)
    valid_x, valid_y = moment_matched_samples(VALIDATION_COUNT, length, epsilon, generator)
    return PACClassificationTask(
        f"MomentMatchedL4-eps{epsilon:.3f}-T{length}",
        train_x,
        train_y,
        valid_x,
        valid_y,
        train_x[:0],
        train_y[:0],
        2,
    )


def raw_moment_bacc(task: PACClassificationTask, max_lag: int) -> float:
    train = empirical_autocovariances(task.train_inputs, max_lag)
    valid = empirical_autocovariances(task.validation_inputs, max_lag)
    prototypes = torch.stack(
        [train[task.train_labels == label].mean(0) for label in (0, 1)]
    )
    scale = train.std(0, unbiased=True).clamp_min(1.0e-6)
    distances = ((valid[:, None] - prototypes[None] ) / scale).square().sum(-1)
    return balanced_accuracy(distances.argmin(1), task.validation_labels)


def config_for(job: Job, family: ConfirmatoryFamily = "pac_tf") -> PACExperimentConfig:
    trial = BASELINE_TRIALS.get(family, TRIAL)
    spec = confirmatory_trial_spec(family, trial)
    return PACExperimentConfig(
        TRAIN_COUNT,
        VALIDATION_COUNT,
        0,
        job.length,
        raw_input_dim=1,
        output_dim=2,
        model_dim=MODEL_DIM,
        modes=MODES,
        epochs=EPOCHS,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        seeds=(job.seed,),
        device="cuda",
        optimizer_mode="fused",
    )


def build_model(job: Job) -> tuple[nn.Module, PACExperimentConfig, str]:
    if job.section == "baseline":
        family = cast("ConfirmatoryFamily", job.variant)
        config = config_for(job, family)
        trial = BASELINE_TRIALS[family]
        model = build_confirmatory_family(
            family,
            MODEL_DIM,
            config,
            2,
            validation_trial=trial,
            input_dim=1,
        )
        return model, config, confirmatory_trial_spec(family, trial).architecture_label
    config = config_for(job)
    model = FinalTwoScanAblation(
        config,
        2,
        variant=cast("Literal['full', 'fixed_random_poles']", job.variant),
        objective="classification",
        random_pole_seed=17_071 + job.seed,
    )
    return model, config, job.variant


def pole_phases(model: nn.Module) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name, block in (
        ("direct", model.forward_block),
        ("cascaded", model.backward_block),
    ):
        phases = torch.pi * torch.tanh(block.raw_frequency.detach())
        result[name] = [float(value) for value in phases.cpu()]
    return result


def phase_mass_distance(phases: dict[str, list[float]]) -> float:
    """Mean distance to an extremum of |cos(5 theta)|, normalized by pi."""
    targets = torch.arange(-5, 6, dtype=torch.float64) * (math.pi / 5.0)
    values = torch.tensor(
        phases["direct"] + phases["cascaded"],
        dtype=torch.float64,
    )
    return float((values[:, None] - targets[None]).abs().min(1).values.mean() / math.pi)


def phase_extremum_alignment(phases: dict[str, list[float]]) -> float:
    """Mean absolute spectral-difference alignment |cos(5 omega)|."""
    values = torch.tensor(phases["direct"] + phases["cascaded"], dtype=torch.float64)
    return float(torch.cos(5.0 * values).abs().mean())


def result_path(root: Path, job: Job) -> Path:
    return root / "completed" / f"{job.key}.json"


def run_job(job: Job, root: Path, device: PACDevice) -> dict[str, object]:
    output = result_path(root, job)
    if output.exists():
        return cast("dict[str, object]", json.loads(output.read_text()))
    runtime = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    _seed_everything(job.seed)
    task = (
        make_task(job.seed, job.epsilon)
        if job.length == 128
        else make_length_task(job.seed, job.epsilon, job.length)
    )
    model, config, architecture = build_model(job)
    model = model.to(runtime)
    initial_phases = pole_phases(model) if job.section != "baseline" else None
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        config,
        runtime,
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    metrics = classification_metric_bundle(
        model,
        task.validation_inputs.to(runtime),
        task.validation_labels.to(runtime),
        batch_size=config.batch_size,
    )
    final_phases = pole_phases(model) if job.section != "baseline" else None
    row = {
        "schema": "alphabet.moment_matched_extended.result.v1",
        **asdict(job),
        "job_key": job.key,
        "architecture": architecture,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "validation_error": 1.0 - metrics.balanced_accuracy,
        "bayes_balanced_accuracy": bayes_balanced_accuracy(task, job.epsilon),
        "raw_moment_0_4_balanced_accuracy": raw_moment_bacc(task, 4),
        "raw_moment_0_8_balanced_accuracy": raw_moment_bacc(task, 8),
        "parameters": count_parameters(model),
        "best_epoch": outcome.best_epoch,
        "elapsed_seconds": perf_counter() - started,
        "initial_pole_phases": initial_phases,
        "final_pole_phases": final_phases,
        "initial_phase_mass_distance_pi": (
            phase_mass_distance(initial_phases) if initial_phases else None
        ),
        "final_phase_mass_distance_pi": (
            phase_mass_distance(final_phases) if final_phases else None
        ),
        "initial_phase_extremum_alignment": (
            phase_extremum_alignment(initial_phases) if initial_phases else None
        ),
        "final_phase_extremum_alignment": (
            phase_extremum_alignment(final_phases) if final_phases else None
        ),
        "evaluation_split": "synthetic TRAIN-derived validation",
        "official_test_accessed": False,
    }
    _write_json(output, row)
    return row


def prepare(root: Path) -> dict[str, object]:
    active = jobs()
    root.mkdir(parents=True, exist_ok=True)
    (root / "queue.jsonl").write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in active)
    )
    _write_json(
        root / "contract.json",
        {
            "schema": "alphabet.moment_matched_extended.contract.v1",
            "status": "frozen",
            "jobs": len(active),
            "baselines": list(BASELINES),
            "pole_variants": list(POLE_VARIANTS),
            "lengths": list(LENGTHS),
            "epsilons": list(EPSILONS),
            "seeds": list(SEEDS),
            "width": MODEL_DIM,
            "modes": MODES,
            "epochs": EPOCHS,
            "alphabet_validation_trial": TRIAL,
            "baseline_validation_trials": BASELINE_TRIALS,
            "official_test_accessed": False,
        },
    )
    return status(root)


def selected_jobs(
    section: str | None,
    shard_index: int,
    shard_count: int,
) -> list[Job]:
    active = [job for job in jobs() if section is None or job.section == section]
    return [job for index, job in enumerate(active) if index % shard_count == shard_index]


def run(
    root: Path,
    device: PACDevice,
    section: str | None,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    for job in selected_jobs(section, shard_index, shard_count):
        run_job(job, root, device)
    return status(root)


def status(root: Path) -> dict[str, object]:
    active = jobs()
    completed = sum(result_path(root, job).exists() for job in active)
    by_section = {
        section: {
            "expected": sum(job.section == section for job in active),
            "completed": sum(
                job.section == section and result_path(root, job).exists() for job in active
            ),
        }
        for section in ("baseline", "poles", "length")
    }
    return {
        "expected": len(active),
        "completed": completed,
        "remaining": len(active) - completed,
        "done": completed == len(active),
        "by_section": by_section,
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "sample_sd": stdev(values)}


def _slope(xs: list[float], ys: list[float]) -> float:
    x = torch.log(torch.tensor(xs, dtype=torch.float64))
    y = torch.log(torch.tensor(ys, dtype=torch.float64).clamp_min(1.0e-6))
    return float(((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).square().sum())


def report(root: Path) -> dict[str, object]:
    current = status(root)
    if not current["done"]:
        return current
    rows = [json.loads(result_path(root, job).read_text()) for job in jobs()]
    baseline_summary = {}
    for epsilon in EPSILONS:
        baseline_summary[f"{epsilon:.3f}"] = {}
        for family in BASELINES:
            chosen = [
                row
                for row in rows
                if row["section"] == "baseline"
                and row["epsilon"] == epsilon
                and row["variant"] == family
            ]
            baseline_summary[f"{epsilon:.3f}"][family] = {
                **_stats([row["validation_balanced_accuracy"] for row in chosen]),
                "parameters": chosen[0]["parameters"],
            }
        representative = next(
            row
            for row in rows
            if row["section"] == "baseline" and row["epsilon"] == epsilon
        )
        baseline_summary[f"{epsilon:.3f}"]["raw_moment_0_8"] = _stats(
            [
                row["raw_moment_0_8_balanced_accuracy"]
                for row in rows
                if row["section"] == "baseline"
                and row["epsilon"] == epsilon
                and row["variant"] == BASELINES[0]
            ]
        )
        baseline_summary[f"{epsilon:.3f}"]["bayes"] = _stats(
            [
                row["bayes_balanced_accuracy"]
                for row in rows
                if row["section"] == "baseline"
                and row["epsilon"] == epsilon
                and row["variant"] == BASELINES[0]
            ]
        )
        del representative
    pole_summary = {}
    for epsilon in EPSILONS:
        pole_summary[f"{epsilon:.3f}"] = {}
        for variant in POLE_VARIANTS:
            chosen = [
                row
                for row in rows
                if row["section"] == "poles"
                and row["epsilon"] == epsilon
                and row["variant"] == variant
            ]
            pole_summary[f"{epsilon:.3f}"][variant] = {
                **_stats([row["validation_balanced_accuracy"] for row in chosen]),
                "parameters": chosen[0]["parameters"],
                "initial_phase_mass_distance_pi": _stats(
                    [row["initial_phase_mass_distance_pi"] for row in chosen]
                ),
                "final_phase_mass_distance_pi": _stats(
                    [row["final_phase_mass_distance_pi"] for row in chosen]
                ),
                "initial_phase_extremum_alignment": _stats(
                    [row["initial_phase_extremum_alignment"] for row in chosen]
                ),
                "final_phase_extremum_alignment": _stats(
                    [row["final_phase_extremum_alignment"] for row in chosen]
                ),
            }
    length_rows = [row for row in rows if row["section"] == "length"]
    original_path = ROOT / ".omx/results/pac-moment-matched-l4-20260727/summary.json"
    if not original_path.exists():
        original_path = (
            Path.cwd() / ".omx/results/pac-moment-matched-l4-20260727/summary.json"
        )
    original = json.loads(original_path.read_text())
    length_summary: dict[str, object] = {}
    full_errors: list[float] = []
    bayes_errors: list[float] = []
    for length in LENGTHS:
        if length == 128:
            full = original["summary"]["0.100"]["full"]
            bayes = original["summary"]["0.100"]["bayes"]
            full_stats = {"mean": 1.0 - full["mean"], "sample_sd": full["sample_sd"]}
            bayes_stats = {"mean": 1.0 - bayes["mean"], "sample_sd": bayes["sample_sd"]}
        else:
            chosen = [row for row in length_rows if row["length"] == length]
            full_stats = _stats([row["validation_error"] for row in chosen])
            bayes_stats = _stats([1.0 - row["bayes_balanced_accuracy"] for row in chosen])
        length_summary[str(length)] = {"full_error": full_stats, "bayes_error": bayes_stats}
        full_errors.append(full_stats["mean"])
        bayes_errors.append(bayes_stats["mean"])
    payload = {
        "schema": "alphabet.moment_matched_extended.summary.v1",
        "status": current,
        "baseline": baseline_summary,
        "poles": pole_summary,
        "length": {
            "points": length_summary,
            "full_log_error_log_T_slope": _slope(list(LENGTHS), full_errors),
            "bayes_log_error_log_T_slope": _slope(list(LENGTHS), bayes_errors),
        },
        "official_test_accessed": False,
    }
    _write_json(root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "status", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--section", choices=("baseline", "poles", "length"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count)")
    if args.command == "prepare":
        payload = prepare(args.root)
    elif args.command == "run":
        payload = run(
            args.root,
            cast("PACDevice", args.device),
            args.section,
            args.shard_index,
            args.shard_count,
        )
    elif args.command == "report":
        payload = report(args.root)
    else:
        payload = status(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

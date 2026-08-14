"""Actual-UCR full-fit validation for baseline execution backends."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_compact_h_only_systems import StaticTransformerInference
from .pac_confirmatory_baselines import (
    build_confirmatory_family,
    confirmatory_trial_spec,
)
from .pac_eval_sections import clean_validation_classification_task
from .pac_metrics import count_parameters
from .pac_model_training_comparison import (
    BenchmarkConfig as GraphBenchmarkConfig,
)
from .pac_model_training_comparison import (
    GenericCudaGraphRuntime,
)
from .pac_real_data import ensure_ucr_train_only
from .pac_training import (
    classification_metric_bundle,
    evaluate_classification_loss,
    train_classifier,
)
from .pac_types import PACClassificationTask, PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

Family = Literal[
    "cnn1d",
    "tcn",
    "gru",
    "lstm",
    "mamba",
    "s4d",
    "lru",
    "s5",
    "transformer",
]
Backend = Literal["eager_default", "eager_fused", "compile_fused", "cuda_graph_full_step"]

FAMILIES: Final[tuple[Family, ...]] = (
    "cnn1d",
    "tcn",
    "gru",
    "lstm",
    "mamba",
    "s4d",
    "lru",
    "s5",
    "transformer",
)
BACKENDS: Final[tuple[Backend, ...]] = (
    "eager_default",
    "eager_fused",
    "compile_fused",
    "cuda_graph_full_step",
)
SELECTED_TRIAL: Final[dict[Family, int]] = {
    "cnn1d": 3,
    "tcn": 6,
    "gru": 6,
    "lstm": 5,
    "mamba": 6,
    "s4d": 5,
    "lru": 6,
    "s5": 6,
    "transformer": 6,
}


@dataclass(frozen=True, slots=True)
class FullFitConfig:
    dataset: str = "Wafer"
    width: int = 64
    epochs: int = 20
    split_seed: int = 7
    seeds: tuple[int, ...] = (7, 11)
    graph_warmups: int = 3


def run_fullfit_screen(
    *,
    family: Family,
    data_root: Path,
    output: Path,
    config: FullFitConfig,
    backends: tuple[Backend, ...] = BACKENDS,
    device: str = "cuda",
) -> dict[str, object]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        message = "full-fit backend screen requires CUDA"
        raise ValueError(message)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = ensure_ucr_train_only(config.dataset, data_root, allow_download=False)
    task = clean_validation_classification_task(dataset, config.split_seed)
    spec = confirmatory_trial_spec(family, SELECTED_TRIAL[family])
    experiment = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        task.test_inputs.shape[0],
        task.train_inputs.shape[1],
        raw_input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        model_dim=config.width,
        modes=16,
        epochs=config.epochs,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        seeds=config.seeds,
        device="cuda",
        precision="fp32",
    )
    rows: list[dict[str, object]] = []
    payload = _payload(family, task, experiment, config, backends, rows)
    _write_json(output, payload)
    for seed in config.seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        base_model = _build_model(family, task, experiment, config.width)
        state_dict = copy.deepcopy(base_model.state_dict())
        parameter_count = count_parameters(base_model)
        del base_model
        seed_rows: list[dict[str, object]] = []
        trained_states: dict[Backend, dict[str, Tensor]] = {}
        for backend in backends:
            row, final_state = _fit_backend(
                family,
                backend,
                state_dict,
                task,
                experiment,
                config,
                seed,
                parameter_count,
                device,
            )
            seed_rows.append(row)
            trained_states[backend] = final_state
        reference_state = trained_states.get("eager_fused")
        for row in seed_rows:
            backend = cast("Backend", row["backend"])
            row["final_parameter_max_abs_error_vs_fused"] = (
                _state_error(reference_state, trained_states[backend])
                if reference_state is not None
                else None
            )
        rows.extend(seed_rows)
        payload = _payload(family, task, experiment, config, backends, rows)
        _write_json(output, payload)
    return payload


def _fit_backend(
    family: Family,
    backend: Backend,
    initial_state: dict[str, Tensor],
    task: PACClassificationTask,
    experiment: PACExperimentConfig,
    screen: FullFitConfig,
    seed: int,
    parameter_count: int,
    device: str,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _build_model(family, task, experiment, screen.width)
    model.load_state_dict(initial_state, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    if backend == "cuda_graph_full_step" and family == "transformer":
        model = StaticTransformerInference(
            model,
            length=task.train_inputs.shape[1],
            device=next(model.parameters()).device,
        )
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    if backend == "cuda_graph_full_step":
        best_epoch, validation_loss, graph_steps, tail_steps = _train_with_cuda_graph(
            model,
            task,
            experiment,
            seed=seed,
            graph_warmups=screen.graph_warmups,
            device=device,
        )
    else:
        run_config = replace(
            experiment,
            compile_mode="reduce-overhead" if backend == "compile_fused" else "none",
            optimizer_mode="default" if backend == "eager_default" else "fused",
        )
        outcome = train_classifier(
            model,
            task,
            run_config,
            device,
            seed,
            evaluate_test=False,
            restore_best_validation=True,
        )
        best_epoch = outcome.best_epoch
        validation_loss = outcome.validation_loss
        graph_steps = 0
        tail_steps = 0
    elapsed = perf_counter() - started
    validation_inputs = task.validation_inputs.to(device=device)
    validation_labels = task.validation_labels.to(device=device)
    metrics = classification_metric_bundle(
        model,
        validation_inputs,
        validation_labels,
        batch_size=experiment.batch_size,
    )
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    row: dict[str, object] = {
        "family": family,
        "backend": backend,
        "seed": seed,
        "status": "done",
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "validation_accuracy": metrics.accuracy,
        "parameters": parameter_count,
        "graph_full_steps": graph_steps,
        "eager_tail_steps": tail_steps,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    del model, validation_inputs, validation_labels
    torch.cuda.empty_cache()
    return row, state


def _train_with_cuda_graph(
    model: nn.Module,
    task: PACClassificationTask,
    config: PACExperimentConfig,
    *,
    seed: int,
    graph_warmups: int,
    device: str,
) -> tuple[int, float, int, int]:
    model.train()
    train_inputs = task.train_inputs.to(device=device)
    train_labels = task.train_labels.to(device=device)
    validation_inputs = task.validation_inputs.to(device=device)
    validation_labels = task.validation_labels.to(device=device)
    generator = torch.Generator(device=train_inputs.device).manual_seed(seed)
    runtime: GenericCudaGraphRuntime | None = None
    optimizer: torch.optim.AdamW | None = None
    best_state: dict[str, Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    graph_steps = 0
    tail_steps = 0
    graph_config = GraphBenchmarkConfig(
        graph_warmups=graph_warmups,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )
    for epoch in range(config.epochs):
        order = torch.randperm(
            train_inputs.shape[0],
            device=train_inputs.device,
            generator=generator,
        )
        for indices in order.split(config.batch_size):
            inputs = train_inputs[indices]
            labels = train_labels[indices]
            if inputs.shape[0] == config.batch_size:
                if runtime is None:
                    runtime = GenericCudaGraphRuntime(
                        model,
                        "full",
                        inputs,
                        labels,
                        config=graph_config,
                    )
                    optimizer = runtime.optimizer
                runtime.step(inputs, labels)
                graph_steps += 1
                continue
            if optimizer is None:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    fused=True,
                    capturable=True,
                )
            optimizer.zero_grad(set_to_none=runtime is None)
            loss = functional.cross_entropy(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip_norm,
                foreach=True,
            )
            optimizer.step()
            tail_steps += 1
        current_validation = evaluate_classification_loss(
            model,
            validation_inputs,
            validation_labels,
            batch_size=config.batch_size,
        )
        if current_validation < best_validation:
            best_validation = current_validation
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        message = "CUDA Graph full fit produced no validation checkpoint"
        raise RuntimeError(message)
    model.load_state_dict(best_state)
    return best_epoch, best_validation, graph_steps, tail_steps


def _build_model(
    family: Family,
    task: PACClassificationTask,
    config: PACExperimentConfig,
    width: int,
) -> nn.Module:
    return build_confirmatory_family(
        family,
        width,
        config,
        task.class_count,
        validation_trial=SELECTED_TRIAL[family],
        input_dim=task.train_inputs.shape[-1],
    )


def _state_error(
    reference: dict[str, Tensor],
    candidate: dict[str, Tensor],
) -> float:
    if set(reference) != set(candidate):
        return math.inf
    return max(
        float((reference[key] - candidate[key]).abs().max().item())
        for key in reference
    )


def _payload(
    family: Family,
    task: PACClassificationTask,
    experiment: PACExperimentConfig,
    config: FullFitConfig,
    backends: tuple[Backend, ...],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    elapsed_by_backend = {
        backend: [
            cast("float", row["elapsed_seconds"])
            for row in rows
            if row["backend"] == backend
        ]
        for backend in backends
    }
    summary = {
        backend: {
            "runs": len(values),
            "mean_elapsed_seconds": statistics.mean(values) if values else None,
            "warm_cache_elapsed_seconds": values[-1] if len(values) > 1 else None,
        }
        for backend, values in elapsed_by_backend.items()
    }
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema": "pac.baseline_fullfit_backend_screen.v1",
        "environment": {
            "device": properties.name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": False,
        },
        "protocol": {
            "purpose": "optimization screening only; not paper test evidence",
            "official_test_accessed": False,
            "split": "clean stratified train/validation split",
            "compile_cache_reused_across_seeds_in_one_process": True,
            "same_initialization_within_seed": True,
            "same_gpu_batch_permutation_within_seed": True,
            "precision": "fp32",
        },
        "family": family,
        "backends": list(backends),
        "screen_config": asdict(config),
        "experiment": asdict(experiment),
        "task": {
            "label": task.label,
            "train_count": task.train_inputs.shape[0],
            "validation_count": task.validation_inputs.shape[0],
            "sequence_length": task.train_inputs.shape[1],
            "input_dim": task.train_inputs.shape[-1],
            "class_count": task.class_count,
        },
        "rows": rows,
        "summary": summary,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(path)


def _csv_tuple(value: str, *, allowed: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        message = f"expected subset of {allowed}, got {value!r}"
        raise argparse.ArgumentTypeError(message)
    return values


def _positive_int_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        message = "expected comma-separated integers"
        raise argparse.ArgumentTypeError(message) from error
    if not values or any(item <= 0 for item in values):
        message = "all values must be positive"
        raise argparse.ArgumentTypeError(message)
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--dataset", default="Wafer")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--seeds", default="7,11")
    parser.add_argument("--graph-warmups", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    backends = cast(
        "tuple[Backend, ...]",
        _csv_tuple(args.backends, allowed=cast("tuple[str, ...]", BACKENDS)),
    )
    config = FullFitConfig(
        dataset=args.dataset,
        width=args.width,
        epochs=args.epochs,
        split_seed=args.split_seed,
        seeds=_positive_int_csv(args.seeds),
        graph_warmups=args.graph_warmups,
    )
    payload = run_fullfit_screen(
        family=cast("Family", args.family),
        data_root=args.data_root,
        output=args.output,
        config=config,
        backends=backends,
    )
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

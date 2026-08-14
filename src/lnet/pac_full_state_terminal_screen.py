from __future__ import annotations

import argparse
import gc
import traceback
from pathlib import Path
from time import perf_counter
from typing import cast

import torch

from .pac_efp16_ablation import _atomic_json  # pyright: ignore[reportPrivateUsage]
from .pac_eval_sections import clean_validation_classification_task
from .pac_full_state_terminal_analyzer import (
    FULL_STATE_INJECTION_VARIANTS,
    STATE_GRADIENT_SCALES,
    FullStateInjectionVariant,
    build_full_state_injection_analyzer,
)
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_train_only
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig


def run_screen_job(
    dataset: str,
    seed: int,
    variant: FullStateInjectionVariant,
    output_root: Path,
    device: str,
) -> Path:
    result_path = output_root / "completed" / f"{dataset}_{variant}_seed{seed}.json"
    if result_path.exists():
        return result_path
    try:
        ucr = ensure_ucr_train_only(dataset, Path(".omx/data/ucr"), allow_download=True)
        task = clean_validation_classification_task(ucr, seed)
        config = PACExperimentConfig(
            task.train_inputs.shape[0],
            task.validation_inputs.shape[0],
            0,
            task.train_inputs.shape[1],
            raw_input_dim=task.train_inputs.shape[-1],
            output_dim=task.class_count,
            model_dim=32,
            modes=16,
            epochs=100,
            batch_size=64,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
            grad_clip_norm=1.0,
            seeds=(seed,),
            device=cast("PACDevice", device),
        )
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        model = build_full_state_injection_analyzer(config, task.class_count, variant)
        started = perf_counter()
        outcome = train_classifier(
            model,
            task,
            config,
            device,
            seed,
            evaluate_test=False,
            restore_best_validation=True,
        )
        metrics = classification_metric_bundle(
            model,
            task.validation_inputs.to(device=device),
            task.validation_labels.to(device=device),
            batch_size=config.batch_size,
        )
        _atomic_json(
            result_path,
            {
                "schema": "pac_full_state_terminal_ucr_screen_result.v1",
                "status": "done",
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "official_test_accessed": False,
                "best_epoch": outcome.best_epoch,
                "validation_loss": outcome.validation_loss,
                "validation_accuracy": metrics.accuracy,
                "validation_macro_f1": metrics.macro_f1,
                "validation_weighted_f1": metrics.weighted_f1,
                "validation_balanced_accuracy": metrics.balanced_accuracy,
                "params_trainable": count_parameters(model),
                "train_seconds": perf_counter() - started,
                "config": {
                    "model_dim": 32,
                    "modes": 16,
                    "epochs": 100,
                    "batch_size": 64,
                    "learning_rate": 3.0e-3,
                    "weight_decay": 1.0e-4,
                    "grad_clip_norm": 1.0,
                    "second_projection_input_dim": (
                        32 if variant == "full_late_dense" else 64
                    ),
                    "fusion_output_dim": 32,
                    "state_adapter_input_dim": (
                        32 if variant == "full_late_dense" else 0
                    ),
                    "state_adapter_output_dim": (
                        32 if variant == "full_late_dense" else 0
                    ),
                    "fusion_init": (
                        "identity H1 projection + zero state adapter"
                        if variant == "full_late_dense"
                        else "[identity|zero]"
                    ),
                    "state_evidence": variant,
                    "state_gradient_scale": (
                        1.0 if variant == "full_late_dense" else STATE_GRADIENT_SCALES[variant]
                    ),
                    "state_injection_position": (
                        "terminal_excitation"
                        if variant == "full_late_dense"
                        else "before_terminal_local_lift"
                    ),
                    "terminal_local_lift_retained": True,
                    "terminal_moment_readout_retained": True,
                },
            },
        )
    except Exception as error:
        _atomic_json(
            output_root / "failed" / result_path.name,
            {
                "schema": "pac_full_state_terminal_ucr_screen_result.v1",
                "status": "failed",
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "official_test_accessed": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--variant", choices=FULL_STATE_INJECTION_VARIANTS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_screen_job(
        args.dataset,
        args.seed,
        args.variant,
        args.output_root,
        args.device,
    )


if __name__ == "__main__":
    main()

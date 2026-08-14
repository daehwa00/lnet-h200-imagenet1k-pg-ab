#!/usr/bin/env python3
"""Run the restart-safe ALPHABET-2D synthetic campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from lnet.alphabet2d_experiment import (
    TASKS,
    VARIANTS,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    run_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny CPU/GPU integration recipe, still with exact pole math.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the bounded 32x32 three-seed RTX 4090 diagnostic campaign.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Override the frozen default seed list in a new campaign root.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = ExperimentConfig()
    if args.tasks:
        config = replace(config, tasks=tuple(args.tasks))
    if args.variants:
        config = replace(config, variants=tuple(args.variants))
    if args.seeds:
        config = replace(config, seeds=tuple(args.seeds))
    if args.smoke:
        config = replace(
            config,
            model=ModelConfig(
                image_size=16,
                patch_size=4,
                model_dim=8,
                modes=2,
                depth=1,
                windows="global",
                recurrence_backend="real2d_loop",
            ),
            training=TrainingConfig(
                train_per_class=4,
                validation_per_class=2,
                test_per_class=2,
                batch_size=4,
                max_epochs=1,
                patience=1,
                throughput_warmup=0,
                throughput_repetitions=1,
            ),
            off_axis=replace(config.off_axis, height=16, width=16),
            equal_power_phase=replace(config.equal_power_phase, height=16, width=16),
        )
    elif args.pilot:
        config = replace(
            config,
            seeds=tuple(args.seeds or (11, 23, 47)),
            model=ModelConfig(
                image_size=32,
                patch_size=2,
                model_dim=32,
                modes=16,
                depth=1,
                windows="global_2x2",
                recurrence_backend="auto",
            ),
            training=TrainingConfig(
                train_per_class=256,
                validation_per_class=128,
                test_per_class=256,
                batch_size=64,
                max_epochs=40,
                patience=8,
                learning_rate=3.0e-3,
                weight_decay=1.0e-4,
                grad_clip_norm=1.0,
                throughput_warmup=2,
                throughput_repetitions=5,
            ),
        )
    result = run_campaign(
        args.root,
        config,
        device=args.device,
        max_jobs=args.max_jobs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()

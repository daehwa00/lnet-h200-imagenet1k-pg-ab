#!/usr/bin/env python3
"""Train the frozen LNet K96/P128x4/D2262 on ImageNet-1K."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_k_family_wave_a_imagenet100 as family
import run_h200_baseline_worker as worker
from torch import Tensor, nn

MODEL_KEY = "lnet_k96_p128x4_d2262_optimized_v2"
DISPLAY_NAME = "LNet-K96-P128x4-D2262-OptimizedV2"
VARIANT = "XL-K96-U125"
SEEDS = (501, 509, 521)
LEARNING_RATE = 3.0e-3
EPOCHS = 100
EXPECTED_PARAMETERS = 3_253_224
WANDB_ENTITY = "daehwa"
WANDB_PROJECT = "alphabet2d-imagenet1k-h200-baselines"
WANDB_GROUP = "h200-imagenet1k-moga-emo-100ep-s501-v2"


class PrimaryLogitsAdapter(nn.Module):
    """Expose the canonical joint Q4 logits to the matched baseline worker."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.model(inputs)
        if not isinstance(output, tuple) or len(output) != 5:
            raise RuntimeError("LNet ImageNet-1K model lost its five-output Q-head contract")
        joint = output[0]
        if not isinstance(joint, Tensor):
            raise TypeError("LNet ImageNet-1K joint logits are not a tensor")
        return joint


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wandb-mode", choices=("disabled", "online"), default="online")
    return parser.parse_args()


def _model_spec(key: str) -> SimpleNamespace:
    if key != MODEL_KEY:
        raise ValueError(f"unsupported internal ImageNet-1K model: {key}")
    return SimpleNamespace(
        backend="internal",
        display_name=DISPLAY_NAME,
        precision="bfloat16",
    )


def _build_model(
    key: str,
    _source_root: str | Path | None,
    num_classes: int,
) -> nn.Module:
    if key != MODEL_KEY or num_classes != 1000:
        raise ValueError("LNet ImageNet-1K builder received an incompatible task")
    model = family._build(VARIANT, runtime.model_config(output_dim=num_classes))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"LNet ImageNet-1K parameter count changed: {parameters} != {EXPECTED_PARAMETERS}"
        )
    prepared = model.prepare_for_compiled_training_()
    return PrimaryLogitsAdapter(prepared)


def _source_fingerprint() -> str:
    repo = Path(__file__).resolve().parents[1]
    paths = runtime.source_dependency_paths(repo, (Path(__file__).stem,))
    return runtime.source_fingerprint(repo, paths)


def _install_worker_contract() -> None:
    original_model_spec = worker.registry.model_spec
    original_source_digests = worker._source_digests

    def model_spec(key: str) -> Any:
        return _model_spec(key) if key == MODEL_KEY else original_model_spec(key)

    def source_digests(key: str) -> dict[str, str]:
        payload = original_source_digests(key)
        if key == MODEL_KEY:
            payload["lnet_internal_closure"] = _source_fingerprint()
            payload["imagenet1k_runner"] = runtime.digest(__file__)
        return payload

    worker.registry.model_spec = model_spec
    worker._source_digests = source_digests


def _configure_wandb(mode: str, seed: int) -> None:
    if mode != "online":
        return
    run_id = hashlib.sha256(f"{WANDB_GROUP}:{MODEL_KEY}:seed{seed}".encode()).hexdigest()[:16]
    os.environ.update(
        {
            "H200_BASELINE_RUN_ID": run_id,
            "H200_BASELINE_DISPLAY_NAME": f"H200-LNet-I1K-K96-P128x4-s{seed}",
            "H200_BASELINE_TAGS_JSON": json.dumps(
                [
                    "H200",
                    "ImageNet-1K",
                    "LNet",
                    "K96",
                    "P128x4",
                    "D2262",
                    "100ep",
                    f"seed{seed}",
                ],
                separators=(",", ":"),
            ),
            "WANDB_APP_URL": "https://wandb.ai",
            "WANDB_BASE_URL": os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"),
            "WANDB_CONSOLE": "off",
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_GROUP": WANDB_GROUP,
            "WANDB_MODE": "online",
            "WANDB_PROJECT": WANDB_PROJECT,
        }
    )


def _task(args: argparse.Namespace) -> worker.BaselineTask:
    seed = int(args.seed)
    output = args.output_root.resolve() / MODEL_KEY / f"seed_{seed}"
    checkpoint = output / "checkpoint.pt"
    return worker.BaselineTask(
        phase="full",
        model_key=MODEL_KEY,
        seed=seed,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        data_root=args.data_root.resolve(),
        output_dir=output,
        result_path=output / "result.json",
        checkpoint_path=checkpoint,
        source_root=Path(__file__).resolve().parents[1],
        batch_size=args.batch_size,
        workers=args.workers,
        wandb_mode=args.wandb_mode,
        resume=checkpoint.is_file(),
    )


def main() -> None:
    args = _arguments()
    _install_worker_contract()
    _configure_wandb(str(args.wandb_mode), int(args.seed))
    os.environ.setdefault("H200_GPU_MEMORY_FRACTION", "1.0")
    result = worker.run_task(_task(args), model_builder=_build_model)
    print("LNET_K96_IMAGENET1K_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

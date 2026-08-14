#!/usr/bin/env python3
"""Train the unchanged A2D-Deep4 model and evaluate parameter EMA weights."""

# ruff: noqa: ANN401, SLF001

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_resaux1_deep4_imagenet100 as deep4
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import Tensor, nn

from lnet.complex_scan import ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterator

    from torch.utils.data import DataLoader


VARIANT = "D4-EMA"
SEEDS = (501,)
EMA_DECAY = 0.9999
heads = deep4.heads


class ParameterEMA(nn.Module):
    """Persistent parameter shadows with reversible evaluation-time swapping."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            message = "parameter EMA decay must lie in [0, 1)"
            raise ValueError(message)
        named_parameters = tuple(model.named_parameters())
        if not named_parameters:
            message = "parameter EMA requires at least one model parameter"
            raise ValueError(message)
        self.decay = float(decay)
        self.parameter_names = tuple(name for name, _ in named_parameters)
        # A tuple does not register duplicate parameters on this helper module.
        # The owning model remains the sole optimizer/state_dict parameter owner.
        object.__setattr__(
            self,
            "_tracked_parameters",
            tuple(parameter for _, parameter in named_parameters),
        )
        for index, (_, parameter) in enumerate(named_parameters):
            self.register_buffer(
                self._shadow_name(index),
                parameter.detach().clone(),
                persistent=True,
            )
        self.register_buffer(
            "num_updates",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

    @staticmethod
    def _shadow_name(index: int) -> str:
        return f"shadow_{index:04d}"

    def _tracked(self) -> tuple[nn.Parameter, ...]:
        return cast("tuple[nn.Parameter, ...]", self._tracked_parameters)

    def _shadows(self) -> tuple[Tensor, ...]:
        return tuple(
            cast("Tensor", getattr(self, self._shadow_name(index)))
            for index in range(len(self.parameter_names))
        )

    @torch.no_grad()
    def update_(self) -> None:
        one_minus_decay = 1.0 - self.decay
        for parameter, shadow in zip(
            self._tracked(),
            self._shadows(),
            strict=True,
        ):
            shadow.mul_(self.decay).add_(parameter.detach(), alpha=one_minus_decay)
        self.num_updates.add_(1)

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        parameters = self._tracked()
        backups = [parameter.detach().clone() for parameter in parameters]
        try:
            with torch.no_grad():
                for parameter, shadow in zip(
                    parameters,
                    self._shadows(),
                    strict=True,
                ):
                    parameter.copy_(shadow)
            yield
        finally:
            with torch.no_grad():
                for parameter, backup in zip(parameters, backups, strict=True):
                    parameter.copy_(backup)


def _ema(model: nn.Module) -> ParameterEMA:
    tracker = getattr(model, "parameter_ema", None)
    if not isinstance(tracker, ParameterEMA):
        message = "A2D-Deep4 EMA model is missing its parameter tracker"
        raise TypeError(message)
    return tracker


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported A2D Deep4 EMA variant: {variant}"
        raise ValueError(message)
    model = deep4._build(deep4.VARIANT, config)
    model.parameter_ema = ParameterEMA(model, EMA_DECAY)
    return model


def _train_epoch(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mixup_generator: Any,
    mixup_alpha: float,
    precision: str,
    gradient_accumulation_steps: int = 1,
    channels_last: bool = False,
) -> dict[str, float]:
    """Run the exact Deep4 loop, extending only completed optimizer steps."""
    tracker = _ema(model)
    original_step = optimizer.step

    def step_and_update(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        tracker.update_()
        return result

    optimizer.step = step_and_update  # type: ignore[method-assign]
    try:
        source = resaux_base
        return source.structured._train_epoch(
            model,
            runtime,
            loader,
            optimizer,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=mixup_alpha,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            channels_last=channels_last,
        )
    finally:
        optimizer.step = original_step  # type: ignore[method-assign]


def _evaluate(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    """Evaluate EMA weights and restore live training parameters afterwards."""
    with _ema(model).average_parameters():
        return resaux_base.heads._evaluate(
            model,
            runtime,
            loader,
            device,
            precision=precision,
            channels_last=channels_last,
        )


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = deep4._wandb_model_metrics(model)
    tracker = _ema(model)
    metrics["ema/decay"] = tracker.decay
    metrics["ema/updates"] = float(tracker.num_updates)
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = deep4._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload.update(
        {
            "schema": "lnet.a2d.deep4_ema.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch unchanged Deep4 with parameter EMA",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        VARIANT: {
            "backbone": {
                "name": "A2D-D4-PathMix-PostCarry-PostFFN-4Stage",
                "modes": [deep4.STAGE_MODES] * 4,
                "spatial_resolutions": [56, 28, 14, 7],
                "descriptor_dim": deep4.DESCRIPTOR_DIM,
            },
            "head": {
                "main": "Fusion768-384-256",
                "affine_auxiliary_weight": deep4.AFFINE_AUXILIARY_WEIGHT,
                "lrq": False,
            },
            "ema": {
                "decay": EMA_DECAY,
                "update": "after every completed optimizer step",
                "validation": "EMA parameters",
                "training_forward_backward": "live parameters, unchanged",
                "checkpoint": "live parameters plus persistent EMA shadows and update count",
            },
        }
    }
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "The exact A2D-Deep4 architecture, optimizer, objective, and live-parameter "
            "training path, augmented only by decay-0.9999 parameter EMA updates after "
            "completed optimizer steps. Every epoch and the final metric evaluate EMA "
            "weights; live weights are restored before training resumes."
        )
    }
    payload["source_sha256"]["a2d_deep4_ema_runner"] = deep4.baseline.heads.harness._digest(
        Path(__file__)
    )
    return json.loads(json.dumps(payload))


def main() -> None:
    source = resaux_base
    residuals = a2d_base.residuals
    harness = source.heads.harness
    source.heads.VARIANTS = (VARIANT,)
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=(VARIANT,),
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=_train_epoch,
            evaluate=_evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()

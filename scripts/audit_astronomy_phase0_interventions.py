from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from lnet.astronomy.phase0 import Phase0RunConfig, build_model
from lnet.astronomy.plasticc import (
    LightCurveBatch,
    PlasticcDataset,
    collate_light_curves,
    read_light_curves,
    read_phase0_labels,
    stratified_object_split,
)
from lnet.pac_learned_pole_ledger import (
    balanced_accuracy,
    decompose_margin,
    neutralize_poles,
    original_class_margin,
    prediction_agreement,
    rank_correlation,
    retain_poles,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lnet.alphabet import Alphabet


def _move(batch: LightCurveBatch) -> LightCurveBatch:
    return LightCurveBatch(
        flux=batch.flux.cuda(non_blocking=True),
        time_delta=batch.time_delta.cuda(non_blocking=True),
        observation_mask=batch.observation_mask.cuda(non_blocking=True),
        valid_mask=batch.valid_mask.cuda(non_blocking=True),
        target=batch.target.cuda(non_blocking=True),
        object_id=batch.object_id.cuda(non_blocking=True),
    )


def _capture(
    model: Alphabet,
    dataset: PlasticcDataset,
    *,
    batch_size: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    descriptors: list[Tensor] = []
    logits: list[Tensor] = []
    labels: list[Tensor] = []
    captured: list[Tensor] = []

    def hook(_module: nn.Module, arguments: tuple[object, ...]) -> None:
        if len(arguments) != 2 or not all(
            isinstance(argument, Tensor) for argument in arguments
        ):
            message = "ALPHABET head did not receive writer and reader tensors"
            raise RuntimeError(message)
        writer, reader = cast("tuple[Tensor, Tensor]", arguments)
        captured.append(torch.cat((writer, reader), dim=-1).detach().cpu())

    handle = model.head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                examples = [
                    dataset[index]
                    for index in range(start, min(start + batch_size, len(dataset)))
                ]
                batch = collate_light_curves(examples)
                gpu_batch = _move(batch)
                captured.clear()
                output = model(
                    gpu_batch.flux,
                    time_delta=gpu_batch.time_delta,
                    observation_mask=gpu_batch.observation_mask,
                    valid_mask=gpu_batch.valid_mask,
                )
                if len(captured) != 1:
                    message = f"expected one descriptor capture, observed {len(captured)}"
                    raise RuntimeError(message)
                descriptors.append(captured[0])
                logits.append(output.detach().cpu())
                labels.append(batch.target)
    finally:
        handle.remove()
    return torch.cat(descriptors), torch.cat(logits), torch.cat(labels)


def _intervention_metrics(
    descriptors: Tensor,
    reference: Tensor,
    weight: Tensor,
    bias: Tensor,
    labels: Tensor,
    original_logits: Tensor,
    modes: int,
    selected: Sequence[int],
    *,
    retain: bool,
) -> dict[str, float]:
    changed = (
        retain_poles(descriptors, reference, modes, selected)
        if retain
        else neutralize_poles(descriptors, reference, modes, selected)
    )
    logits = torch.nn.functional.linear(changed, weight, bias)
    original_class = original_logits.argmax(dim=-1)
    old_margin = original_class_margin(original_logits, original_class)
    new_margin = original_class_margin(logits, original_class)
    return {
        "balanced_accuracy": balanced_accuracy(logits, labels),
        "prediction_agreement": prediction_agreement(original_logits, logits),
        "mean_margin_drop": float((old_margin - new_margin).mean()),
        "flip_rate": float((logits.argmax(-1) != original_class).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--random-draws", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        message = "pole intervention audit requires a CUDA host"
        raise RuntimeError(message)

    labels = read_phase0_labels(args.data_dir / "plasticc_train_metadata.csv.gz")
    curves = read_light_curves(args.data_dir / "plasticc_train_lightcurves.csv.gz", labels)
    split = stratified_object_split(labels, seed=20260729)
    train = PlasticcDataset(curves, labels, split.train)
    test = PlasticcDataset(curves, labels, split.test)
    model = cast(
        "Alphabet",
        build_model(
            Phase0RunConfig(model="alphabet", seed=args.seed),
            max(curve.flux.shape[0] for curve in curves.values()),
        ),
    ).cuda()
    checkpoint = args.results_dir / f"alphabet-seed{args.seed}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    model.eval()

    train_descriptors, train_logits, _train_labels = _capture(model, train)
    test_descriptors, test_logits, test_labels = _capture(model, test)
    weight = model.head.classifier.weight.detach().cpu()
    bias_parameter = model.head.classifier.bias
    if bias_parameter is None:
        message = "ALPHABET affine head must have a bias"
        raise RuntimeError(message)
    bias = bias_parameter.detach().cpu()
    reference = train_descriptors.mean(dim=0)
    train_decomposition = decompose_margin(
        train_descriptors,
        train_logits,
        weight,
        bias,
        reference,
        model.modes,
    )
    test_decomposition = decompose_margin(
        test_descriptors,
        test_logits,
        weight,
        bias,
        reference,
        model.modes,
    )
    train_importance = train_decomposition.contributions.abs().mean(dim=0).flatten()
    test_importance = test_decomposition.contributions.abs().mean(dim=0).flatten()
    order = torch.argsort(train_importance, descending=True).tolist()
    bottom = list(reversed(order))
    rows: list[dict[str, object]] = []
    for k in (1, 2, 4, 8, 16, 32):
        if k > 2 * model.modes:
            continue
        for strategy, selected in (("top", order[:k]), ("bottom", bottom[:k])):
            for intervention, retain in (("neutralize", False), ("retain", True)):
                rows.append(
                    {
                        "k": k,
                        "strategy": strategy,
                        "intervention": intervention,
                        **_intervention_metrics(
                            test_descriptors,
                            reference,
                            weight,
                            bias,
                            test_labels,
                            test_logits,
                            model.modes,
                            selected,
                            retain=retain,
                        ),
                    }
                )
        generator = random.Random(args.seed * 1009 + k)  # noqa: S311
        random_values: dict[tuple[str, str], list[dict[str, float]]] = {}
        for _ in range(args.random_draws):
            selected = generator.sample(range(2 * model.modes), k)
            for intervention, retain in (("neutralize", False), ("retain", True)):
                random_values.setdefault(("random", intervention), []).append(
                    _intervention_metrics(
                        test_descriptors,
                        reference,
                        weight,
                        bias,
                        test_labels,
                        test_logits,
                        model.modes,
                        selected,
                        retain=retain,
                    )
                )
        for (strategy, intervention), values in random_values.items():
            rows.append(
                {
                    "k": k,
                    "strategy": strategy,
                    "intervention": intervention,
                    "draws": args.random_draws,
                    **{
                        key: mean(value[key] for value in values)
                        for key in values[0]
                    },
                }
            )
    reconstructed = torch.nn.functional.linear(test_descriptors, weight, bias)
    payload = {
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "selection_split": "train only",
        "evaluation_split": "held-out phase0 test",
        "train_objects": len(train),
        "test_objects": len(test),
        "full_test_balanced_accuracy": balanced_accuracy(test_logits, test_labels),
        "max_logit_reconstruction_residual": float(
            (reconstructed - test_logits).abs().max()
        ),
        "max_margin_decomposition_residual": max(
            train_decomposition.max_residual,
            test_decomposition.max_residual,
        ),
        "train_test_importance_rank_correlation": rank_correlation(
            train_importance,
            test_importance,
        ),
        "pole_order_by_train_importance": order,
        "interventions": rows,
    }
    output = args.results_dir / f"pole-interventions-seed{args.seed}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

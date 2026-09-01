#!/usr/bin/env python3
"""Compare predictive tensor organization in ALPHABET and Mamba-1 states.

The diagnostic uses disjoint validation sequences to fit and evaluate three
linear maps from a recurrent state tensor to the model's own final hidden:

* temporal-axis collapse, retaining content coordinates;
* content-axis collapse, retaining temporal coordinates;
* a low-output-rank map from the fully flattened state.

The full map is unfolded along both declared axes to measure whether its
predictive directions are approximately separable.  Reported next-token losses
are obtained by feeding probe-predicted hiddens through each model's frozen tied
LM head; the language models themselves are never updated.
"""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
import argparse
import json
import random
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import (
    LaplaceMambaLMConfig,
    VectorImagePostFusionAlphabet2LM,
)
from lnet.alphabet_lm_data import TokenBlockDataset
from lnet.alphabet_lm_mamba import MambaLM, MambaLMConfig
from lnet.alphabet_lm_tensor_probe import (
    TensorAxisProbe,
    axis_spectrum_metrics,
)


def _checkpoint_model(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError(f"checkpoint has no model state: {path}")
    return cast("dict[str, Tensor]", payload["model"])


def _positions(labels: Tensor, pad_id: int, count: int) -> Tensor:
    valid = torch.nonzero(labels != pad_id, as_tuple=False).flatten()
    if not valid.numel():
        raise RuntimeError("diagnostic sequence contains no non-padding labels")
    if valid.numel() <= count:
        return valid
    selection = torch.linspace(0, valid.numel() - 1, count).round().long()
    return valid[selection].unique(sorted=True)


@torch.no_grad()
def _collect_alphabet(
    model: VectorImagePostFusionAlphabet2LM,
    dataset: TokenBlockDataset,
    indices: range,
    layers: tuple[int, ...],
    positions_per_sequence: int,
    device: torch.device,
) -> tuple[dict[int, Tensor], Tensor, Tensor]:
    captured: dict[int, tuple[Tensor, Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer in layers:
        memory = cast("nn.Module", model.blocks[layer].memory)

        def hook(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: tuple[Tensor, Tensor],
            *,
            active_layer: int = layer,
        ) -> None:
            captured[active_layer] = output

        handles.append(memory.register_forward_hook(hook))
    states: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    targets: list[Tensor] = []
    labels_out: list[Tensor] = []
    try:
        for index in indices:
            tokens = dataset[index]
            inputs = tokens[:-1].unsqueeze(0).to(device)
            labels = tokens[1:]
            positions = _positions(labels, dataset.manifest.pad_id, positions_per_sequence)
            captured.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model.hidden(inputs)
            if set(captured) != set(layers):
                raise RuntimeError("not every requested ALPHABET memory layer was captured")
            positions_cuda = positions.to(device)
            targets.append(hidden[0, positions_cuda].detach().to("cpu", torch.float16))
            labels_out.append(labels[positions].to(torch.long))
            for layer in layers:
                state_real, state_imag = captured[layer]
                state = torch.cat((state_real, state_imag), dim=-1)
                states[layer].append(
                    state[0, positions_cuda].detach().to("cpu", torch.float16)
                )
    finally:
        for handle in handles:
            handle.remove()
    return (
        {layer: torch.cat(values) for layer, values in states.items()},
        torch.cat(targets),
        torch.cat(labels_out),
    )


@torch.no_grad()
def _mamba_states_from_projection(
    mixer: Any,
    projected_input: Tensor,
    projected_parameters: Tensor,
    positions: Tensor,
) -> Tensor:
    """Reconstruct official Mamba-1 states after the selected token updates."""

    batch = 1
    steps, inner = projected_input.shape
    state_width = int(mixer.d_state)
    dt_rank = int(mixer.dt_rank)
    dt_raw, variable_b, _variable_c = projected_parameters.split(
        (dt_rank, state_width, state_width), dim=-1
    )
    delta = functional.softplus(
        functional.linear(
            dt_raw.float(),
            mixer.dt_proj.weight.float(),
            mixer.dt_proj.bias.float(),
        )
    )
    transition = -torch.exp(mixer.A_log.float())
    active = torch.zeros((batch, inner, state_width), device=delta.device)
    position_list = positions.tolist()
    cursor = 0
    selected: list[Tensor] = []
    x = projected_input.float()
    b = variable_b.float()
    for step in range(steps):
        step_delta = delta[step]
        decay = torch.exp(step_delta[:, None] * transition)
        forcing = step_delta[:, None] * b[step][None, :] * x[step][:, None]
        active.mul_(decay).add_(forcing)
        if cursor < len(position_list) and step == position_list[cursor]:
            selected.append(active[0].t().clone())
            cursor += 1
    if cursor != len(position_list):
        raise RuntimeError("failed to collect every requested Mamba token state")
    return torch.stack(selected)


@torch.no_grad()
def _collect_mamba(
    model: MambaLM,
    dataset: TokenBlockDataset,
    indices: range,
    layers: tuple[int, ...],
    positions_per_sequence: int,
    device: torch.device,
) -> tuple[dict[int, Tensor], Tensor, Tensor]:
    official_model = cast("Any", model.model)
    backbone = official_model.backbone
    mixers: dict[int, Any] = {
        layer: backbone.layers[layer].mixer for layer in layers
    }
    captured: dict[int, tuple[Tensor, Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    original_fast_path = {layer: bool(mixer.use_fast_path) for layer, mixer in mixers.items()}
    for layer, mixer in mixers.items():
        mixer.use_fast_path = False

        def hook(
            _module: nn.Module,
            inputs: tuple[Tensor, ...],
            output: Tensor,
            *,
            active_layer: int = layer,
        ) -> None:
            captured[active_layer] = (inputs[0].detach(), output.detach())

        handles.append(mixer.x_proj.register_forward_hook(hook))
    states: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    targets: list[Tensor] = []
    labels_out: list[Tensor] = []
    try:
        for index in indices:
            tokens = dataset[index]
            inputs = tokens[:-1].unsqueeze(0).to(device)
            labels = tokens[1:]
            positions = _positions(labels, dataset.manifest.pad_id, positions_per_sequence)
            captured.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = backbone(inputs)
            if set(captured) != set(layers):
                raise RuntimeError("not every requested Mamba mixer layer was captured")
            positions_cuda = positions.to(device)
            targets.append(hidden[0, positions_cuda].detach().to("cpu", torch.float16))
            labels_out.append(labels[positions].to(torch.long))
            for layer, mixer in mixers.items():
                projected_input, projected_parameters = captured[layer]
                state = _mamba_states_from_projection(
                    mixer,
                    projected_input,
                    projected_parameters,
                    positions_cuda,
                )
                states[layer].append(state.to("cpu", torch.float16))
    finally:
        for layer, mixer in mixers.items():
            mixer.use_fast_path = original_fast_path[layer]
        for handle in handles:
            handle.remove()
    return (
        {layer: torch.cat(values) for layer, values in states.items()},
        torch.cat(targets),
        torch.cat(labels_out),
    )


@torch.no_grad()
def _prediction_metrics(
    predictions: Tensor,
    targets: Tensor,
    labels: Tensor,
    head_weight: Tensor,
    *,
    batch_size: int = 128,
) -> dict[str, float]:
    squared_error = 0.0
    squared_target = 0.0
    cosine_sum = 0.0
    loss_sum = 0.0
    correct = 0
    count = predictions.shape[0]
    target_mean = targets.float().mean(dim=0)
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        prediction = predictions[start:stop].float()
        target = targets[start:stop].float()
        active_labels = labels[start:stop]
        squared_error += float((prediction - target).square().sum().item())
        squared_target += float((target - target_mean).square().sum().item())
        cosine_sum += float(
            functional.cosine_similarity(prediction, target, dim=-1).sum().item()
        )
        logits = functional.linear(prediction, head_weight.float())
        loss_sum += float(
            functional.cross_entropy(logits, active_labels, reduction="sum").item()
        )
        correct += int((logits.argmax(dim=-1) == active_labels).sum().item())
    return {
        "hidden_mse": squared_error / (count * targets.shape[-1]),
        "hidden_r2": 1.0 - squared_error / max(squared_target, 1.0e-30),
        "hidden_cosine": cosine_sum / count,
        "next_token_loss": loss_sum / count,
        "next_token_accuracy": correct / count,
    }


def _fit_probe(
    train_states: Tensor,
    train_targets: Tensor,
    eval_states: Tensor,
    eval_targets: Tensor,
    eval_labels: Tensor,
    head_weight: Tensor,
    *,
    mode: str,
    probe_rank: int,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[TensorAxisProbe, dict[str, float]]:
    temporal, content = train_states.shape[1:]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = train_states.float().square().mean().sqrt().clamp_min(1.0e-6)
    probe = TensorAxisProbe(
        temporal,
        content,
        train_targets.shape[-1],
        mode=cast("Any", mode),
        probe_rank=probe_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    probe.train()
    for _step in range(steps):
        indices = torch.randint(
            train_states.shape[0],
            (min(batch_size, train_states.shape[0]),),
            generator=generator,
        )
        states = train_states[indices].to(device, dtype=torch.float32) / scale.to(device)
        targets = train_targets[indices].to(device, dtype=torch.float32)
        prediction = probe(states)
        loss = functional.mse_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
    probe.eval()
    predictions: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, eval_states.shape[0], batch_size):
            active = eval_states[start : start + batch_size].to(device, dtype=torch.float32)
            predictions.append(probe(active / scale.to(device)).cpu())
    metrics = _prediction_metrics(
        torch.cat(predictions),
        eval_targets,
        eval_labels,
        head_weight.cpu(),
    )
    metrics["state_rms"] = float(scale.item())
    metrics["parameters"] = float(sum(parameter.numel() for parameter in probe.parameters()))
    return probe, metrics


@torch.no_grad()
def _rank_one_predictions(
    states: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    axis: int,
    state_rms: float,
) -> Tensor:
    matrix = weight.movedim(axis, 0).flatten(1).float()
    if matrix.shape[0] <= 64:
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        left = u[:, 0]
        right = singular[0] * vh[0]
    else:
        u, singular, v = torch.svd_lowrank(matrix, q=1, niter=6)
        left = u[:, 0]
        right = singular[0] * v[:, 0]
    normalized = states.float() / state_rms
    if axis == 0:
        collapsed = torch.einsum("ntc,t->nc", normalized, left)
        output_weight = right.reshape(weight.shape[1], weight.shape[2])
    else:
        collapsed = torch.einsum("ntc,c->nt", normalized, left)
        output_weight = right.reshape(weight.shape[0], weight.shape[2])
    return collapsed @ output_weight + bias.float()


def _analyze_model(
    name: str,
    train_states: dict[int, Tensor],
    train_targets: Tensor,
    eval_states: dict[int, Tensor],
    eval_targets: Tensor,
    eval_labels: Tensor,
    head_weight: Tensor,
    layers: tuple[int, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    oracle = _prediction_metrics(
        eval_targets,
        eval_targets,
        eval_labels,
        head_weight.cpu(),
    )
    layer_rows: dict[str, object] = {}
    for layer in layers:
        row: dict[str, object] = {}
        full_probe: TensorAxisProbe | None = None
        for offset, mode in enumerate(("temporal", "content", "full")):
            probe, metrics = _fit_probe(
                train_states[layer],
                train_targets,
                eval_states[layer],
                eval_targets,
                eval_labels,
                head_weight,
                mode=mode,
                probe_rank=args.probe_rank,
                steps=args.probe_steps,
                learning_rate=args.probe_learning_rate,
                batch_size=args.probe_batch_size,
                seed=args.seed + 100 * layer + offset,
                device=device,
            )
            row[mode] = metrics
            if mode == "full":
                full_probe = probe
        if full_probe is None:
            raise AssertionError("full tensor probe was not fitted")
        weight = full_probe.weight_tensor().detach()
        spectra = {
            "temporal": axis_spectrum_metrics(weight, axis=0),
            "content": axis_spectrum_metrics(weight, axis=1),
        }
        full_metrics = cast("dict[str, float]", row["full"])
        rank_one: dict[str, object] = {}
        for axis_name, axis in (("temporal", 0), ("content", 1)):
            predictions = _rank_one_predictions(
                eval_states[layer].to(device),
                weight,
                full_probe.output.bias.detach(),
                axis=axis,
                state_rms=full_metrics["state_rms"],
            ).cpu()
            rank_one[axis_name] = _prediction_metrics(
                predictions,
                eval_targets,
                eval_labels,
                head_weight.cpu(),
            )
        row["full_weight_spectrum"] = spectra
        row["full_weight_rank1_approximation"] = rank_one
        row["state_shape"] = list(train_states[layer].shape[1:])
        layer_rows[str(layer)] = row
        print(
            "TENSOR_PROBE_PROGRESS="
            + json.dumps(
                {
                    "model": name,
                    "layer": layer,
                    "temporal_loss": cast("dict[str, float]", row["temporal"])[
                        "next_token_loss"
                    ],
                    "content_loss": cast("dict[str, float]", row["content"])[
                        "next_token_loss"
                    ],
                    "full_loss": full_metrics["next_token_loss"],
                    "temporal_top1": spectra["temporal"]["top1_energy"],
                    "content_top1": spectra["content"]["top1_energy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del full_probe, weight
        torch.cuda.empty_cache()
    return {
        "oracle": oracle,
        "train_samples": int(train_targets.shape[0]),
        "eval_samples": int(eval_targets.shape[0]),
        "layers": layer_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet-checkpoint", type=Path, required=True)
    parser.add_argument("--mamba-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,9,18")
    parser.add_argument("--train-sequences", type=int, default=8)
    parser.add_argument("--eval-sequences", type=int, default=4)
    parser.add_argument("--positions-per-sequence", type=int, default=256)
    parser.add_argument("--probe-rank", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--probe-learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=501)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("state tensor diagnostic requires CUDA")
    layers = tuple(int(value) for value in args.layers.split(","))
    if not layers or min(layers) < 0 or max(layers) >= 19:
        raise ValueError("diagnostic layers must lie in [0, 18]")
    if args.train_sequences <= 0 or args.eval_sequences <= 0:
        raise ValueError("probe train/eval sequence counts must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dataset = TokenBlockDataset(args.validation_manifest)
    sequence_count = args.train_sequences + args.eval_sequences
    if sequence_count > len(dataset):
        raise ValueError("diagnostic requests more sequences than validation contains")
    train_indices = range(args.train_sequences)
    eval_indices = range(args.train_sequences, sequence_count)

    alphabet = VectorImagePostFusionAlphabet2LM(
        LaplaceMambaLMConfig(
            vocab_size=dataset.manifest.vocab_size,
            layers=19,
            pole_modes=32,
            state_size=4,
            head_width=16,
            conv_width=3,
        )
    ).to(device)
    alphabet.load_state_dict(_checkpoint_model(args.alphabet_checkpoint))
    alphabet.eval()
    alphabet_head = torch.cat(
        (alphabet.embedding_real.weight, alphabet.embedding_imag.weight), dim=-1
    ).detach()
    alphabet_train = _collect_alphabet(
        alphabet,
        dataset,
        train_indices,
        layers,
        args.positions_per_sequence,
        device,
    )
    alphabet_eval = _collect_alphabet(
        alphabet,
        dataset,
        eval_indices,
        layers,
        args.positions_per_sequence,
        device,
    )
    del alphabet
    torch.cuda.empty_cache()

    mamba = MambaLM(
        MambaLMConfig(
            vocab_size=dataset.manifest.vocab_size,
            model_width=512,
            layers=19,
            state_size=16,
            conv_width=4,
            expand=2,
            architecture="Mamba1",
        )
    ).to(device)
    mamba.load_state_dict(_checkpoint_model(args.mamba_checkpoint))
    mamba.eval()
    mamba_head = cast("Any", mamba.model).lm_head.weight.detach()
    mamba_train = _collect_mamba(
        mamba,
        dataset,
        train_indices,
        layers,
        args.positions_per_sequence,
        device,
    )
    mamba_eval = _collect_mamba(
        mamba,
        dataset,
        eval_indices,
        layers,
        args.positions_per_sequence,
        device,
    )
    del mamba
    torch.cuda.empty_cache()

    result = {
        "schema": "lnet.alphabet_lm.state_tensor_probe.v1",
        "protocol": {
            "layers": list(layers),
            "train_sequences": args.train_sequences,
            "eval_sequences": args.eval_sequences,
            "positions_per_sequence": args.positions_per_sequence,
            "probe_rank": args.probe_rank,
            "probe_steps": args.probe_steps,
            "probe_learning_rate": args.probe_learning_rate,
            "probe_batch_size": args.probe_batch_size,
            "seed": args.seed,
            "target": "model-specific final hidden, evaluated through frozen tied LM head",
            "split": "disjoint validation sequences for probe fitting and evaluation",
        },
        "alphabet": _analyze_model(
            "alphabet",
            alphabet_train[0],
            alphabet_train[1],
            alphabet_eval[0],
            alphabet_eval[1],
            alphabet_eval[2],
            alphabet_head,
            layers,
            args,
            device,
        ),
        "mamba": _analyze_model(
            "mamba",
            mamba_train[0],
            mamba_train[1],
            mamba_eval[0],
            mamba_eval[1],
            mamba_eval[2],
            mamba_head,
            layers,
            args,
            device,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"TENSOR_PROBE_COMPLETE={args.output}", flush=True)


if __name__ == "__main__":
    main()

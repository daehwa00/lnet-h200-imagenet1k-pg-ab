"""Fail-closed runtime preflight for three-stage causal LogSig V5."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import argparse
import json
import sys

import torch

from optimization.three_stage_causal_log_signature import (
    FullyPoleNativeThreeStageCausalALPHABET,
)


def _finite_model(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _training_case(
    device: torch.device,
    *,
    length: int,
    metadata: bool,
) -> dict[str, object]:
    torch.manual_seed(9501 + length + int(metadata))
    model = FullyPoleNativeThreeStageCausalALPHABET(1, 32, 16, 5).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    values = torch.randn(3, length, 1, device=device, requires_grad=True)
    kwargs: dict[str, torch.Tensor] = {}
    if metadata:
        delta = 0.05 + torch.rand(3, length, 1, device=device)
        observation = torch.ones_like(delta)
        observation[:, 5::11] = 0.0
        valid = torch.ones_like(delta)
        valid[1, -3:] = 0.0
        kwargs = {
            "time_delta": delta,
            "observation_mask": observation,
            "valid_mask": valid,
        }
    output = model(values, **kwargs)
    loss = output.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    if (
        output.shape != (3, 5)
        or not torch.isfinite(output).all()
        or not torch.isfinite(loss)
        or values.grad is None
        or not torch.isfinite(values.grad).all()
        or any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients)
    ):
        raise RuntimeError(f"non-finite LogSig V5 training case: N={length}, metadata={metadata}")
    optimizer.step()
    if not _finite_model(model):
        raise RuntimeError(f"non-finite LogSig V5 optimizer state: N={length}")
    expected_reader = (
        "triton_fused_training" if device.type == "cuda" and length >= 32 else "materialized_bmm"
    )
    expected_writer = (
        "triton_fused_state_training"
        if device.type == "cuda" and length >= 32
        else "materialized_bmm"
    )
    writer_dispatch = model.last_writer_log_signature_dispatch()  # pyright: ignore[reportCallIssue]
    reader_dispatch = model.last_log_signature_dispatch()  # pyright: ignore[reportCallIssue]
    if writer_dispatch != expected_writer or reader_dispatch != expected_reader:
        message = " ".join(
            (
                f"unexpected V5 dispatch for N={length}: writer={writer_dispatch};",
                f"reader={reader_dispatch}; expected={expected_writer}/{expected_reader}",
            )
        )
        raise RuntimeError(message)
    return {
        "length": length,
        "metadata": metadata,
        "writer_dispatch": writer_dispatch,
        "reader_dispatch": reader_dispatch,
        "loss": float(loss.detach().cpu()),
    }


def _causal_local_case(device: torch.device) -> dict[str, object]:
    torch.manual_seed(9513)
    model = FullyPoleNativeThreeStageCausalALPHABET(1, 32, 16, 5).to(device).eval()
    left = torch.randn(2, 25, 1, device=device)
    right = left.clone()
    right[:, 14:] = torch.randn_like(right[:, 14:])
    with torch.inference_mode():
        left_input = model.stem(left)
        right_input = model.stem(right)
        left_writer = model.writer_local(left_input)
        right_writer = model.writer_local(right_input)
        left_reader = model.reader_local(left_writer)
        right_reader = model.reader_local(right_writer)
    errors = {
        "input": float((left_input[:, :14] - right_input[:, :14]).abs().max().cpu()),
        "writer": float((left_writer[:, :14] - right_writer[:, :14]).abs().max().cpu()),
        "reader": float((left_reader[:, :14] - right_reader[:, :14]).abs().max().cpu()),
    }
    if any(error != 0.0 for error in errors.values()) or left_reader.shape[1] != left.shape[1]:
        raise RuntimeError(f"V5 local path is not exactly causal/node-aligned: {errors}")
    return {"prefix_max_abs_error": errors, "node_length_preserved": True}


def _inference_case(device: torch.device) -> dict[str, object]:
    torch.manual_seed(9521)
    model = FullyPoleNativeThreeStageCausalALPHABET(1, 32, 16, 5).to(device).eval()
    model.prepare_for_inference_(validate_metadata=False)
    values = torch.randn(2, 65, 1, device=device)
    with torch.inference_mode():
        output = model(values)
    if output.shape != (2, 5) or not torch.isfinite(output).all():
        raise RuntimeError("non-finite LogSig V5 inference output")
    expected_writer = "triton_fused_stateful" if device.type == "cuda" else "materialized_bmm"
    expected_reader = "triton_fused" if device.type == "cuda" else "materialized_bmm"
    writer_dispatch = model.last_writer_log_signature_dispatch()
    reader_dispatch = model.last_log_signature_dispatch()
    if writer_dispatch != expected_writer or reader_dispatch != expected_reader:
        message = " ".join(
            (
                f"unexpected LogSig V5 inference dispatch: writer={writer_dispatch};",
                f"reader={reader_dispatch}",
            )
        )
        raise RuntimeError(message)
    return {"writer_dispatch": writer_dispatch, "reader_dispatch": reader_dispatch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    device = torch.device(args.device)
    payload = {
        "schema": "pac_logsig_v5_preflight.v1",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "causal_local_path": _causal_local_case(device),
        "training": [
            _training_case(device, length=17, metadata=False),
            _training_case(device, length=65, metadata=False),
            _training_case(device, length=65, metadata=True),
        ],
        "inference": _inference_case(device),
        "passed": True,
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

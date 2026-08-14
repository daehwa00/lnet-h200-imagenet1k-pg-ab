"""Opt-in fused FP32 clip-and-AdamW tail for compact PAC models.

PyTorch's campaign path launches per-tensor norm, gradient-scaling,
step-increment, and fused AdamW kernels.  This research runtime keeps the same
per-tensor-L2 then global-L2 structure, but performs the tail with one norm
kernel and one pointer-table AdamW kernel.  The CUDA implementation specializes
the canonical 18- and 24-tensor layouts while retaining a bounded dynamic path
for other compact PAC variants.  EFP16's signed QR stem projection deliberately
remains outside this module.
"""

from __future__ import annotations

import os
import sysconfig
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn

_MAX_TENSOR_COUNT = 64


class _FusedOptimizerExtension(Protocol):
    def fused_clip_adamw_step(
        self,
        pointers: int,
        numels: int,
        norms: int,
        total_norm: int,
        clip_coefficient: int,
        step_size: int,
        bias_correction2_sqrt: int,
        stream: int,
        tensor_count: int,
        max_norm: float,
        learning_rate: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        epsilon: float,
    ) -> None: ...


@lru_cache(maxsize=1)
def _load_fused_optimizer_extension() -> _FusedOptimizerExtension:
    cuda_version = torch.version.cuda
    packaged_cuda_home = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    use_packaged_cuda = (
        cuda_version is not None
        and int(cuda_version.split(".")[0]) >= 13
        and (packaged_cuda_home / "bin" / "nvcc").is_file()
    )
    previous_cuda_home = os.environ.get("CUDA_HOME")
    if use_packaged_cuda:
        os.environ["CUDA_HOME"] = str(packaged_cuda_home)
    try:
        from torch.utils.cpp_extension import load  # noqa: PLC0415

        module = load(
            name="pac_cuda_fused_optimizer_v6",
            sources=[
                str(Path(__file__).resolve().parents[2] / "csrc" / "pac_cuda_fused_optimizer.cu")
            ],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    finally:
        if use_packaged_cuda:
            if previous_cuda_home is None:
                os.environ.pop("CUDA_HOME", None)
            else:
                os.environ["CUDA_HOME"] = previous_cuda_home
    return cast("_FusedOptimizerExtension", cast("object", module))


class FusedClipAdamW:
    """Own or borrow state for PAC's bounded dynamic-tensor CUDA tail."""

    def __init__(
        self,
        parameters: tuple[nn.Parameter, ...],
        *,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1.0e-4,
        epsilon: float = 1.0e-8,
        max_norm: float = 1.0,
        exp_avgs: tuple[Tensor, ...] | None = None,
        exp_avg_sqs: tuple[Tensor, ...] | None = None,
        state_steps: tuple[Tensor, ...] | None = None,
    ) -> None:
        tensor_count = len(parameters)
        if tensor_count < 1 or tensor_count > _MAX_TENSOR_COUNT:
            message = (
                f"fused PAC AdamW requires between 1 and {_MAX_TENSOR_COUNT} parameter tensors"
            )
            raise ValueError(message)
        device = parameters[0].device
        if device.type != "cuda":
            message = "fused PAC AdamW requires CUDA parameters"
            raise ValueError(message)
        if any(
            parameter.device != device
            or parameter.dtype != torch.float32
            or not parameter.is_contiguous()
            or parameter.grad is None
            or parameter.grad.dtype != torch.float32
            or not parameter.grad.is_contiguous()
            for parameter in parameters
        ):
            message = "fused PAC AdamW requires contiguous FP32 parameters and persistent gradients"
            raise ValueError(message)
        if learning_rate <= 0.0 or weight_decay < 0.0 or epsilon <= 0.0 or max_norm <= 0.0:
            message = "fused PAC AdamW hyperparameters are outside their supported domain"
            raise ValueError(message)
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            message = "fused PAC AdamW betas must be in [0, 1)"
            raise ValueError(message)

        self.parameters = parameters
        self.tensor_count = tensor_count
        self.device = device
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.weight_decay = float(weight_decay)
        self.epsilon = float(epsilon)
        self.max_norm = float(max_norm)
        self.exp_avgs = (
            tuple(torch.zeros_like(parameter) for parameter in parameters)
            if exp_avgs is None
            else exp_avgs
        )
        self.exp_avg_sqs = (
            tuple(torch.zeros_like(parameter) for parameter in parameters)
            if exp_avg_sqs is None
            else exp_avg_sqs
        )
        self.state_steps = (
            tuple(torch.zeros((), device=device, dtype=torch.float32) for _ in parameters)
            if state_steps is None
            else state_steps
        )
        self._validate_state()
        gradients = tuple(cast("Tensor", parameter.grad) for parameter in parameters)
        pointer_rows = (parameters, gradients, self.exp_avgs, self.exp_avg_sqs, self.state_steps)
        self._expected_pointers = tuple(tensor.data_ptr() for row in pointer_rows for tensor in row)
        self._pointers = torch.tensor(
            self._expected_pointers,
            device=device,
            dtype=torch.int64,
        )
        self._numels = torch.tensor(
            tuple(parameter.numel() for parameter in parameters),
            device=device,
            dtype=torch.int64,
        )
        self.per_tensor_norms = torch.empty(
            tensor_count,
            device=device,
            dtype=torch.float32,
        )
        self.total_norm = torch.empty((), device=device, dtype=torch.float32)
        self.clip_coefficient = torch.empty((), device=device, dtype=torch.float32)
        self.step_size = torch.empty((), device=device, dtype=torch.float64)
        self.bias_correction2_sqrt = torch.empty(
            (),
            device=device,
            dtype=torch.float64,
        )
        self._extension = _load_fused_optimizer_extension()

    @classmethod
    def from_adamw(
        cls,
        optimizer: torch.optim.AdamW,
        *,
        max_norm: float,
    ) -> FusedClipAdamW:
        """Borrow a materialized canonical fused AdamW state."""
        if len(optimizer.param_groups) != 1:
            message = "fused PAC AdamW requires exactly one optimizer parameter group"
            raise ValueError(message)
        group = optimizer.param_groups[0]
        if (
            group.get("amsgrad") is True
            or group.get("maximize") is True
            or group.get("differentiable") is True
        ):
            message = "fused PAC AdamW supports canonical non-AMSGrad minimization only"
            raise ValueError(message)
        parameters = tuple(cast("nn.Parameter", parameter) for parameter in group["params"])
        try:
            exp_avgs = tuple(cast("Tensor", optimizer.state[p]["exp_avg"]) for p in parameters)
            exp_avg_sqs = tuple(
                cast("Tensor", optimizer.state[p]["exp_avg_sq"]) for p in parameters
            )
            state_steps = tuple(cast("Tensor", optimizer.state[p]["step"]) for p in parameters)
        except KeyError as error:
            message = "AdamW state must be materialized before installing the fused PAC tail"
            raise ValueError(message) from error
        learning_rate = group["lr"]
        if isinstance(learning_rate, Tensor):
            message = "fused PAC AdamW requires a scalar Python learning rate"
            raise TypeError(message)
        betas = cast("tuple[float, float]", group["betas"])
        return cls(
            parameters,
            learning_rate=float(learning_rate),
            betas=betas,
            weight_decay=float(group["weight_decay"]),
            epsilon=float(group["eps"]),
            max_norm=max_norm,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            state_steps=state_steps,
        )

    def validate_addresses(self) -> None:
        """Reject pointer changes before capture or replay setup."""
        gradients = tuple(cast("Tensor", parameter.grad) for parameter in self.parameters)
        rows = (
            self.parameters,
            gradients,
            self.exp_avgs,
            self.exp_avg_sqs,
            self.state_steps,
        )
        actual = tuple(tensor.data_ptr() for row in rows for tensor in row)
        if actual != self._expected_pointers:
            message = "fused PAC AdamW tensor addresses changed after preparation"
            raise RuntimeError(message)

    def step(self) -> Tensor:
        """Launch the two borrowed-state CUDA kernels and return total norm."""
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self._extension.fused_clip_adamw_step(
            self._pointers.data_ptr(),
            self._numels.data_ptr(),
            self.per_tensor_norms.data_ptr(),
            self.total_norm.data_ptr(),
            self.clip_coefficient.data_ptr(),
            self.step_size.data_ptr(),
            self.bias_correction2_sqrt.data_ptr(),
            stream,
            self.tensor_count,
            self.max_norm,
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.weight_decay,
            self.epsilon,
        )
        return self.total_norm

    def _validate_state(self) -> None:
        state_groups = (self.exp_avgs, self.exp_avg_sqs, self.state_steps)
        if any(len(group) != self.tensor_count for group in state_groups):
            message = "fused PAC AdamW state must contain one entry per parameter"
            raise ValueError(message)
        if any(
            value.device != self.device or value.dtype != torch.float32 or not value.is_contiguous()
            for group in state_groups
            for value in group
        ):
            message = "fused PAC AdamW state must be contiguous FP32 CUDA tensors"
            raise ValueError(message)
        if any(
            average.shape != parameter.shape or square.shape != parameter.shape
            for parameter, average, square in zip(
                self.parameters,
                self.exp_avgs,
                self.exp_avg_sqs,
                strict=True,
            )
        ):
            message = "fused PAC AdamW moment shapes must match their parameters"
            raise ValueError(message)
        if any(step.shape != () for step in self.state_steps):
            message = "fused PAC AdamW state steps must be scalar tensors"
            raise ValueError(message)
        steps = torch.stack(self.state_steps)
        if not torch.equal(steps, steps[:1].expand_as(steps)):
            message = "fused PAC AdamW requires lockstep parameter state steps"
            raise ValueError(message)


__all__ = ["FusedClipAdamW"]

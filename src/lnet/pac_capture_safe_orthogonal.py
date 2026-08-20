"""CUDA-Graph-safe replacement for PyTorch's matrix-exp orthogonal map.

PyTorch's orthogonal parametrization uses ``torch.matrix_exp``.  That is a
good general-purpose implementation, but its CUDA implementation is not a
stable capture boundary for a whole PAC training step.  PAC only needs small
FP32 matrices whose tangent coordinates stay close to the origin.  This
module keeps PyTorch's dynamic-trivialization representation exactly as-is
and replaces only the exponential evaluation with a fixed computation graph.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

DEFAULT_TAYLOR_DEGREE: Final = 12
DEFAULT_SCALING_STEPS: Final = 3


def fixed_taylor_matrix_exp(
    matrix: Tensor,
    *,
    taylor_degree: int = DEFAULT_TAYLOR_DEGREE,
    scaling_steps: int = DEFAULT_SCALING_STEPS,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """Approximate ``exp(matrix)`` with a fixed capture-safe FP32 graph.

    The Taylor polynomial is evaluated in Horner form after scaling by
    ``2**scaling_steps`` and the result is squared ``scaling_steps`` times.
    Both loop counts are Python constants for a prepared parametrization, so
    CUDA Graph capture and ``torch.compile(fullgraph=True)`` see no data-
    dependent control flow.
    """

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        message = f"expected one square matrix, got shape={tuple(matrix.shape)}"
        raise ValueError(message)
    if not matrix.is_floating_point():
        message = f"expected floating-point matrix, got dtype={matrix.dtype}"
        raise TypeError(message)
    if taylor_degree < 1:
        message = f"taylor_degree must be positive, got {taylor_degree}"
        raise ValueError(message)
    if scaling_steps < 0:
        message = f"scaling_steps must be non-negative, got {scaling_steps}"
        raise ValueError(message)
    if compute_dtype not in (None, torch.float32, torch.float64):
        message = f"compute_dtype must be float32 or float64, got {compute_dtype}"
        raise ValueError(message)

    active = matrix if compute_dtype is None else matrix.to(dtype=compute_dtype)
    identity = torch.eye(active.shape[0], dtype=active.dtype, device=active.device)
    scaled = active * (2.0**-scaling_steps)
    result = identity
    for denominator in range(taylor_degree, 0, -1):
        result = identity + torch.matmul(scaled, result) / denominator
    for _ in range(scaling_steps):
        result = torch.matmul(result, result)
    return result.to(dtype=matrix.dtype)


class CaptureSafeMatrixExpOrthogonal(nn.Module):
    """Drop-in forward map for PyTorch matrix-exp dynamic trivialization."""

    base: Tensor

    def __init__(
        self,
        shape: torch.Size | tuple[int, ...],
        base: Tensor,
        *,
        taylor_degree: int = DEFAULT_TAYLOR_DEGREE,
        scaling_steps: int = DEFAULT_SCALING_STEPS,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if len(shape) != 2:
            message = f"expected a matrix shape, got {tuple(shape)}"
            raise ValueError(message)
        if taylor_degree < 1:
            message = f"taylor_degree must be positive, got {taylor_degree}"
            raise ValueError(message)
        if scaling_steps < 0:
            message = f"scaling_steps must be non-negative, got {scaling_steps}"
            raise ValueError(message)

        rows, columns = int(shape[-2]), int(shape[-1])
        expected_base_shape = (max(rows, columns), max(rows, columns))
        if tuple(base.shape) != expected_base_shape:
            message = (
                f"expected dynamic-trivialization base shape {expected_base_shape}, "
                f"got {tuple(base.shape)}"
            )
            raise ValueError(message)
        self.shape = torch.Size((rows, columns))
        self.taylor_degree = taylor_degree
        self.scaling_steps = scaling_steps
        self.compute_dtype = compute_dtype
        self.register_buffer("base", base)
        self.base = base

    def forward(self, coordinates: Tensor) -> Tensor:
        rows, columns = coordinates.shape[-2:]
        transposed = rows < columns
        working = coordinates.mT if transposed else coordinates
        rows, columns = working.shape[-2:]

        lower = working.tril()
        if rows != columns:
            padding = lower.new_zeros((rows, rows - columns))
            lower = torch.cat((lower, padding), dim=-1)
        skew = lower - lower.mT
        orthogonal = fixed_taylor_matrix_exp(
            skew,
            taylor_degree=self.taylor_degree,
            scaling_steps=self.scaling_steps,
            compute_dtype=self.compute_dtype,
        )
        if rows != columns:
            orthogonal = orthogonal[:, :columns]
        orthogonal = torch.matmul(self.base, orthogonal)
        return orthogonal.mT if transposed else orthogonal


def replace_matrix_exp_orthogonal_(
    module: nn.Module,
    name: str = "weight",
    *,
    taylor_degree: int = DEFAULT_TAYLOR_DEGREE,
    scaling_steps: int = DEFAULT_SCALING_STEPS,
    compute_dtype: torch.dtype | None = None,
) -> bool:
    """Replace one existing PyTorch matrix-exp parametrization in place.

    The containing ``ParametrizationList`` and its ``original`` parameter are
    retained.  The existing ``base`` tensor is registered on the replacement,
    preserving parameter identity, optimizer references, parameter count, and
    state-dict keys in both directions.
    """

    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or name not in parametrizations:
        return False
    parametrization_list = parametrizations[name]
    replacements: list[tuple[int, nn.Module]] = []
    for index, candidate in enumerate(parametrization_list):
        orthogonal_map = getattr(candidate, "orthogonal_map", None)
        if getattr(orthogonal_map, "name", None) != "matrix_exp":
            continue
        base = candidate._buffers.get("base")
        shape = getattr(candidate, "shape", None)
        if not isinstance(base, Tensor) or shape is None:
            message = "matrix-exp orthogonal parametrization must use dynamic trivialization"
            raise ValueError(message)
        replacement = CaptureSafeMatrixExpOrthogonal(
            shape,
            base,
            taylor_degree=taylor_degree,
            scaling_steps=scaling_steps,
            compute_dtype=compute_dtype,
        )
        replacement.train(candidate.training)
        replacements.append((index, replacement))

    for index, replacement in replacements:
        parametrization_list[index] = replacement
    return bool(replacements)


def prepare_capture_safe_orthogonal_(
    model: nn.Module,
    *,
    taylor_degree: int = DEFAULT_TAYLOR_DEGREE,
    scaling_steps: int = DEFAULT_SCALING_STEPS,
    compute_dtype: torch.dtype | None = None,
) -> tuple[str, ...]:
    """Replace every existing matrix-exp orthogonal parametrization in ``model``.

    Returns the qualified tensor paths that were changed.  Calling the
    function again is idempotent and returns an empty tuple.
    """

    replaced: list[str] = []
    for module_path, module in tuple(model.named_modules()):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        replaced.extend(
            f"{module_path}.{name}" if module_path else name
            for name in tuple(parametrizations.keys())
            if replace_matrix_exp_orthogonal_(
                module,
                name,
                taylor_degree=taylor_degree,
                scaling_steps=scaling_steps,
                compute_dtype=compute_dtype,
            )
        )
    return tuple(replaced)


__all__ = [
    "DEFAULT_SCALING_STEPS",
    "DEFAULT_TAYLOR_DEGREE",
    "CaptureSafeMatrixExpOrthogonal",
    "fixed_taylor_matrix_exp",
    "prepare_capture_safe_orthogonal_",
    "replace_matrix_exp_orthogonal_",
]

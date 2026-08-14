"""Shape-specialized CUDA Graph VJP for PyTorch's FP32 matrix exponential.

PyTorch 2.6 evaluates ``matrix_exp_backward`` by exponentiating a 2n-by-2n
block matrix with an adaptive optimized Taylor polynomial.  The adaptive norm
decision synchronizes CUDA with the host, and the generic implementation
recreates coefficients and workspaces on every call.  This module preserves
the same T1/T2/T4/T8/T12/T18 formulas and operation order, but captures one
static graph for each branch used by PAC's fixed 64-by-64 frame maps.
"""

from __future__ import annotations

import math
import shutil
import struct
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, Literal, Protocol

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

MatrixExpDispatch = Literal["host", "cuda_switch"]


class NativeMatrixExpReplay(Protocol):
    """Common borrowed-output interface for captured matrix-exp dispatchers."""

    def replay_forward(self, matrix: Tensor, branch: str | None = None) -> Tensor: ...

    def replay(
        self,
        matrix: Tensor,
        output_gradient: Tensor,
        branch: str | None = None,
    ) -> Tensor: ...

    def destroy(self) -> None: ...


def reset_cuda_graphs(
    graphs: Iterable[torch.cuda.CUDAGraph | None],
) -> int:
    """Reset each distinct owned CUDA graph exactly once."""
    reset_count = 0
    seen: set[int] = set()
    for graph in graphs:
        if graph is None or id(graph) in seen:
            continue
        seen.add(id(graph))
        graph.reset()
        reset_count += 1
    return reset_count


_THETAS: Final = (
    1.192092800768788e-07,
    5.978858893805233e-04,
    5.116619363445086e-02,
    5.800524627688768e-01,
    1.461661507209034e00,
    3.010066362817634e00,
)
_MAX_SCALING_STEPS: Final = 4

_T12_COEFFICIENTS: Final = (
    (9.0198e-16, 0.46932117595418237389, -0.20099424927047284052, -0.04623946134063071740),
    (
        5.31597895759871264183,
        1.19926790417132231573,
        0.01179296240992997031,
        0.01108844528519167989,
    ),
    (
        0.18188869982170434744,
        0.05502798439925399070,
        0.09351590770535414968,
        0.00610700528898058230,
    ),
    (-2.0861320e-13, -0.13181061013830184015, -0.02027855540589259079, -0.00675951846863086359),
)
_T18_COEFFICIENTS: Final = (
    (
        0.0,
        -1.00365581030144618291e-01,
        -8.02924648241156932449e-03,
        -8.92138498045729985177e-04,
        0.0,
    ),
    (
        0.0,
        3.97849749499645077844e-01,
        1.36783778460411720168,
        4.98289622525382669416e-01,
        -6.37898194594723280150e-04,
    ),
    (
        -1.09676396052962061844e01,
        1.68015813878906206114,
        5.71779846478865511061e-02,
        -6.98210122488052056106e-03,
        3.34975017086070470649e-05,
    ),
    (
        -9.04316832390810593223e-02,
        -6.76404519071381882256e-02,
        6.75961301770459654925e-02,
        2.95552570429315521194e-02,
        -1.39180257516060693404e-05,
    ),
    (
        0.0,
        0.0,
        -9.23364619367118555360e-02,
        -1.69364939002081722752e-02,
        -1.40086798182036094347e-05,
    ),
)


def _fp32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@contextmanager
def _temporary_float32_matmul_policy(*, allow_tf32: bool) -> Generator[None]:
    """Temporarily select one FP32 matmul policy and restore it exactly.

    ``torch.set_float32_matmul_precision`` and the CUDA backend flag overlap,
    but callers are allowed to put them in an unusual combination.  Save and
    restore both (plus the untouched cuDNN flag) so a failed graph capture
    cannot leak process-global numerical policy into the training runtime.
    """
    previous_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def matrix_exp_vjp_one_norm(matrix: Tensor, output_gradient: Tensor) -> Tensor:
    """Return the 1-norm of PyTorch's matrix-exp backward block matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        message = "matrix-exp VJP expects one square matrix"
        raise ValueError(message)
    if output_gradient.shape != matrix.shape:
        message = "matrix-exp output gradient must match the matrix shape"
        raise ValueError(message)
    transposed_column_sums = matrix.mT.abs().sum(dim=-2)
    gradient_column_sums = output_gradient.abs().sum(dim=-2)
    return (transposed_column_sums + gradient_column_sums).max()


def matrix_exp_one_norm(matrix: Tensor) -> Tensor:
    """Return the operator 1-norm used by PyTorch's forward dispatch."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        message = "matrix-exp expects one square matrix"
        raise ValueError(message)
    return matrix.abs().sum(dim=-2).max()


def matrix_exp_vjp_branch(norm: float) -> str | None:
    """Map a finite FP32 1-norm to PyTorch 2.6's adaptive branch."""
    if not math.isfinite(norm) or norm < 0.0:
        return None
    for degree, threshold in zip((1, 2, 4, 8, 12), _THETAS[:-1], strict=True):
        if norm <= threshold:
            return f"t{degree}"
    scaling_steps = max(0, math.ceil(math.log2(norm / _THETAS[-1])))
    if scaling_steps > _MAX_SCALING_STEPS:
        return None
    return f"t18_s{scaling_steps}"


def cuda_switch_matrix_exp_capability() -> tuple[bool, str]:
    """Return whether the device-selected ten-way SWITCH backend can load."""
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
        return False, "PyTorch raw CUDA Graph access is required"
    cuda_version = torch.version.cuda
    if cuda_version is None:
        return False, "PyTorch has no CUDA runtime"
    runtime_version = tuple(int(part) for part in cuda_version.split(".")[:2])
    if runtime_version < (12, 8):
        return False, "CUDA 12.8+ SWITCH graph support is required"
    if shutil.which("nvcc") is None:
        return False, "nvcc is unavailable"
    return True, "available"


def make_native_matrix_exp_replay(
    size: int,
    device: torch.device,
    *,
    dispatch: MatrixExpDispatch,
    forward_tf32: bool = False,
    direct_skew_vjp: bool = False,
) -> NativeMatrixExpReplay:
    """Build one host-selected or device-SWITCH borrowed-output runtime."""
    if dispatch == "host":
        if direct_skew_vjp:
            message = "direct skew VJP requires CUDA SWITCH dispatch"
            raise ValueError(message)
        return CapturedNativeMatrixExpVJP(size, device, forward_tf32=forward_tf32)
    available, reason = cuda_switch_matrix_exp_capability()
    if not available:
        message = f"CUDA SWITCH matrix-exp backend is unavailable: {reason}"
        raise RuntimeError(message)
    from .pac_cuda_conditional_matrix_exp import SwitchNativeMatrixExp  # noqa: PLC0415

    return SwitchNativeMatrixExp(
        size,
        device,
        forward_tf32=forward_tf32,
        direct_skew_vjp=direct_skew_vjp,
    )


class CapturedNativeMatrixExpVJP:
    """Replay native-equivalent fixed-branch FP32 matrix-exp VJPs."""

    def __init__(
        self,
        size: int,
        device: torch.device,
        *,
        forward_tf32: bool = False,
    ) -> None:
        if size < 2:
            message = "captured matrix-exp VJP requires size >= 2"
            raise ValueError(message)
        if device.type != "cuda":
            message = "captured matrix-exp VJP requires CUDA"
            raise ValueError(message)
        self.size = size
        self.device = device
        self.forward_tf32 = forward_tf32
        self._destroyed = False
        self._matrix = torch.zeros(size, size, device=device, dtype=torch.float32)
        self._forward_matrix = torch.zeros_like(self._matrix)
        self._output_gradient = torch.zeros_like(self._matrix)
        self._identity = torch.eye(2 * size, device=device, dtype=torch.float32)
        self._forward_identity = torch.eye(size, device=device, dtype=torch.float32)
        self._t4_inner_coefficients = torch.tensor(
            ((0.5, 1.0 / 6.0, 1.0 / 24.0),), device=device, dtype=torch.float32
        )
        self._t4_output_coefficients = torch.tensor(
            ((1.0, 1.0, 1.0),), device=device, dtype=torch.float32
        )
        sqrt_177 = _fp32(0.1330413469565007072504e2)
        x3 = _fp32(2.0 / 3.0)
        x1 = _fp32(x3 * ((1.0 + sqrt_177) / 88.0))
        x2 = _fp32(x3 * ((1.0 + sqrt_177) / 352.0))
        x4 = _fp32((-271.0 + 29.0 * sqrt_177) / (315.0 * x3))
        x5 = _fp32((-11.0 + 11.0 * sqrt_177) / (1260.0 * x3))
        x6 = _fp32((-99.0 + 11.0 * sqrt_177) / (5040.0 * x3))
        x7 = _fp32((89.0 - sqrt_177) / (5040.0 * x3))
        y2 = _fp32((857.0 - 58.0 * sqrt_177) / 630.0)
        self._t8_a4_coefficients = torch.tensor(
            ((x1, x2),),
            device=device,
            dtype=torch.float32,
        )
        self._t8_left_coefficients = torch.tensor(((x3, 1.0),), device=device, dtype=torch.float32)
        self._t8_right_coefficients = torch.tensor(
            (
                (
                    x4,
                    x5,
                    x6,
                    x7,
                ),
            ),
            device=device,
            dtype=torch.float32,
        )
        self._t8_output_coefficients = torch.tensor(
            ((1.0, 1.0, y2, 0.0, 1.0),),
            device=device,
            dtype=torch.float32,
        )
        self._t12_coefficients = torch.tensor(
            _T12_COEFFICIENTS,
            device=device,
            dtype=torch.float32,
        )
        self._t18_coefficients = torch.tensor(
            _T18_COEFFICIENTS,
            device=device,
            dtype=torch.float32,
        )
        if forward_tf32:
            with _temporary_float32_matmul_policy(allow_tf32=False):
                self._graphs, raw_outputs = self._capture_branches(
                    self._backward_block_matrix,
                    self._identity,
                )
            self._outputs = {
                name: output[: self.size, self.size :] for name, output in raw_outputs.items()
            }
            with _temporary_float32_matmul_policy(allow_tf32=True):
                self._forward_graphs, self._forward_outputs = self._capture_branches(
                    lambda: self._forward_matrix,
                    self._forward_identity,
                )
        else:
            self._graphs, raw_outputs = self._capture_branches(
                self._backward_block_matrix,
                self._identity,
            )
            self._outputs = {
                name: output[: self.size, self.size :] for name, output in raw_outputs.items()
            }
            self._forward_graphs, self._forward_outputs = self._capture_branches(
                lambda: self._forward_matrix,
                self._forward_identity,
            )

    def replay(
        self,
        matrix: Tensor,
        output_gradient: Tensor,
        branch: str | None = None,
    ) -> Tensor:
        """Copy dynamic inputs, replay one branch, and borrow its VJP output."""
        if branch is None:
            message = "host matrix-exp VJP replay requires an explicit branch"
            raise ValueError(message)
        graph = self._graphs.get(branch)
        if graph is None:
            message = f"uncaptured matrix-exp VJP branch: {branch}"
            raise ValueError(message)
        self._matrix.copy_(matrix)
        self._output_gradient.copy_(output_gradient)
        graph.replay()
        return self._outputs[branch]

    def replay_forward(self, matrix: Tensor, branch: str | None = None) -> Tensor:
        """Copy one dynamic matrix and borrow its captured exponential."""
        if branch is None:
            message = "host matrix-exp forward replay requires an explicit branch"
            raise ValueError(message)
        graph = self._forward_graphs.get(branch)
        if graph is None:
            message = f"uncaptured matrix-exp forward branch: {branch}"
            raise ValueError(message)
        self._forward_matrix.copy_(matrix)
        graph.replay()
        return self._forward_outputs[branch]

    def destroy(self) -> None:
        """Release every captured branch graph and its private graph pool."""
        if getattr(self, "_destroyed", False):
            return
        reset_cuda_graphs((*self._graphs.values(), *self._forward_graphs.values()))
        self._graphs.clear()
        self._forward_graphs.clear()
        self._outputs.clear()
        self._forward_outputs.clear()
        self._destroyed = True

    def _capture_branches(
        self,
        matrix_factory: Callable[[], Tensor],
        identity: Tensor,
    ) -> tuple[dict[str, torch.cuda.CUDAGraph], dict[str, Tensor]]:
        branch_functions: dict[str, Callable[[Tensor], Tensor]] = {
            "t1": self._t1,
            "t2": self._t2,
            "t4": self._t4,
            "t8": self._t8,
            "t12": self._t12,
            **{
                f"t18_s{scaling_steps}": (
                    lambda matrix, steps=scaling_steps: self._t18(matrix, steps)
                )
                for scaling_steps in range(_MAX_SCALING_STEPS + 1)
            },
        }
        previous_identity = self._identity
        self._identity = identity
        stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(self.device)
        graphs: dict[str, torch.cuda.CUDAGraph] = {}
        outputs: dict[str, Tensor] = {}
        for name, operation in branch_functions.items():
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    output = operation(matrix_factory())
            current_stream.wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                output = operation(matrix_factory())
            current_stream.wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graphs[name] = graph
            outputs[name] = output
        self._identity = previous_identity
        return graphs, outputs

    def _backward_block_matrix(self) -> Tensor:
        transposed = self._matrix.mT
        zeros = torch.zeros_like(transposed)
        return torch.cat(
            (
                torch.cat((transposed, self._output_gradient), dim=1),
                torch.cat((zeros, transposed), dim=1),
            ),
            dim=0,
        )

    def _powers(self, matrix: Tensor, count: int) -> Tensor:
        values = [self._identity, matrix]
        if count >= 3:
            values.append(matrix @ matrix)
        if count >= 4:
            values.append(matrix @ values[2])
        if count >= 5:
            values.append(values[3] @ values[3])
        return torch.stack(values)

    @staticmethod
    def _linear_combination(powers: Tensor, coefficients: Tensor) -> Tensor:
        return torch.ops.aten._compute_linear_combination(  # noqa: SLF001
            powers, coefficients
        )

    def _t1(self, matrix: Tensor) -> Tensor:
        return self._powers(matrix, 2).sum(dim=0)

    def _t2(self, matrix: Tensor) -> Tensor:
        powers = self._powers(matrix, 3)
        return torch.stack((powers[0], powers[1], powers[2] / 2.0)).sum(dim=0)

    def _t4(self, matrix: Tensor) -> Tensor:
        powers = self._powers(matrix, 3)
        inner = self._linear_combination(
            powers,
            self._t4_inner_coefficients,
        )[0]
        return self._linear_combination(
            torch.stack((powers[0], powers[1], powers[2] @ inner)),
            self._t4_output_coefficients,
        )[0]

    def _t8(self, matrix: Tensor) -> Tensor:
        powers = self._powers(matrix, 3)
        a4 = powers[2] @ self._linear_combination(powers[1:3], self._t8_a4_coefficients)[0]
        powers4 = torch.cat((powers, a4.unsqueeze(0)), dim=0)
        a8 = (
            self._linear_combination(powers4[2:4], self._t8_left_coefficients)[0]
            @ self._linear_combination(powers4, self._t8_right_coefficients)[0]
        )
        return self._linear_combination(
            torch.cat((powers4, a8.unsqueeze(0)), dim=0),
            self._t8_output_coefficients,
        )[0]

    def _t12(self, matrix: Tensor) -> Tensor:
        powers = self._powers(matrix, 4)
        b0, b1, b2, b3 = self._linear_combination(powers, self._t12_coefficients).unbind(0)
        b2 = b2 + b3 @ b3
        b1 = b1 + b2
        return b0 + b1 @ b2

    def _t18(self, matrix: Tensor, scaling_steps: int) -> Tensor:
        scaled = matrix * (2.0**-scaling_steps)
        powers = self._powers(scaled, 5)
        b0, b1, b2, b3, b4 = self._linear_combination(powers, self._t18_coefficients).unbind(0)
        b3 = b3 + b0 @ b4
        result = b1 + (b2 + b3) @ b3
        for _ in range(scaling_steps):
            result = result @ result
        return result


__all__ = [
    "CapturedNativeMatrixExpVJP",
    "MatrixExpDispatch",
    "NativeMatrixExpReplay",
    "cuda_switch_matrix_exp_capability",
    "make_native_matrix_exp_replay",
    "matrix_exp_one_norm",
    "matrix_exp_vjp_branch",
    "matrix_exp_vjp_one_norm",
    "reset_cuda_graphs",
]

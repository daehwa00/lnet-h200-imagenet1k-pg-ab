"""Device-dispatched FP32 matrix exponential for CUDA Graph training.

This module removes the host synchronization
left in PAC's exact-split training runtime.  It evaluates the same optimized
Taylor formulas used by PyTorch 2.6 (T1/T2/T4/T8/T12/T18), but records the
adaptive decisions as CUDA conditional graph nodes.  Both the forward matrix
exponential and its block-matrix VJP therefore select their formula without a
device-to-host norm copy.

The public replay interface intentionally matches ``CapturedNativeMatrixExpVJP``
so a successful probe can replace that runtime without changing model code.
Returned tensors borrow graph-owned storage and are overwritten by the next
replay of the same graph.
"""

from __future__ import annotations

import struct
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import torch
from torch import Tensor

from .pac_triton_skew_matrix_exp_vjp import fused_direct_skew_vjp

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

_THETAS: Final = (
    1.192092800768788e-07,
    5.978858893805233e-04,
    5.116619363445086e-02,
    5.800524627688768e-01,
    1.461661507209034e00,
    3.010066362817634e00,
)
_MAX_SCALING_STEPS: Final = 4


def _reset_cuda_graphs(
    graphs: Iterable[torch.cuda.CUDAGraph | None],
) -> int:
    reset_count = 0
    seen: set[int] = set()
    for graph in graphs:
        if graph is None or id(graph) in seen:
            continue
        seen.add(id(graph))
        graph.reset()
        reset_count += 1
    return reset_count
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


class _ConditionalGraph(Protocol):
    def begin_capture_to_if_node(self, predicate: Tensor) -> None: ...

    def end_capture_to_conditional_node(self) -> None: ...


class _ConditionalGraphType(Protocol):
    @staticmethod
    def get_currently_capturing_graph() -> _ConditionalGraph: ...


class _SwitchExtension(Protocol):
    def add_matrix_exp_switch(
        self,
        root_graph: int,
        norm_pointer: int,
        branch_graphs: list[int],
    ) -> None: ...


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


@lru_cache(maxsize=1)
def _load_switch_extension() -> _SwitchExtension:
    import os  # noqa: PLC0415

    from torch.utils.cpp_extension import CUDA_HOME, load  # noqa: PLC0415

    if CUDA_HOME is None:
        message = "CUDA toolkit root is unavailable"
        raise RuntimeError(message)
    cuda_home = Path(CUDA_HOME)
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        message = "CUDA 12.8+ nvcc is required for matrix-exp SWITCH graphs"
        raise RuntimeError(message)
    previous_cuda_home = os.environ.get("CUDA_HOME")
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["CUDA_HOME"] = str(cuda_home)
    capability = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
    try:
        module = load(
            name="pac_cuda_conditional_switch_v2",
            sources=[
                str(Path(__file__).resolve().parents[2] / "csrc" / "pac_cuda_conditional_switch.cu")
            ],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    finally:
        if previous_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = previous_cuda_home
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    return cast("_SwitchExtension", cast("object", module))


def _fp32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@contextmanager
def _temporary_float32_matmul_policy(*, allow_tf32: bool) -> Generator[None]:
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


class ConditionalNativeMatrixExp:
    """Replay adaptive native-formula matrix-exp and VJP CUDA Graphs.

    The captured domain is the one used by PAC's PyTorch 2.6 specialization:
    finite matrices whose operator one-norm needs at most four T18 squarings.
    ``maximum_supported_norm`` is exposed so a training campaign can validate
    that invariant before selecting this runtime.
    """

    def __init__(
        self,
        size: int,
        device: torch.device,
        *,
        forward_tf32: bool = False,
    ) -> None:
        if size < 2:
            message = "conditional matrix-exp requires size >= 2"
            raise ValueError(message)
        if device.type != "cuda":
            message = "conditional matrix-exp requires CUDA"
            raise ValueError(message)
        required_graph_methods = (
            "begin_capture_to_if_node",
            "end_capture_to_conditional_node",
            "get_currently_capturing_graph",
        )
        if any(not hasattr(torch.cuda.CUDAGraph, name) for name in required_graph_methods):
            message = "installed PyTorch lacks conditional CUDA Graph capture support"
            raise RuntimeError(message)

        self.size = size
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.forward_tf32 = forward_tf32
        self.direct_skew_vjp = False
        self._destroyed = False
        device = self.device
        self.maximum_supported_norm = _THETAS[-1] * (2**_MAX_SCALING_STEPS)
        self._matrix = torch.zeros(size, size, device=device, dtype=torch.float32)
        self._forward_matrix = torch.zeros_like(self._matrix)
        self._output_gradient = torch.zeros_like(self._matrix)
        self._identity = torch.eye(2 * size, device=device, dtype=torch.float32)
        self._forward_identity = torch.eye(size, device=device, dtype=torch.float32)
        self._forward_output_buffer = torch.empty_like(self._forward_matrix)
        self._backward_output_buffer = torch.empty_like(self._identity)
        self._backward_block_buffer = torch.empty_like(self._identity)
        self._branch_boundaries = torch.tensor(
            (*_THETAS[:-1], *(_THETAS[-1] * (2**step) for step in range(4))),
            device=device,
            dtype=torch.float32,
        )
        self._coefficients = self._make_coefficients(device)

        if forward_tf32:
            with _temporary_float32_matmul_policy(allow_tf32=True):
                self._warmup(self._forward_matrix, self._forward_identity)
                self._forward_graph, self._forward_output = self._capture_forward()
            with _temporary_float32_matmul_policy(allow_tf32=False):
                self._warmup(self._backward_block_matrix(), self._identity)
                self._backward_graph, backward_output = self._capture_backward()
        else:
            self._warmup(self._forward_matrix, self._forward_identity)
            self._warmup(self._backward_block_matrix(), self._identity)
            self._forward_graph, self._forward_output = self._capture_forward()
            self._backward_graph, backward_output = self._capture_backward()
        self._backward_output = backward_output[:size, size:]

    def replay_forward(self, matrix: Tensor, branch: str | None = None) -> Tensor:
        """Replay forward adaptive dispatch and return graph-owned output.

        ``branch`` is accepted for drop-in compatibility and deliberately
        ignored: the captured GPU norm selects the branch.
        """
        del branch
        self._validate_matrix(matrix)
        self._forward_matrix.copy_(matrix)
        self._forward_graph.replay()
        return self._forward_output

    def replay(
        self,
        matrix: Tensor,
        output_gradient: Tensor,
        branch: str | None = None,
    ) -> Tensor:
        """Replay adaptive block-matrix VJP and return graph-owned gradient."""
        del branch
        self._validate_matrix(matrix)
        self._validate_matrix(output_gradient)
        self._matrix.copy_(matrix)
        self._output_gradient.copy_(output_gradient)
        self._backward_graph.replay()
        return self._backward_output

    def destroy(self) -> None:
        """Release the two conditional root graphs idempotently."""
        if getattr(self, "_destroyed", False):
            return
        _reset_cuda_graphs((self._forward_graph, self._backward_graph))
        self._destroyed = True

    def _capture_forward(self) -> tuple[torch.cuda.CUDAGraph, Tensor]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            norm = self._forward_matrix.abs().sum(dim=-2).max()
            output = self._adaptive_exp(
                self._forward_matrix,
                self._forward_identity,
                norm,
                self._forward_output_buffer,
            )
        return graph, output

    def _capture_backward(self) -> tuple[torch.cuda.CUDAGraph, Tensor]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block = self._backward_block_matrix()
            transposed_column_sums = self._matrix.mT.abs().sum(dim=-2)
            gradient_column_sums = self._output_gradient.abs().sum(dim=-2)
            norm = (transposed_column_sums + gradient_column_sums).max()
            output = self._adaptive_exp(
                block,
                self._identity,
                norm,
                self._backward_output_buffer,
            )
        return graph, output

    def _adaptive_exp(
        self,
        matrix: Tensor,
        identity: Tensor,
        norm: Tensor,
        output: Tensor,
    ) -> Tensor:
        branch_index = torch.bucketize(norm, self._branch_boundaries)
        branches: list[Callable[[Tensor], Tensor]] = [
            lambda value: self._t1(value, identity),
            lambda value: self._t2(value, identity),
            lambda value: self._t4(value, identity),
            lambda value: self._t8(value, identity),
            lambda value: self._t12(value, identity),
            *(lambda value, step=step: self._t18(value, identity, step) for step in range(5)),
        ]
        graph_type = cast("_ConditionalGraphType", cast("object", torch.cuda.CUDAGraph))
        graph = graph_type.get_currently_capturing_graph()
        for index, operation in enumerate(branches):
            predicate = branch_index == index
            graph.begin_capture_to_if_node(predicate)
            try:
                output.copy_(operation(matrix))
            finally:
                graph.end_capture_to_conditional_node()
        return output

    def _warmup(self, matrix: Tensor, identity: Tensor) -> None:
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._t1(matrix, identity)
                self._t2(matrix, identity)
                self._t4(matrix, identity)
                self._t8(matrix, identity)
                self._t12(matrix, identity)
                for scaling_steps in range(_MAX_SCALING_STEPS + 1):
                    self._t18(matrix, identity, scaling_steps)
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)

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

    def _powers(self, matrix: Tensor, identity: Tensor, count: int) -> Tensor:
        values = [identity, matrix]
        if count >= 3:
            values.append(matrix @ matrix)
        if count >= 4:
            values.append(matrix @ values[2])
        if count >= 5:
            values.append(values[3] @ values[3])
        return torch.stack(values)

    @staticmethod
    def _linear_combination(powers: Tensor, coefficients: Tensor) -> Tensor:
        return torch.ops.aten._compute_linear_combination(powers, coefficients)  # noqa: SLF001

    def _t1(self, matrix: Tensor, identity: Tensor) -> Tensor:
        return self._powers(matrix, identity, 2).sum(dim=0)

    def _t2(self, matrix: Tensor, identity: Tensor) -> Tensor:
        powers = self._powers(matrix, identity, 3)
        return torch.stack((powers[0], powers[1], powers[2] / 2.0)).sum(dim=0)

    def _t4(self, matrix: Tensor, identity: Tensor) -> Tensor:
        powers = self._powers(matrix, identity, 3)
        inner = self._linear_combination(powers, self._coefficients["t4_inner"])[0]
        return self._linear_combination(
            torch.stack((powers[0], powers[1], powers[2] @ inner)),
            self._coefficients["t4_output"],
        )[0]

    def _t8(self, matrix: Tensor, identity: Tensor) -> Tensor:
        powers = self._powers(matrix, identity, 3)
        a4 = powers[2] @ self._linear_combination(powers[1:3], self._coefficients["t8_a4"])[0]
        powers4 = torch.cat((powers, a4.unsqueeze(0)), dim=0)
        a8 = (
            self._linear_combination(powers4[2:4], self._coefficients["t8_left"])[0]
            @ self._linear_combination(powers4, self._coefficients["t8_right"])[0]
        )
        return self._linear_combination(
            torch.cat((powers4, a8.unsqueeze(0)), dim=0),
            self._coefficients["t8_output"],
        )[0]

    def _t12(self, matrix: Tensor, identity: Tensor) -> Tensor:
        powers = self._powers(matrix, identity, 4)
        b0, b1, b2, b3 = self._linear_combination(powers, self._coefficients["t12"]).unbind(0)
        b2 = b2 + b3 @ b3
        b1 = b1 + b2
        return b0 + b1 @ b2

    def _t18(self, matrix: Tensor, identity: Tensor, scaling_steps: int) -> Tensor:
        scaled = matrix * (2.0**-scaling_steps)
        powers = self._powers(scaled, identity, 5)
        b0, b1, b2, b3, b4 = self._linear_combination(powers, self._coefficients["t18"]).unbind(0)
        b3 = b3 + b0 @ b4
        result = b1 + (b2 + b3) @ b3
        for _ in range(scaling_steps):
            result = result @ result
        return result

    def _validate_matrix(self, matrix: Tensor) -> None:
        if matrix.device != self.device:
            message = "matrix-exp replay input is on the wrong device"
            raise ValueError(message)
        if matrix.dtype != torch.float32 or matrix.shape != (self.size, self.size):
            message = f"matrix-exp replay expects FP32 [{self.size}, {self.size}]"
            raise ValueError(message)

    @staticmethod
    def _make_coefficients(device: torch.device) -> dict[str, Tensor]:
        sqrt_177 = _fp32(0.1330413469565007072504e2)
        x3 = _fp32(2.0 / 3.0)
        x1 = _fp32(x3 * ((1.0 + sqrt_177) / 88.0))
        x2 = _fp32(x3 * ((1.0 + sqrt_177) / 352.0))
        x4 = _fp32((-271.0 + 29.0 * sqrt_177) / (315.0 * x3))
        x5 = _fp32((-11.0 + 11.0 * sqrt_177) / (1260.0 * x3))
        x6 = _fp32((-99.0 + 11.0 * sqrt_177) / (5040.0 * x3))
        x7 = _fp32((89.0 - sqrt_177) / (5040.0 * x3))
        y2 = _fp32((857.0 - 58.0 * sqrt_177) / 630.0)

        def tensor(values: object) -> Tensor:
            return torch.tensor(values, device=device, dtype=torch.float32)

        return {
            "t4_inner": tensor(((0.5, 1.0 / 6.0, 1.0 / 24.0),)),
            "t4_output": tensor(((1.0, 1.0, 1.0),)),
            "t8_a4": tensor(((x1, x2),)),
            "t8_left": tensor(((x3, 1.0),)),
            "t8_right": tensor(((x4, x5, x6, x7),)),
            "t8_output": tensor(((1.0, 1.0, y2, 0.0, 1.0),)),
            "t12": tensor(_T12_COEFFICIENTS),
            "t18": tensor(_T18_COEFFICIENTS),
        }


class SwitchNativeMatrixExp(ConditionalNativeMatrixExp):
    """Ten-way CUDA SWITCH variant with one device selector kernel."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        size: int,
        device: torch.device,
        *,
        forward_tf32: bool = False,
        direct_skew_vjp: bool = False,
    ) -> None:
        if size < 2:
            message = "conditional matrix-exp requires size >= 2"
            raise ValueError(message)
        if device.type != "cuda":
            message = "conditional matrix-exp requires CUDA"
            raise ValueError(message)
        if direct_skew_vjp and size != 64:
            message = "direct skew VJP is specialized for 64x64 frame maps"
            raise ValueError(message)
        if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
            message = "matrix-exp SWITCH graphs require raw CUDA Graph access"
            raise RuntimeError(message)
        cuda_version = torch.version.cuda
        if cuda_version is None or tuple(int(part) for part in cuda_version.split(".")[:2]) < (
            12,
            8,
        ):
            message = "matrix-exp SWITCH graphs require CUDA 12.8 or newer"
            raise RuntimeError(message)

        self.size = size
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.forward_tf32 = forward_tf32
        self.direct_skew_vjp = direct_skew_vjp
        self._destroyed = False
        self.direct_fast_path_maximum_norm = 0.0
        device = self.device
        self._capture_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device)
        self._capture_stream.wait_stream(current_stream)
        self.maximum_supported_norm = _THETAS[-1] * (2**_MAX_SCALING_STEPS)
        self._matrix = torch.zeros(size, size, device=device, dtype=torch.float32)
        self._forward_matrix = torch.zeros_like(self._matrix)
        self._output_gradient = torch.zeros_like(self._matrix)
        self._identity = torch.eye(2 * size, device=device, dtype=torch.float32)
        self._forward_identity = torch.eye(size, device=device, dtype=torch.float32)
        self._forward_output_buffer = torch.empty_like(self._forward_matrix)
        self._backward_output_buffer = torch.empty_like(self._identity)
        self._backward_block_buffer = torch.empty_like(self._identity)
        self._branch_boundaries = torch.empty(0, device=device, dtype=torch.float32)
        self._coefficients = self._make_coefficients(device)

        extension = _load_switch_extension()
        if forward_tf32:
            with _temporary_float32_matmul_policy(allow_tf32=True):
                (
                    self._forward_branch_graphs,
                    self._forward_graph,
                    self._forward_norm,
                ) = self._prepare_forward_switch(extension)
        else:
            (
                self._forward_branch_graphs,
                self._forward_graph,
                self._forward_norm,
            ) = self._prepare_forward_switch(extension)
        self._forward_output = self._forward_output_buffer

        if forward_tf32 or direct_skew_vjp:
            with _temporary_float32_matmul_policy(allow_tf32=False):
                (
                    self._backward_branch_graphs,
                    self._backward_graph,
                    self._backward_norm,
                    self._backward_output,
                ) = self._prepare_backward_runtime(extension)
        else:
            (
                self._backward_branch_graphs,
                self._backward_graph,
                self._backward_norm,
                self._backward_output,
            ) = self._prepare_backward_runtime(extension)
        current_stream.wait_stream(self._capture_stream)

    def _prepare_forward_switch(
        self,
        extension: _SwitchExtension,
    ) -> tuple[tuple[torch.cuda.CUDAGraph, ...], torch.cuda.CUDAGraph, Tensor]:
        self._warmup(self._forward_matrix, self._forward_identity)
        branches = self._capture_switch_branches(
            self._forward_matrix,
            self._forward_identity,
            self._forward_output_buffer,
        )
        graph, norm = self._capture_switch_root(
            lambda: self._forward_matrix.abs().sum(dim=-2).max(),
            branches,
            extension,
        )
        return branches, graph, norm

    def destroy(self) -> None:
        """Destroy SWITCH roots before their referenced branch graphs."""
        if getattr(self, "_destroyed", False):
            return
        _reset_cuda_graphs((self._forward_graph, self._backward_graph))
        _reset_cuda_graphs(
            (*self._forward_branch_graphs, *self._backward_branch_graphs)
        )
        self._forward_branch_graphs = ()
        self._backward_branch_graphs = ()
        self._destroyed = True

    def _prepare_backward_runtime(
        self,
        extension: _SwitchExtension,
    ) -> tuple[tuple[torch.cuda.CUDAGraph, ...], torch.cuda.CUDAGraph, Tensor, Tensor]:
        if self.direct_skew_vjp:
            self.direct_fast_path_maximum_norm = _THETAS[3]
            output = torch.empty_like(self._matrix)
            self._warmup(self._backward_block_matrix(), self._identity)
            self._warmup_direct_vjp()
            branches = self._capture_guarded_direct_vjp_branches(output)
            graph, norm = self._capture_switch_root(
                self._backward_one_norm,
                branches,
                extension,
            )
            return branches, graph, norm, output

        self._warmup(self._backward_block_matrix(), self._identity)
        branches = self._capture_switch_branches(
            self._backward_block_buffer,
            self._identity,
            self._backward_output_buffer,
        )
        graph, norm = self._capture_switch_root(
            self._prepare_backward_switch,
            branches,
            extension,
        )
        output = self._backward_output_buffer[: self.size, self.size :]
        return branches, graph, norm, output

    def _capture_switch_branches(
        self,
        matrix: Tensor,
        identity: Tensor,
        output: Tensor,
    ) -> tuple[torch.cuda.CUDAGraph, ...]:
        operations: list[Callable[[Tensor], Tensor]] = [
            lambda value: self._t1(value, identity),
            lambda value: self._t2(value, identity),
            lambda value: self._t4(value, identity),
            lambda value: self._t8(value, identity),
            lambda value: self._t12(value, identity),
            *(lambda value, step=step: self._t18(value, identity, step) for step in range(5)),
        ]
        pool = torch.cuda.graph_pool_handle()
        graphs: list[torch.cuda.CUDAGraph] = []
        for operation in operations:
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=pool, stream=self._capture_stream):
                output.copy_(operation(matrix))
            graphs.append(graph)
        return tuple(graphs)

    def _direct_skew_vjp(self, order: int) -> Tensor:
        """Evaluate the skew-matrix exponential VJP without a 2n block exp.

        For skew ``A`` and ``Q = exp(A)``, the ambient VJP is

        ``phi(ad_A)(Q.T @ G)``, where ``phi(z) = (exp(z) - 1) / z``.

        The forward replay already owns ``Q``.  In the production T8 domain,
        three commutators achieve sub-micro absolute error while replacing the
        128x128 block exponential with seven 64x64 matrix products.  Fused
        commutator kernels issue those products in four launches without
        materializing stacked operands.  The backward SWITCH guards this
        approximation with the conservative block norm (which upper-bounds
        ``||A||_1``); T12/T18 dispatches retain the exact block VJP.
        """
        return fused_direct_skew_vjp(
            self._matrix,
            self._forward_output_buffer,
            self._output_gradient,
            order=order,
        )

    def _warmup_direct_vjp(self) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        self._capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._capture_stream):
            for _ in range(2):
                self._direct_skew_vjp(order=3)
        current_stream.wait_stream(self._capture_stream)
        torch.cuda.synchronize(self.device)

    def _capture_guarded_direct_vjp_branches(
        self,
        output: Tensor,
    ) -> tuple[torch.cuda.CUDAGraph, ...]:
        direct_operations = tuple(
            lambda order=order: self._direct_skew_vjp(order) for order in range(4)
        )
        exact_operations: tuple[Callable[[Tensor], Tensor], ...] = (
            lambda value: self._t12(value, self._identity),
            *(
                lambda value, step=step: self._t18(value, self._identity, step)
                for step in range(_MAX_SCALING_STEPS + 1)
            ),
        )
        pool = torch.cuda.graph_pool_handle()
        graphs: list[torch.cuda.CUDAGraph] = []
        for operation in direct_operations:
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=pool, stream=self._capture_stream):
                output.copy_(operation())
            graphs.append(graph)
        for operation in exact_operations:
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=pool, stream=self._capture_stream):
                block_output = operation(self._backward_block_matrix())
                output.copy_(block_output[: self.size, self.size :])
            graphs.append(graph)
        return tuple(graphs)

    def _capture_switch_root(
        self,
        norm_factory: Callable[[], Tensor],
        branches: tuple[torch.cuda.CUDAGraph, ...],
        extension: _SwitchExtension,
    ) -> tuple[torch.cuda.CUDAGraph, Tensor]:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=self._capture_stream):
            norm = norm_factory()
        extension.add_matrix_exp_switch(
            graph.raw_cuda_graph(),
            norm.data_ptr(),
            [branch.raw_cuda_graph() for branch in branches],
        )
        graph.instantiate()
        return graph, norm

    def _backward_one_norm(self) -> Tensor:
        transposed_column_sums = self._matrix.mT.abs().sum(dim=-2)
        gradient_column_sums = self._output_gradient.abs().sum(dim=-2)
        return (transposed_column_sums + gradient_column_sums).max()

    def _prepare_backward_switch(self) -> Tensor:
        self._backward_block_buffer.copy_(self._backward_block_matrix())
        return self._backward_one_norm()


def benchmark_conditional_matrix_exp(
    *,
    size: int = 64,
    iterations: int = 200,
    seed: int = 7,
) -> dict[str, float | int | str]:
    """Run a bounded native-versus-conditional RTX microbenchmark."""
    if not torch.cuda.is_available():
        message = "conditional matrix-exp benchmark requires CUDA"
        raise RuntimeError(message)
    device = torch.device("cuda")
    torch.manual_seed(seed)
    matrix = torch.randn(size, size, device=device, dtype=torch.float32)
    matrix = matrix - matrix.mT
    matrix.mul_(1.0 / matrix.abs().sum(dim=-2).max())
    gradient = torch.randn_like(matrix)
    gradient.mul_(1.0 / gradient.abs().sum(dim=-2).max())
    runtime = ConditionalNativeMatrixExp(size, device)

    expected_forward = torch.matrix_exp(matrix)
    expected_vjp = torch.ops.aten.matrix_exp_backward(matrix, gradient)
    actual_forward = runtime.replay_forward(matrix).clone()
    actual_vjp = runtime.replay(matrix, gradient).clone()
    torch.cuda.synchronize(device)

    def elapsed(operation: Callable[[], object]) -> float:
        for _ in range(10):
            operation()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize(device)
        return (time.perf_counter() - started) * 1000.0 / iterations

    native_forward_ms = elapsed(lambda: torch.matrix_exp(matrix))
    conditional_forward_ms = elapsed(lambda: runtime.replay_forward(matrix))
    native_vjp_ms = elapsed(lambda: torch.ops.aten.matrix_exp_backward(matrix, gradient))
    conditional_vjp_ms = elapsed(lambda: runtime.replay(matrix, gradient))
    return {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "size": size,
        "iterations": iterations,
        "forward_max_abs_error": float((actual_forward - expected_forward).abs().max()),
        "vjp_max_abs_error": float((actual_vjp - expected_vjp).abs().max()),
        "native_forward_ms": native_forward_ms,
        "conditional_forward_ms": conditional_forward_ms,
        "forward_speedup": native_forward_ms / conditional_forward_ms,
        "native_vjp_ms": native_vjp_ms,
        "conditional_vjp_ms": conditional_vjp_ms,
        "vjp_speedup": native_vjp_ms / conditional_vjp_ms,
    }


__all__ = [
    "ConditionalNativeMatrixExp",
    "SwitchNativeMatrixExp",
    "benchmark_conditional_matrix_exp",
]

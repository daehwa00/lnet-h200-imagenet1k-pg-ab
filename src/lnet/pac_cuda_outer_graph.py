# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
"""Raw CUDA outer graphs for EFP16 exact-split training.

This module leaves the reusable exact-split runtime unchanged and wraps its
stages with ``keep_graph=True``. It nests ordinary stages as sequential CUDA
child-graph nodes and flattens CUDA 12.8+ SWITCH nodes into the root graph. The
resulting replay replaces the host sequence of two forward matrix-exponential
launches, the captured model body, two matrix-exponential VJPs, the optimizer,
and the model's post-optimizer projection with one compute graph launch.

``EFP16ExactSplitOuterGraph.step`` accepts arbitrary same-shape input tensors
and stages them into the runtime-owned buffers before the compute replay. For a
strict one-launch path, callers may either fill those buffers and call
``step_static`` or retain the construction tensors as a fixed-address lease and
call ``step_leased``. Its loss is graph-owned and overwritten by the next
replay.
"""

from __future__ import annotations

import os
import sysconfig
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional

from .pac_native_matrix_exp_vjp import matrix_exp_vjp_branch, reset_cuda_graphs

if TYPE_CHECKING:
    from collections.abc import Callable

    from .pac_efp16_exact_split_training import EFP16ExactSplitTraining


class _NativeSequentialGraph(Protocol):
    def launch(self, stream: int) -> None: ...

    def raw_cuda_graph(self) -> int: ...


class _OuterGraphExtension(Protocol):
    def SequentialChildGraph(self, child_graphs: list[int]) -> _NativeSequentialGraph: ...  # noqa: N802

    def SequentialMixedGraph(  # noqa: N802
        self,
        child_graphs: list[int],
        switch_after_child: list[int],
        norm_pointers: list[int],
        branch_graphs: list[list[int]],
    ) -> _NativeSequentialGraph: ...

    def DependencyMixedGraph(  # noqa: N802
        self,
        child_graphs: list[int],
        dependency_indices: list[list[int]],
        switch_after_child: list[int],
        norm_pointers: list[int],
        branch_graphs: list[list[int]],
    ) -> _NativeSequentialGraph: ...


class _FrameMap(Protocol):
    base: Tensor


class _FrameWeight(Protocol):
    original: Tensor

    def __getitem__(self, index: int) -> _FrameMap: ...


class _FrameParametrizations(Protocol):
    weight: _FrameWeight


class _Frame(Protocol):
    parametrizations: _FrameParametrizations


class _FrameBlock(Protocol):
    frame: _Frame


class _SwitchStages(Protocol):
    direct_skew_vjp: bool
    _forward_graph: torch.cuda.CUDAGraph
    _backward_graph: torch.cuda.CUDAGraph
    _forward_matrix: Tensor
    _forward_output: Tensor
    _matrix: Tensor
    _output_gradient: Tensor
    _backward_output: Tensor
    _backward_block_buffer: Tensor
    _forward_branch_graphs: tuple[torch.cuda.CUDAGraph, ...]
    _backward_branch_graphs: tuple[torch.cuda.CUDAGraph, ...]


class _AdaptiveBackwardSwitchStages(_SwitchStages, Protocol):
    _backward_block_buffer: Tensor
    _backward_branch_graphs: tuple[torch.cuda.CUDAGraph, ...]


class _HostStages(Protocol):
    _forward_matrix: Tensor
    _forward_graphs: dict[str, torch.cuda.CUDAGraph]
    _forward_outputs: dict[str, Tensor]
    _matrix: Tensor
    _output_gradient: Tensor
    _graphs: dict[str, torch.cuda.CUDAGraph]
    _outputs: dict[str, Tensor]


def cuda_outer_graph_capability() -> tuple[bool, str]:
    """Return whether raw child-graph composition is available."""
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
        return False, "PyTorch raw CUDA Graph access is unavailable"
    cuda_version = torch.version.cuda
    if cuda_version is None:
        return False, "PyTorch has no CUDA runtime"
    runtime_version = tuple(int(part) for part in cuda_version.split(".")[:2])
    if runtime_version < (12, 8):
        return False, "CUDA 12.8+ mixed outer graphs are required"
    try:
        _cuda_home()
    except RuntimeError as error:
        return False, str(error)
    return True, "available"


def _cuda_home() -> Path:
    cuda_version = torch.version.cuda
    if cuda_version is not None:
        major = cuda_version.split(".", maxsplit=1)[0]
        packaged = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / f"cu{major}"
        if (packaged / "bin" / "nvcc").is_file():
            return packaged
    configured = os.environ.get("CUDA_HOME")
    candidates = tuple(
        candidate
        for candidate in (
            Path(configured) if configured else None,
            Path("/usr/local/cuda"),
        )
        if candidate is not None
    )
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").is_file():
            return candidate
    message = "a CUDA toolkit matching the installed PyTorch build is required"
    raise RuntimeError(message)


@lru_cache(maxsize=1)
def _load_outer_graph_extension() -> _OuterGraphExtension:
    previous_cuda_home = os.environ.get("CUDA_HOME")
    os.environ["CUDA_HOME"] = str(_cuda_home())
    try:
        from torch.utils.cpp_extension import load  # noqa: PLC0415

        module = load(
            name="pac_cuda_outer_graph_v2",
            sources=[str(Path(__file__).resolve().parents[2] / "csrc" / "pac_cuda_outer_graph.cu")],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    finally:
        if previous_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = previous_cuda_home
    return cast("_OuterGraphExtension", cast("object", module))


class SequentialCudaGraph:
    """Own one executable CUDA graph made from ordered child graphs."""

    def __init__(self, children: tuple[torch.cuda.CUDAGraph, ...]) -> None:
        if not children:
            message = "sequential CUDA graph requires at least one child"
            raise ValueError(message)
        raw_children: list[int] = []
        for child in children:
            try:
                raw_children.append(child.raw_cuda_graph())
            except RuntimeError as error:
                message = "every child CUDA graph must be captured with keep_graph=True"
                raise ValueError(message) from error
        extension = _load_outer_graph_extension()
        self._children = children
        self._native = extension.SequentialChildGraph(raw_children)

    def replay(self) -> None:
        """Launch on PyTorch's current CUDA stream."""
        stream = torch.cuda.current_stream()
        self._native.launch(stream.cuda_stream)

    def raw_cuda_graph(self) -> int:
        """Return the outer ``cudaGraph_t`` for diagnostic nesting."""
        return self._native.raw_cuda_graph()

    def destroy(self) -> None:
        """Drop the native executable before releasing borrowed children."""
        self.__dict__.pop("_native", None)
        self._children = ()


_SwitchInsertion = tuple[int, Tensor, tuple[torch.cuda.CUDAGraph, ...]]


class SequentialMixedCudaGraph:
    """Sequential children with CUDA 12.8+ SWITCH nodes flattened into the root."""

    def __init__(
        self,
        children: tuple[torch.cuda.CUDAGraph, ...],
        switches: tuple[_SwitchInsertion, ...],
    ) -> None:
        if not children or not switches:
            message = "mixed CUDA graph requires children and at least one switch"
            raise ValueError(message)
        raw_children = [child.raw_cuda_graph() for child in children]
        switch_indices: list[int] = []
        norm_pointers: list[int] = []
        raw_branches: list[list[int]] = []
        for child_index, norm, branches in switches:
            if child_index < 0 or child_index >= len(children):
                message = "mixed CUDA graph has an invalid switch insertion index"
                raise ValueError(message)
            if len(branches) != 10:
                message = "mixed CUDA graph matrix-exp switch requires ten branches"
                raise ValueError(message)
            switch_indices.append(child_index)
            norm_pointers.append(norm.data_ptr())
            raw_branches.append([branch.raw_cuda_graph() for branch in branches])
        extension = _load_outer_graph_extension()
        self._children = children
        self._switches = switches
        self._native = extension.SequentialMixedGraph(
            raw_children,
            switch_indices,
            norm_pointers,
            raw_branches,
        )

    def replay(self) -> None:
        """Launch the flattened graph on PyTorch's current CUDA stream."""
        self._native.launch(torch.cuda.current_stream().cuda_stream)

    def destroy(self) -> None:
        """Drop the native executable before releasing borrowed children."""
        self.__dict__.pop("_native", None)
        self._children = ()
        self._switches = ()


class DependencyMixedCudaGraph:
    """CUDA graph whose child terminals follow an explicit dependency DAG."""

    def __init__(
        self,
        children: tuple[torch.cuda.CUDAGraph, ...],
        dependencies: tuple[tuple[int, ...], ...],
        switches: tuple[_SwitchInsertion, ...],
    ) -> None:
        if not children or len(dependencies) != len(children):
            message = "dependency CUDA graph requires dependencies for every child"
            raise ValueError(message)
        for child_index, child_dependencies in enumerate(dependencies):
            if len(set(child_dependencies)) != len(child_dependencies) or any(
                dependency < 0 or dependency >= child_index for dependency in child_dependencies
            ):
                message = "dependency CUDA graph requires unique earlier child dependencies"
                raise ValueError(message)
        raw_children = [child.raw_cuda_graph() for child in children]
        switch_indices: list[int] = []
        norm_pointers: list[int] = []
        raw_branches: list[list[int]] = []
        for child_index, norm, branches in switches:
            if child_index < 0 or child_index >= len(children):
                message = "dependency CUDA graph has an invalid switch insertion index"
                raise ValueError(message)
            if len(branches) != 10:
                message = "dependency CUDA graph matrix-exp switch requires ten branches"
                raise ValueError(message)
            switch_indices.append(child_index)
            norm_pointers.append(norm.data_ptr())
            raw_branches.append([branch.raw_cuda_graph() for branch in branches])
        if switch_indices != sorted(set(switch_indices)):
            message = "dependency CUDA graph switches must have unique increasing child indices"
            raise ValueError(message)
        extension = _load_outer_graph_extension()
        self._children = children
        self._dependencies = dependencies
        self._switches = switches
        self._native = extension.DependencyMixedGraph(
            raw_children,
            [list(child_dependencies) for child_dependencies in dependencies],
            switch_indices,
            norm_pointers,
            raw_branches,
        )

    def replay(self) -> None:
        """Launch the dependency graph on PyTorch's current CUDA stream."""
        self._native.launch(torch.cuda.current_stream().cuda_stream)

    def destroy(self) -> None:
        """Drop the native executable before releasing borrowed children."""
        self.__dict__.pop("_native", None)
        self._children = ()
        self._dependencies = ()
        self._switches = ()

    def raw_cuda_graph(self) -> int:
        """Return the dependency graph's ``cudaGraph_t`` for diagnostics."""
        return self._native.raw_cuda_graph()


class EFP16ExactSplitOuterGraph:
    """Safe staged and opt-in one-launch paths around an initialized EFP runtime."""

    def __init__(  # noqa: PLR0915
        self,
        runtime: EFP16ExactSplitTraining,
        borrowed_inputs: Tensor,
        borrowed_labels: Tensor,
        *,
        capture_optimizer: bool = True,
    ) -> None:
        self.runtime = runtime
        self._capture_optimizer = capture_optimizer
        runtime._validate_step_tensors(borrowed_inputs, borrowed_labels)
        if capture_optimizer and not runtime.capture_optimizer_tail:
            message = (
                "full EFP16 outer graph requires a capturable optimizer; "
                "use EFP16ExactSplitComputeOuterGraph for an eager optimizer tail"
            )
            raise ValueError(message)
        if runtime.matrix_exp_dispatch != "cuda_switch":
            message = "EFP16 outer graph requires the CUDA SWITCH matrix-exp dispatcher"
            raise ValueError(message)
        if not runtime.specialized_matrix_exp_vjp:
            message = "EFP16 outer graph requires specialized matrix-exp VJPs"
            raise ValueError(message)
        if len(runtime.blocks) != 2 or len(runtime._captured_matrix_exp_vjps) != 2:
            message = "EFP16 outer graph requires exactly two PAC frame blocks"
            raise ValueError(message)
        raw_switches = runtime._captured_matrix_exp_vjps
        parallel_cuda_switch_frames = bool(getattr(runtime, "parallel_cuda_switch_frames", False))
        switches_are_distinct = raw_switches[0] is not raw_switches[1]
        if parallel_cuda_switch_frames != switches_are_distinct:
            message = (
                "parallel EFP16 outer graphs require one distinct CUDA SWITCH runtime "
                "per frame; serial outer graphs require one shared runtime"
            )
            raise ValueError(message)
        required_switch_fields = (
            "_forward_graph",
            "_backward_graph",
            "_forward_matrix",
            "_forward_output",
            "_matrix",
            "_output_gradient",
            "_backward_output",
            "_backward_block_buffer",
            "_forward_branch_graphs",
            "_backward_branch_graphs",
        )
        if any(
            not hasattr(raw_switch, name)
            for raw_switch in raw_switches
            for name in required_switch_fields
        ):
            message = "matrix-exp dispatcher does not expose raw SWITCH graph stages"
            raise TypeError(message)
        switch_runtimes = tuple(
            cast("_SwitchStages", cast("object", raw_switch)) for raw_switch in raw_switches
        )

        self._borrowed_inputs = borrowed_inputs
        self._borrowed_labels = borrowed_labels
        self._input_pointer = borrowed_inputs.data_ptr()
        self._label_pointer = borrowed_labels.data_ptr()
        self._stage_graphs: list[torch.cuda.CUDAGraph] = []
        self._outer_loss: Tensor | None = None
        self._captures_post_optimizer_step = False
        self._parallel_cuda_switch_frames = parallel_cuda_switch_frames
        self._parallel_cuda_switch_lane_dag = bool(
            getattr(runtime, "parallel_cuda_switch_lane_dag", False)
        )
        self._parallel_graph_pools: tuple[torch.cuda._POOL_HANDLE, ...] = ()
        self._parallel_capture_streams: tuple[torch.cuda.Stream, ...] = ()
        dependencies: tuple[tuple[int, ...], ...] = ()
        snapshot = _RuntimeSnapshot.capture(runtime)
        try:
            input_graph = self._capture_input_stage()
            if parallel_cuda_switch_frames:
                children, dependencies, switches, staged_graphs = (
                    self._capture_parallel_compute_children(switch_runtimes)
                )
            else:
                children, switches, staged_graphs = self._capture_compute_children(
                    switch_runtimes[0]
                )
        finally:
            snapshot.restore(runtime)
        self._input_graph = input_graph
        self._staged_graphs = staged_graphs
        if parallel_cuda_switch_frames:
            leased_dependencies = self._prefix_dependencies(dependencies)
            leased_switches = self._shift_switches(switches)
            self._compute_outer = DependencyMixedCudaGraph(children, dependencies, switches)
            self._leased_outer = DependencyMixedCudaGraph(
                (input_graph, *children),
                leased_dependencies,
                leased_switches,
            )
        else:
            leased_switches = self._shift_switches(switches)
            self._compute_outer = SequentialMixedCudaGraph(children, switches)
            self._leased_outer = SequentialMixedCudaGraph(
                (input_graph, *children),
                leased_switches,
            )
        # Preserve the private attribute used by early benchmark scripts while
        # making its semantics the safe, runtime-owned-buffer compute replay.
        self._outer = self._compute_outer
        self._destroyed = False

    def destroy(self) -> None:
        """Destroy composite executables before resetting owned child graphs."""
        if self._destroyed:
            return
        device = self.runtime.static_inputs.device
        torch.cuda.synchronize(device)
        borrowed_graph_ids = {id(self.runtime.forward_backward_graph)}
        owned_stage_graphs = tuple(
            graph
            for graph in self._stage_graphs
            if id(graph) not in borrowed_graph_ids
        )
        # The native composite objects retain raw child graph handles. Drop every
        # alias before resetting a child, otherwise CUDA keeps the child pool alive.
        self._staged_graphs = ()
        outer_roots: list[
            SequentialCudaGraph
            | SequentialMixedCudaGraph
            | DependencyMixedCudaGraph
        ] = []
        root_ids: set[int] = set()
        for name in ("_outer", "_compute_outer", "_leased_outer"):
            root = self.__dict__.get(name)
            if (
                isinstance(
                    root,
                    (
                        SequentialCudaGraph,
                        SequentialMixedCudaGraph,
                        DependencyMixedCudaGraph,
                    ),
                )
                and id(root) not in root_ids
            ):
                root_ids.add(id(root))
                outer_roots.append(root)
        for root in outer_roots:
            root.destroy()
        for name in ("_outer", "_compute_outer", "_leased_outer"):
            self.__dict__.pop(name, None)
        torch.cuda.synchronize(device)
        reset_cuda_graphs(owned_stage_graphs)
        self._stage_graphs.clear()
        self.__dict__.pop("_input_graph", None)
        self._parallel_graph_pools = ()
        self._parallel_capture_streams = ()
        self.__dict__.pop("_borrowed_inputs", None)
        self.__dict__.pop("_borrowed_labels", None)
        self._outer_loss = None
        self._destroyed = True
        torch.cuda.synchronize(device)

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        """Stage arbitrary same-shape tensors, then replay one compute graph."""
        self._validate_step(inputs, labels)
        self._copy_inputs(inputs, labels)
        return self._replay_compute()

    def step_static(self) -> Tensor:
        """Replay using the runtime-owned static buffers without copying inputs."""
        self._validate_runtime_state()
        return self._replay_compute()

    def _replay_compute(self) -> Tensor:
        self._compute_outer.replay()
        self._finish_uncaptured_tail()
        return self._loss()

    def step_leased(self, inputs: Tensor, labels: Tensor) -> Tensor:
        """Replay one graph that copies from the fixed construction addresses."""
        self._validate_leased_replay(inputs, labels)
        self._leased_outer.replay()
        self._finish_uncaptured_tail()
        return self._loss()

    def staged_step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        """Stage arbitrary tensors, then replay compute stages separately."""
        self._validate_step(inputs, labels)
        self._copy_inputs(inputs, labels)
        for graph in self._staged_graphs:
            graph.replay()
        self._finish_uncaptured_tail()
        return self._loss()

    def staged_step_leased(self, inputs: Tensor, labels: Tensor) -> Tensor:
        """Replay the leased input copy and compute stages as separate graphs."""
        self._validate_leased_replay(inputs, labels)
        self._input_graph.replay()
        for graph in self._staged_graphs:
            graph.replay()
        self._finish_uncaptured_tail()
        return self._loss()

    @property
    def static_inputs(self) -> Tensor:
        """Return the runtime-owned input buffer used by ``step_static``."""
        return self.runtime.static_inputs

    @property
    def static_labels(self) -> Tensor:
        """Return the runtime-owned label buffer used by ``step_static``."""
        return self.runtime.static_labels

    @property
    def captures_post_optimizer_step(self) -> bool:
        """Return whether the full outer graph contains the post-optimizer stage."""
        return self._captures_post_optimizer_step

    @property
    def parallel_cuda_switch_frames(self) -> bool:
        """Return whether the outer graph overlaps its two frame lanes."""
        return self._parallel_cuda_switch_frames

    def _finish_uncaptured_tail(self) -> None:
        if self._capture_optimizer:
            return
        self.runtime._optimizer_body()
        if self.runtime.post_optimizer_graph is not None:
            self.runtime.post_optimizer_graph.replay()
        elif not self.runtime._post_step_in_optimizer_graph:
            self.runtime._post_optimizer_step()

    def _loss(self) -> Tensor:
        if self._outer_loss is None:
            message = "EFP16 outer graph has no captured loss buffer"
            raise RuntimeError(message)
        return self._outer_loss.detach()

    @torch.no_grad()
    def _copy_inputs(self, inputs: Tensor, labels: Tensor) -> None:
        self.runtime.static_inputs.copy_(inputs, non_blocking=True)
        self.runtime.static_labels.copy_(labels, non_blocking=True)

    def _validate_step(self, inputs: Tensor, labels: Tensor) -> None:
        self.runtime._validate_step_tensors(inputs, labels)
        self._validate_runtime_state()

    def _validate_leased_replay(self, inputs: Tensor, labels: Tensor) -> None:
        self._validate_step(inputs, labels)
        if inputs.data_ptr() != self._input_pointer or labels.data_ptr() != self._label_pointer:
            message = "leased EFP16 outer replay requires the construction tensor addresses"
            raise ValueError(message)

    def _validate_runtime_state(self) -> None:
        if self._destroyed:
            message = "a destroyed EFP16 outer graph cannot be replayed"
            raise RuntimeError(message)
        self.runtime.activate()
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            message = "EFP16 outer graph requires TF32 to remain disabled"
            raise RuntimeError(message)

    def _capture_input_stage(self) -> torch.cuda.CUDAGraph:
        return self._capture_stage(
            lambda: (
                self.runtime.static_inputs.copy_(self._borrowed_inputs, non_blocking=True),
                self.runtime.static_labels.copy_(self._borrowed_labels, non_blocking=True),
            )
        )

    def _capture_compute_children(
        self, switch: _SwitchStages
    ) -> tuple[
        tuple[torch.cuda.CUDAGraph, ...],
        tuple[_SwitchInsertion, ...],
        tuple[torch.cuda.CUDAGraph, ...],
    ]:
        children: list[torch.cuda.CUDAGraph] = []
        switches: list[_SwitchInsertion] = []
        staged_graphs: list[torch.cuda.CUDAGraph] = []
        for block, override in zip(self.runtime.blocks, self.runtime.frame_overrides, strict=True):
            frame_block = cast("_FrameBlock", cast("object", block))
            prepare_graph = self._capture_stage(
                lambda block=frame_block: self._prepare_forward(block, switch)
            )
            children.append(prepare_graph)
            staged_graphs.append(prepare_graph)
            norm_graph, norm = self._capture_tensor_stage(
                lambda: switch._forward_matrix.abs().sum(dim=-2).max()
            )
            children.append(norm_graph)
            switches.append((len(children) - 1, norm, switch._forward_branch_graphs))
            staged_graphs.append(switch._forward_graph)
            finish_graph = self._capture_stage(
                lambda block=frame_block, override=override: self._finish_forward(
                    block, override, switch
                )
            )
            children.append(finish_graph)
            staged_graphs.append(finish_graph)

        body_graph = self._retained_body_graph()
        self._stage_graphs.append(body_graph)
        children.append(body_graph)
        staged_graphs.append(body_graph)

        for block, override in zip(self.runtime.blocks, self.runtime.frame_overrides, strict=True):
            frame_block = cast("_FrameBlock", cast("object", block))
            prepare_graph = self._capture_stage(
                lambda block=frame_block, override=override: self._prepare_backward(
                    block, override, switch
                )
            )
            children.append(prepare_graph)
            staged_graphs.append(prepare_graph)
            adaptive_switch = cast("_AdaptiveBackwardSwitchStages", switch)
            norm_graph, norm = self._capture_tensor_stage(
                lambda switch=adaptive_switch: self._prepare_backward_switch(switch)
            )
            children.append(norm_graph)
            switches.append(
                (
                    len(children) - 1,
                    norm,
                    adaptive_switch._backward_branch_graphs,
                )
            )
            staged_graphs.append(switch._backward_graph)
            finish_graph = self._capture_stage(
                lambda block=frame_block: self._finish_backward(block, switch)
            )
            children.append(finish_graph)
            staged_graphs.append(finish_graph)

        if self._capture_optimizer:
            # Do not call ``_optimizer_graph_body`` here: that method includes
            # post-processing only when the runtime was originally configured
            # with capture_post_optimizer_step=True. The outer graph owns this
            # complete tail and must preserve model constraints independently of
            # that setup preference.
            optimizer_graph = self._capture_stage(self.runtime._optimizer_body)
            children.append(optimizer_graph)
            staged_graphs.append(optimizer_graph)
            post_optimizer_graph = self._capture_stage(self.runtime._post_optimizer_step)
            children.append(post_optimizer_graph)
            staged_graphs.append(post_optimizer_graph)
            self._captures_post_optimizer_step = True
        return tuple(children), tuple(switches), tuple(staged_graphs)

    def _capture_parallel_compute_children(  # noqa: C901, PLR0915
        self,
        switch_runtimes: tuple[_SwitchStages, ...],
    ) -> tuple[
        tuple[torch.cuda.CUDAGraph, ...],
        tuple[tuple[int, ...], ...],
        tuple[_SwitchInsertion, ...],
        tuple[torch.cuda.CUDAGraph, ...],
    ]:
        if len(switch_runtimes) != len(self.runtime.blocks):
            message = "parallel EFP16 outer graph requires one switch per frame block"
            raise ValueError(message)
        children: list[torch.cuda.CUDAGraph] = []
        dependencies: list[tuple[int, ...]] = []
        switches: list[_SwitchInsertion] = []
        staged_graphs: list[torch.cuda.CUDAGraph] = []
        forward_finishes: list[int] = []
        lane_pools = tuple(torch.cuda.graph_pool_handle() for _ in switch_runtimes)
        lane_streams = tuple(
            torch.cuda.Stream(device=self.runtime.static_inputs.device) for _ in switch_runtimes
        )
        self._parallel_graph_pools = lane_pools
        self._parallel_capture_streams = lane_streams
        lane_local_dependencies = self._parallel_cuda_switch_lane_dag

        lane_values = tuple(
            zip(
                self.runtime.blocks,
                self.runtime.frame_overrides,
                switch_runtimes,
                lane_pools,
                lane_streams,
                strict=True,
            )
        )
        forward_prepare_indices: list[int] = []
        for block, _override, switch, pool, stream in lane_values:
            frame_block = cast("_FrameBlock", cast("object", block))
            prepare_graph = self._capture_stage(
                lambda block=frame_block, switch=switch: self._prepare_forward(block, switch),
                pool=pool,
                stream=stream,
            )
            prepare_index = len(children)
            children.append(prepare_graph)
            # The opt-in DAG keeps roots independent because each frame owns a
            # distinct SWITCH runtime, stream, pool, and tensor set.  The core
            # runtime retains its conservative serial dependency topology.
            dependencies.append(
                ()
                if lane_local_dependencies or not forward_prepare_indices
                else (forward_prepare_indices[-1],)
            )
            staged_graphs.append(prepare_graph)
            forward_prepare_indices.append(prepare_index)

        forward_norm_indices: list[int] = []
        for lane_index, (_block, _override, switch, pool, stream) in enumerate(lane_values):
            norm_graph, norm = self._capture_tensor_stage(
                lambda switch=switch: switch._forward_matrix.abs().sum(dim=-2).max(),
                pool=pool,
                stream=stream,
            )
            norm_index = len(children)
            children.append(norm_graph)
            prepare_dependency = (
                forward_prepare_indices[lane_index]
                if lane_local_dependencies
                else forward_prepare_indices[-1]
            )
            dependencies.append((prepare_dependency,))
            switches.append((norm_index, norm, switch._forward_branch_graphs))
            staged_graphs.append(switch._forward_graph)
            forward_norm_indices.append(norm_index)

        for lane_index, (block, override, switch, pool, stream) in enumerate(lane_values):
            frame_block = cast("_FrameBlock", cast("object", block))
            finish_graph = self._capture_stage(
                lambda block=frame_block, override=override, switch=switch: self._finish_forward(
                    block,
                    override,
                    switch,
                ),
                pool=pool,
                stream=stream,
            )
            finish_index = len(children)
            children.append(finish_graph)
            finish_dependencies = [forward_norm_indices[lane_index]]
            if not lane_local_dependencies and forward_finishes:
                finish_dependencies.append(forward_finishes[-1])
            dependencies.append(tuple(finish_dependencies))
            staged_graphs.append(finish_graph)
            forward_finishes.append(finish_index)

        body_graph = self._retained_body_graph()
        self._stage_graphs.append(body_graph)
        body_index = len(children)
        children.append(body_graph)
        dependencies.append(tuple(forward_finishes))
        staged_graphs.append(body_graph)

        backward_prepare_indices: list[int] = []
        for block, override, switch, pool, stream in lane_values:
            frame_block = cast("_FrameBlock", cast("object", block))
            prepare_graph = self._capture_stage(
                lambda block=frame_block, override=override, switch=switch: self._prepare_backward(
                    block,
                    override,
                    switch,
                ),
                pool=pool,
                stream=stream,
            )
            prepare_index = len(children)
            children.append(prepare_graph)
            dependencies.append(
                (body_index,)
                if lane_local_dependencies or not backward_prepare_indices
                else (backward_prepare_indices[-1],)
            )
            staged_graphs.append(prepare_graph)
            backward_prepare_indices.append(prepare_index)

        backward_norm_indices: list[int] = []
        for lane_index, (_block, _override, switch, pool, stream) in enumerate(lane_values):
            adaptive_switch = cast("_AdaptiveBackwardSwitchStages", switch)
            norm_graph, norm = self._capture_tensor_stage(
                lambda switch=adaptive_switch: self._prepare_backward_switch(switch),
                pool=pool,
                stream=stream,
            )
            norm_index = len(children)
            children.append(norm_graph)
            prepare_dependency = (
                backward_prepare_indices[lane_index]
                if lane_local_dependencies
                else backward_prepare_indices[-1]
            )
            dependencies.append((prepare_dependency,))
            switches.append((norm_index, norm, adaptive_switch._backward_branch_graphs))
            staged_graphs.append(switch._backward_graph)
            backward_norm_indices.append(norm_index)

        backward_finishes: list[int] = []
        for lane_index, (block, _override, switch, pool, stream) in enumerate(lane_values):
            frame_block = cast("_FrameBlock", cast("object", block))
            finish_graph = self._capture_stage(
                lambda block=frame_block, switch=switch: self._finish_backward(block, switch),
                pool=pool,
                stream=stream,
            )
            finish_index = len(children)
            children.append(finish_graph)
            finish_dependencies = [backward_norm_indices[lane_index]]
            if not lane_local_dependencies and backward_finishes:
                finish_dependencies.append(backward_finishes[-1])
            dependencies.append(tuple(finish_dependencies))
            staged_graphs.append(finish_graph)
            backward_finishes.append(finish_index)

        if self._capture_optimizer:
            optimizer_graph = self._capture_stage(self.runtime._optimizer_body)
            optimizer_index = len(children)
            children.append(optimizer_graph)
            dependencies.append(tuple(backward_finishes))
            staged_graphs.append(optimizer_graph)

            post_optimizer_graph = self._capture_stage(self.runtime._post_optimizer_step)
            children.append(post_optimizer_graph)
            dependencies.append((optimizer_index,))
            staged_graphs.append(post_optimizer_graph)
            self._captures_post_optimizer_step = True
        return (
            tuple(children),
            tuple(dependencies),
            tuple(switches),
            tuple(staged_graphs),
        )

    def _retained_body_graph(self) -> torch.cuda.CUDAGraph:
        body_graph = self.runtime.forward_backward_graph
        try:
            body_graph.raw_cuda_graph()
        except RuntimeError as error:
            message = "EFP16 outer graph requires a retained forward/backward graph"
            raise ValueError(message) from error
        if self.runtime.loss is None:
            message = "EFP16 exact-split runtime has no captured loss buffer"
            raise RuntimeError(message)
        self._outer_loss = self.runtime.loss
        # Child-stage capture executes its operations once. Replay the retained
        # body here as setup so the subsequent backward and optimizer captures
        # see the same gradient buffers they will see inside the outer graph.
        body_graph.replay()
        return body_graph

    @staticmethod
    def _shift_switches(
        switches: tuple[_SwitchInsertion, ...],
    ) -> tuple[_SwitchInsertion, ...]:
        return tuple((child_index + 1, norm, branches) for child_index, norm, branches in switches)

    @staticmethod
    def _prefix_dependencies(
        dependencies: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        return (
            (),
            *(
                tuple(dependency + 1 for dependency in child_dependencies)
                if child_dependencies
                else (0,)
                for child_dependencies in dependencies
            ),
        )

    def _capture_stage(
        self,
        operation: Callable[[], object],
        *,
        pool: torch.cuda._POOL_HANDLE | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.no_grad(), torch.cuda.graph(graph, pool=pool, stream=stream):
            operation()
        self._stage_graphs.append(graph)
        return graph

    def _capture_tensor_stage(
        self,
        operation: Callable[[], Tensor],
        *,
        pool: torch.cuda._POOL_HANDLE | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[torch.cuda.CUDAGraph, Tensor]:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.no_grad(), torch.cuda.graph(graph, pool=pool, stream=stream):
            output = operation()
        self._stage_graphs.append(graph)
        return graph, output

    @staticmethod
    def _prepare_forward(block: _FrameBlock, switch: _SwitchStages) -> None:
        original = block.frame.parametrizations.weight.original
        lower = original.tril()
        rows, columns = lower.shape[-2:]
        if rows != columns:
            lower = functional.pad(lower, (0, rows - columns))
        switch._forward_matrix.copy_(lower - lower.mT)

    @staticmethod
    def _finish_forward(block: _FrameBlock, override: Tensor, switch: _SwitchStages) -> None:
        base = block.frame.parametrizations.weight[0].base
        output = base @ switch._forward_output
        columns = block.frame.parametrizations.weight.original.shape[-1]
        override.copy_(output[..., :columns])

    @staticmethod
    def _prepare_backward(block: _FrameBlock, override: Tensor, switch: _SwitchStages) -> None:
        weight = block.frame.parametrizations.weight
        original = weight.original
        lower = original.tril()
        rows, columns = lower.shape[-2:]
        if rows != columns:
            lower = functional.pad(lower, (0, rows - columns))
        switch._matrix.copy_(lower - lower.mT)
        frame_gradient = override.grad
        if frame_gradient is None:
            message = "frame override gradient buffer disappeared"
            raise RuntimeError(message)
        exponential_gradient = weight[0].base.mT @ frame_gradient
        rows, columns = exponential_gradient.shape[-2:]
        if rows != columns:
            exponential_gradient = functional.pad(
                exponential_gradient,
                (0, rows - columns),
            )
        switch._output_gradient.copy_(exponential_gradient)

    @staticmethod
    def _finish_backward(block: _FrameBlock, switch: _SwitchStages) -> None:
        weight = block.frame.parametrizations.weight
        gradient = weight.original.grad
        if gradient is None:
            message = "frame parameter gradient buffer disappeared"
            raise RuntimeError(message)
        skew_gradient = switch._backward_output
        coordinate_gradient = torch.tril(skew_gradient - skew_gradient.mT)
        gradient.copy_(coordinate_gradient[..., : weight.original.shape[-1]])

    @staticmethod
    def _prepare_backward_switch(switch: _AdaptiveBackwardSwitchStages) -> Tensor:
        transposed = switch._matrix.mT
        transposed_column_sums = transposed.abs().sum(dim=-2)
        gradient_column_sums = switch._output_gradient.abs().sum(dim=-2)
        norm = (transposed_column_sums + gradient_column_sums).max()
        if not switch.direct_skew_vjp:
            zeros = torch.zeros_like(transposed)
            block = torch.cat(
                (
                    torch.cat((transposed, switch._output_gradient), dim=1),
                    torch.cat((zeros, transposed), dim=1),
                ),
                dim=0,
            )
            switch._backward_block_buffer.copy_(block)
        return norm


class EFP16ExactSplitComputeOuterGraph(EFP16ExactSplitOuterGraph):
    """Compose forward/backward stages while leaving AdamW and post-step eager."""

    def __init__(
        self,
        runtime: EFP16ExactSplitTraining,
        borrowed_inputs: Tensor,
        borrowed_labels: Tensor,
    ) -> None:
        super().__init__(
            runtime,
            borrowed_inputs,
            borrowed_labels,
            capture_optimizer=False,
        )


class _RuntimeSnapshot:
    def __init__(self, tensors: tuple[tuple[Tensor, Tensor], ...]) -> None:
        self._tensors = tensors

    @classmethod
    def capture(cls, runtime: EFP16ExactSplitTraining) -> _RuntimeSnapshot:
        dynamic: list[Tensor] = list(runtime.model.parameters())
        dynamic.extend(
            parameter.grad for parameter in runtime.model.parameters() if parameter.grad is not None
        )
        dynamic.extend(runtime.frame_overrides)
        dynamic.extend(
            override.grad for override in runtime.frame_overrides if override.grad is not None
        )
        for state in runtime.optimizer.state.values():
            dynamic.extend(value for value in state.values() if isinstance(value, Tensor))
        dynamic.extend((runtime.static_inputs, runtime.static_labels))
        return cls(tuple((tensor, tensor.detach().clone()) for tensor in dynamic))

    @torch.no_grad()
    def restore(self, runtime: EFP16ExactSplitTraining) -> None:
        del runtime
        for destination, source in self._tensors:
            destination.copy_(source)


__all__ = [
    "DependencyMixedCudaGraph",
    "EFP16ExactSplitComputeOuterGraph",
    "EFP16ExactSplitOuterGraph",
    "SequentialCudaGraph",
    "SequentialMixedCudaGraph",
    "cuda_outer_graph_capability",
]

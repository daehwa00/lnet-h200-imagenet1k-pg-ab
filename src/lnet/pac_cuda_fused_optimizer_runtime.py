"""Opt-in exact-split adapters for the two-kernel PAC optimizer tail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import Tensor, nn

from .pac_cuda_fused_optimizer import FusedClipAdamW

if TYPE_CHECKING:
    from .pac_efp16_exact_split_training import EFP16ExactSplitTraining
    from .pac_pa2wp_exact_split_training import PA2WPExactSplitTraining


class _FusedTailOwner(Protocol):
    __dict__: dict[str, object]
    model: nn.Module
    optimizer: torch.optim.AdamW
    grad_clip_norm: float


@dataclass(frozen=True)
class _TailSnapshot:
    tensors: tuple[tuple[Tensor, Tensor], ...]

    @classmethod
    def capture(cls, owner: _FusedTailOwner, tail: FusedClipAdamW) -> _TailSnapshot:
        dynamic = (
            tuple(owner.model.parameters())
            + tuple(cast("Tensor", parameter.grad) for parameter in owner.model.parameters())
            + tail.exp_avgs
            + tail.exp_avg_sqs
            + tail.state_steps
        )
        return cls(tuple((tensor, tensor.detach().clone()) for tensor in dynamic))

    @torch.no_grad()
    def restore(self) -> None:
        for destination, source in self.tensors:
            destination.copy_(source)


def _capture_fused_tail(
    owner: _FusedTailOwner,
    tail: FusedClipAdamW,
) -> torch.cuda.CUDAGraph:
    snapshot = _TailSnapshot.capture(owner, tail)
    stream = torch.cuda.Stream(device=tail.device)
    current = torch.cuda.current_stream(tail.device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        tail.step()
    current.wait_stream(stream)
    torch.cuda.synchronize(tail.device)
    snapshot.restore()

    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(current)
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        tail.step()
    current.wait_stream(stream)
    torch.cuda.synchronize(tail.device)
    snapshot.restore()
    return graph


def _capture_efp_post_step(
    runtime: EFP16ExactSplitTraining,
    tail: FusedClipAdamW,
) -> torch.cuda.CUDAGraph:
    snapshot = _TailSnapshot.capture(runtime, tail)
    stream = torch.cuda.Stream(device=tail.device)
    current = torch.cuda.current_stream(tail.device)
    stream.wait_stream(current)
    with torch.no_grad(), torch.cuda.stream(stream):
        runtime._post_optimizer_step()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    current.wait_stream(stream)
    torch.cuda.synchronize(tail.device)
    snapshot.restore()

    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(current)
    with (
        torch.no_grad(),
        torch.cuda.stream(stream),
        torch.cuda.graph(graph, stream=stream),
    ):
        runtime._post_optimizer_step()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    current.wait_stream(stream)
    torch.cuda.synchronize(tail.device)
    snapshot.restore()
    return graph


def install_efp16_fused_optimizer_tail(
    runtime: EFP16ExactSplitTraining,
) -> FusedClipAdamW:
    """Replace only EFP16's optimizer graph; retain native signed-QR graph."""
    tail = FusedClipAdamW.from_adamw(
        runtime.optimizer,
        max_norm=runtime.grad_clip_norm,
    )
    tail.validate_addresses()
    optimizer_graph = _capture_fused_tail(runtime, tail)
    post_optimizer_graph = _capture_efp_post_step(runtime, tail)
    runtime.optimizer_graph = optimizer_graph
    runtime.post_optimizer_graph = post_optimizer_graph
    runtime.__dict__["_post_step_in_optimizer_graph"] = False
    runtime.__dict__["_fused_optimizer_tail"] = tail
    return tail


def install_outer_graph_fused_optimizer_tail(
    runtime: EFP16ExactSplitTraining,
) -> FusedClipAdamW:
    """Bind the two-kernel tail for a subsequently constructed outer graph.

    The exact-split runtime materializes persistent gradients and AdamW state
    during construction.  Installing here borrows those tensors and replaces
    only the instance-level optimizer body; the outer-graph builder then
    captures this body as an ordinary child graph.  Call this after preparing
    the exact-split runtime and before constructing its outer graph.
    """
    tail = FusedClipAdamW.from_adamw(
        runtime.optimizer,
        max_norm=runtime.grad_clip_norm,
    )
    tail.validate_addresses()
    runtime.__dict__["_optimizer_body"] = tail.step
    runtime.__dict__["_fused_optimizer_tail"] = tail
    return tail


def install_pa2wp_fused_optimizer_tail(
    runtime: PA2WPExactSplitTraining,
) -> FusedClipAdamW:
    """Replace PA2WP's graph-captured AdamW tail with the two-kernel tail."""
    tail = FusedClipAdamW.from_adamw(
        runtime.optimizer,
        max_norm=runtime.grad_clip_norm,
    )
    tail.validate_addresses()
    runtime.__dict__["_optimizer_graph"] = _capture_fused_tail(runtime, tail)
    runtime.__dict__["_fused_optimizer_tail"] = tail
    return tail


__all__ = [
    "install_efp16_fused_optimizer_tail",
    "install_outer_graph_fused_optimizer_tail",
    "install_pa2wp_fused_optimizer_tail",
]

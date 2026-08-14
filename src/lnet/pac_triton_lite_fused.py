from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from .pac_recurrence import RecurrenceBackend


class LitePrlBranch(Protocol):
    def forward_with_backend(self, inputs: Tensor, backend: RecurrenceBackend) -> Tensor: ...


def pac_lite_prl_fused_output(branch: LitePrlBranch, projected: Tensor) -> Tensor:
    return branch.forward_with_backend(projected, "fused_pole_gamma")

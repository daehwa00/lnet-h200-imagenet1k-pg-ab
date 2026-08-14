"""Reader-local removal ablation for ALPHABET."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from torch import Tensor, nn

from .alphabet import Alphabet

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

ReaderLocalVariant = Literal["stem_d1_reader_d1", "stem_d1_reader_none"]
VARIANTS: Final[tuple[ReaderLocalVariant, ...]] = (
    "stem_d1_reader_d1",
    "stem_d1_reader_none",
)


def _set_centered_d1(local: nn.Conv1d) -> None:
    if (
        local.kernel_size != (5,)
        or local.stride != (1,)
        or local.groups != local.in_channels
        or local.in_channels != local.out_channels
    ):
        message = "reader-local ablation requires the canonical stride-1 DWConv5"
        raise ValueError(message)
    local.dilation = (1,)
    local.padding = (2,)


class _ReaderLocalIdentity(nn.Module):
    """Parameter-free channel-first identity compatible with the reader boundary."""

    bias = None

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class ReaderLocalRemovalAlphabet(Alphabet):
    """Fix the input stem at d1 and vary only the reader's local DWConv."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        variant: ReaderLocalVariant,
    ) -> None:
        if variant not in VARIANTS:
            message = f"unsupported reader-local variant: {variant!r}"
            raise ValueError(message)
        super().__init__(config, output_dim)
        if not isinstance(self.stem.local, nn.Conv1d):
            message = "ALPHABET no longer exposes its canonical input DWConv"
            raise TypeError(message)
        _set_centered_d1(self.stem.local)
        if variant == "stem_d1_reader_d1":
            if not isinstance(self.second_local, nn.Conv1d):
                message = "ALPHABET no longer exposes its canonical reader DWConv"
                raise TypeError(message)
            _set_centered_d1(self.second_local)
        else:
            self.second_local = _ReaderLocalIdentity()
        self.reader_local_variant = variant


__all__ = [
    "VARIANTS",
    "ReaderLocalRemovalAlphabet",
    "ReaderLocalVariant",
]

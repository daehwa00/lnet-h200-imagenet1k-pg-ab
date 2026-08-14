"""Run SuperNNova training without its mandatory post-train test prediction.

SuperNNova's CLI always predicts its HDF5 test split after training. Our shared
PLAsTiCC selection artifact intentionally contains only train and validation
objects, so the upstream predictor divides by zero on an empty test split.
The actual training entry point is left unchanged; only optional post-training
prediction and plotting are disabled. Validation metrics are computed later
after explicitly switching the shared validation IDs to the test role.
"""

from __future__ import annotations

from typing import Protocol

# pyright: reportMissingImports=false
from cli import run
from supernnova.data import make_dataset
from supernnova.training import train_rnn


class _Settings(Protocol):
    cyclic: bool


def _train_only(settings: _Settings) -> None:
    """Mirror the upstream training action, stopping before test prediction."""
    make_dataset.resolve_sntypes(settings)
    if settings.cyclic:
        train_rnn.train_cyclic(settings)
    else:
        train_rnn.train(settings)


run.train_rnn_action = _train_only

if __name__ == "__main__":
    run.main()

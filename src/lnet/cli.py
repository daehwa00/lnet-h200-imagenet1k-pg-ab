from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lnet.experiment import (
    SyntheticTaskConfig,
    TrainingConfig,
    make_synthetic_task,
    train_synthetic_model,
)

app = typer.Typer(no_args_is_help=True)


@app.command("train-synthetic")
def train_synthetic(
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 250,
    samples: Annotated[int, typer.Option("--samples", min=8)] = 128,
    validation_samples: Annotated[int, typer.Option("--validation-samples", min=8)] = 32,
    sequence_length: Annotated[int, typer.Option("--sequence-length", min=4)] = 32,
    raw_input_dim: Annotated[int, typer.Option("--raw-input-dim", min=1)] = 2,
    model_dim: Annotated[int, typer.Option("--model-dim", min=1)] = 6,
    output_dim: Annotated[int, typer.Option("--output-dim", min=1)] = 2,
    modes: Annotated[int, typer.Option("--modes", min=1)] = 3,
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=1.0e-4)] = 5.0e-2,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    seed: Annotated[int, typer.Option("--seed")] = 7,
) -> None:
    task = make_synthetic_task(
        SyntheticTaskConfig(
            sample_count=samples,
            validation_count=validation_samples,
            sequence_length=sequence_length,
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=output_dim,
            modes=modes,
            seed=seed,
        ),
    )
    trained = train_synthetic_model(
        task,
        TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
        ),
    )

    table = Table(title="Projected PRL Training Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    summary = trained.summary
    table.add_row("Initial train loss", f"{summary.initial_loss:.6f}")
    table.add_row("Final train loss", f"{summary.final_loss:.6f}")
    table.add_row("Validation loss", f"{summary.validation_loss:.6f}")
    table.add_row("Pole MAE", f"{summary.pole_mae:.6f}")
    table.add_row("True poles", ", ".join(f"{pole:.4f}" for pole in summary.true_poles))
    table.add_row("Learned poles", ", ".join(f"{pole:.4f}" for pole in summary.learned_poles))
    Console().print(table)


if __name__ == "__main__":
    app()

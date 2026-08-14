from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from .pac_external_benchmarks import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    DEFAULT_PAC_MODEL,
    ExternalBenchmarkConfig,
    ExternalModelFamily,
    run_external_benchmarks,
)
from .pac_types import PACDevice  # noqa: TC001 - Typer requires runtime annotations.

if TYPE_CHECKING:
    from .pac_external_tasks import ExternalDatasetName

app = typer.Typer(add_completion=False)
DEFAULT_DATA_ROOT = Path("data/external")
DEFAULT_OUTPUT_ROOT = Path(".omx/results/pac-external-comparisons")


@app.callback(invoke_without_command=True)
def main(
    data_root: Annotated[Path, typer.Option("--data-root")] = DEFAULT_DATA_ROOT,
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
    dataset: Annotated[list[str] | None, typer.Option("--dataset")] = None,
    model: Annotated[list[str] | None, typer.Option("--model")] = None,
    device: Annotated[PACDevice, typer.Option("--device")] = "auto",
    model_dim: Annotated[int, typer.Option("--model-dim")] = 64,
    modes: Annotated[int, typer.Option("--modes")] = 16,
    max_baseline_width: Annotated[int, typer.Option("--max-baseline-width")] = 256,
    parameter_match_tolerance: Annotated[
        float, typer.Option("--parameter-match-tolerance")
    ] = 0.05,
    mitbih_beat_length: Annotated[int, typer.Option("--mitbih-beat-length")] = 256,
    cwru_window_length: Annotated[int, typer.Option("--cwru-window-length")] = 2048,
    forecast_context_length: Annotated[int, typer.Option("--forecast-context-length")] = 96,
    prediction_length: Annotated[int, typer.Option("--prediction-length")] = 96,
    epochs: Annotated[int, typer.Option("--epochs")] = 30,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 64,
    patience: Annotated[int, typer.Option("--patience")] = 8,
    seed: Annotated[list[int] | None, typer.Option("--seed")] = None,
    smoke: Annotated[bool, typer.Option("--smoke")] = False,  # noqa: FBT002
    pac_model: Annotated[str, typer.Option("--pac-model")] = DEFAULT_PAC_MODEL,
) -> None:
    config = ExternalBenchmarkConfig(
        data_root=data_root,
        output_root=output_root,
        datasets=tuple(_dataset(value) for value in (dataset or DEFAULT_DATASETS)),
        models=tuple(_model(value) for value in (model or DEFAULT_MODELS)),
        device=device,
        model_dim=model_dim,
        modes=modes,
        max_baseline_width=max_baseline_width,
        parameter_match_tolerance=parameter_match_tolerance,
        mitbih_beat_length=mitbih_beat_length,
        cwru_window_length=cwru_window_length,
        forecast_context_length=forecast_context_length,
        prediction_length=prediction_length,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seeds=tuple(seed or (7, 11, 19)),
        smoke=smoke,
        latency_warmup=0 if smoke else 5,
        latency_iterations=1 if smoke else 20,
        pac_model=pac_model,
    )
    run_external_benchmarks(config)


def _dataset(value: str) -> ExternalDatasetName:
    if value not in DEFAULT_DATASETS:
        choices = ", ".join(DEFAULT_DATASETS)
        message = f"dataset must be one of: {choices}"
        raise typer.BadParameter(message)
    return value


def _model(value: str) -> ExternalModelFamily:
    if value not in DEFAULT_MODELS:
        choices = ", ".join(DEFAULT_MODELS)
        message = f"model must be one of: {choices}"
        raise typer.BadParameter(message)
    return value


if __name__ == "__main__":
    app()

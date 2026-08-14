from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from .pac_optimization import DEFAULT_OUTPUT_DIR, OptimizationMode, run_optimization

DeviceOption = Literal["auto", "cpu", "cuda"]

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    *,
    mode: Annotated[OptimizationMode, typer.Option("--mode")] = "smoke",
    device: Annotated[DeviceOption, typer.Option("--device")] = "auto",
    output_dir: Annotated[str | None, typer.Option("--output-dir")] = None,
    compare_backends: Annotated[bool, typer.Option("--compare-backends")] = False,
) -> None:
    target_dir = run_optimization(
        mode,
        device,
        DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir),
        compare_backends=compare_backends,
    )
    if compare_backends:
        report = target_dir / "optimization_report.md"
        (target_dir / "advanced_optimization_report.md").write_text(
            report.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    typer.echo(f"Wrote PAC optimization report to {target_dir / 'optimization_report.md'}")


if __name__ == "__main__":
    app()

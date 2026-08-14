from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import typer
from rich.console import Console

from .hybrid_branch_ablation_experiments import config_for_mode, run_hybrid_branch_ablation
from .hybrid_branch_ablation_reports import write_branch_ablation_artifacts

if TYPE_CHECKING:
    from .hybrid_branch_ablation_types import BranchAblationMode
    from .hybrid_experiment_types import DeviceChoice

app = typer.Typer(add_completion=False)
CONSOLE = Console()
DEFAULT_OUTPUT_DIR: Final = Path(".omx/results/hybrid-branch-ablation/full")


@app.command()
def main(
    mode: Annotated[str, typer.Option("--mode")] = "full",
    device: Annotated[str, typer.Option("--device")] = "auto",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_OUTPUT_DIR,
) -> None:
    parsed_mode = _parse_mode(mode)
    parsed_device = _parse_device(device)
    payload = run_hybrid_branch_ablation(config_for_mode(parsed_mode, parsed_device))
    json_path, markdown_path = write_branch_ablation_artifacts(payload, output_dir)
    CONSOLE.print(f"Wrote hybrid branch ablation JSON to {json_path}")
    CONSOLE.print(f"Wrote hybrid branch ablation Markdown to {markdown_path}")
    conclusion = payload["conclusion"]
    if isinstance(conclusion, dict):
        CONSOLE.print(f"conclusion: {conclusion.get('status')}")


def _parse_mode(value: str) -> BranchAblationMode:
    match value:
        case "smoke" | "full":
            return value
        case _:
            message = "mode must be one of: smoke, full"
            CONSOLE.print(message, style="bold red")
            raise typer.Exit(1)


def _parse_device(value: str) -> DeviceChoice:
    match value:
        case "auto" | "cpu" | "cuda":
            return value
        case _:
            message = "device must be one of: auto, cpu, cuda"
            CONSOLE.print(message, style="bold red")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()

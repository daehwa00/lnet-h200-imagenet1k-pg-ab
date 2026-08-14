from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from .pac_experiments import config_for_mode, run_pac_suite
from .pac_reports import write_pac_artifacts

if TYPE_CHECKING:
    from .tapped_prl_followup_schema import JsonRow, JsonValue

PACModeOption = Literal["smoke", "synthetic", "ablation", "ood", "efficiency", "real", "full"]
PACDeviceOption = Literal["auto", "cpu", "cuda"]

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    mode: Annotated[PACModeOption, typer.Option("--mode")] = "smoke",
    device: Annotated[PACDeviceOption, typer.Option("--device")] = "auto",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    target_dir = output_dir or Path(".omx/results/pac-hybrid-prl") / mode
    config = config_for_mode(mode, device, target_dir)
    payload = run_pac_suite(config, mode)
    json_path, markdown_path = write_pac_artifacts(payload, target_dir)
    typer.echo(f"Wrote PAC-Hybrid PRL report JSON to {json_path}")
    typer.echo(f"Wrote PAC-Hybrid PRL report Markdown to {markdown_path}")
    typer.echo(f"conclusion: {_status(payload)}")


def _status(payload: JsonRow) -> JsonValue:
    conclusion = payload.get("conclusion")
    if isinstance(conclusion, dict):
        return conclusion.get("status", "unknown")
    return "unknown"


if __name__ == "__main__":
    app()

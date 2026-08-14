from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer
from rich.console import Console

from .selective_tapped_prl_experiments import run_selective_suite
from .selective_tapped_prl_reports import write_selective_artifacts

if TYPE_CHECKING:
    from .selective_tapped_prl_types import SelectiveMode, SelectiveSuite

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    *,
    suite: Annotated[str, typer.Option("--suite")] = "all",
    mode: Annotated[str, typer.Option("--mode")] = "smoke",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    console = Console()
    try:
        resolved_suite = _parse_suite(suite)
        resolved_mode = _parse_mode(mode)
    except ValueError as error:
        _raise_exit(console, str(error))
    resolved_output_dir = output_dir or (Path(".omx") / "results" / "selective-tapped-prl")
    run = run_selective_suite(suite=resolved_suite, mode=resolved_mode)
    json_path, markdown_path = write_selective_artifacts(run, resolved_output_dir)
    console.print(f"Wrote selective report JSON to {json_path}")
    console.print(f"Wrote selective report Markdown to {markdown_path}")
    for name, section in run.sections.items():
        verdict = section["verdict"]
        status = verdict.get("status") if isinstance(verdict, dict) else "mixed"
        console.print(f"{name}: {status}")


def _parse_suite(value: str) -> SelectiveSuite:
    match value:
        case "all" | "selectivity" | "delay" | "parameter":
            return value
        case _:
            message = "unsupported suite; expected one of: all, selectivity, delay, parameter"
            raise ValueError(message)


def _parse_mode(value: str) -> SelectiveMode:
    match value:
        case "smoke" | "full":
            return value
        case _:
            message = "unsupported mode; expected one of: smoke, full"
            raise ValueError(message)


def _raise_exit(console: Console, message: str) -> NoReturn:
    console.print(message, style="bold red")
    raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()

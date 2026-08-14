from __future__ import annotations

from pathlib import Path

from rich.console import Console

from lnet.benchmarks import render_markdown_report, run_full_benchmark_suite


def main() -> None:
    summary = run_full_benchmark_suite()
    report = render_markdown_report(summary)
    output_path = Path(".omx") / "results" / "benchmark-report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    Console().print(report)


if __name__ == "__main__":
    main()

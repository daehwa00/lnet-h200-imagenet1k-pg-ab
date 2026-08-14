from __future__ import annotations

import argparse
from pathlib import Path

from .pac_tf_confirmatory_report import (
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_P1P2_ROOT,
    DEFAULT_PROTOCOL,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_UNSEEN_ROOT,
    ConfirmatoryReportConfig,
    write_confirmatory_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a partial-safe PAC-TF confirmatory analysis report."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--unseen-root", type=Path, default=DEFAULT_UNSEEN_ROOT)
    parser.add_argument("--p1p2-root", type=Path, default=DEFAULT_P1P2_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--analysis-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_711)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = ConfirmatoryReportConfig(
        protocol_path=arguments.protocol,
        unseen_root=arguments.unseen_root,
        p1p2_root=arguments.p1p2_root,
        evidence_root=arguments.evidence_root,
        output_root=arguments.output_root,
        analysis_registry_path=arguments.analysis_registry,
        bootstrap_seed=arguments.bootstrap_seed,
        bootstrap_iterations=arguments.bootstrap_iterations,
    )
    json_path, markdown_path = write_confirmatory_report(config)
    print(json_path)  # noqa: T201
    print(markdown_path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line interface for the restart-safe balanced HPO campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .pac_balanced_hpo_campaign import (
    EXTERNAL_DATA_ROOT,
    MAX_ATTEMPTS,
    UCR_DATA_ROOT,
    audit_campaign,
    campaign_status,
    code_sha256,
    preflight_manifest,
    run_manifest,
    select_stage1,
    select_stage2,
)
from .pac_balanced_hpo_queue import DEFAULT_ROOT, enqueue_stage1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=(
            "enqueue-stage1",
            "worker",
            "select-stage1",
            "select-stage2",
            "status",
            "audit",
            "code-hash",
            "preflight",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--ucr-data-root", type=Path, default=UCR_DATA_ROOT)
    parser.add_argument("--external-data-root", type=Path, default=EXTERNAL_DATA_ROOT)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "enqueue-stage1":
        payload: object = enqueue_stage1(args.output_root)
    elif args.action == "worker":
        if args.manifest is None:
            message = "--manifest is required for worker"
            raise SystemExit(message)
        summary = run_manifest(
            args.output_root,
            args.manifest,
            device=args.device,
            ucr_data_root=args.ucr_data_root,
            external_data_root=args.external_data_root,
            max_attempts=args.max_attempts,
        )
        payload = asdict(summary)
    elif args.action == "select-stage1":
        payload = select_stage1(args.output_root)
    elif args.action == "select-stage2":
        payload = select_stage2(args.output_root)
    elif args.action == "status":
        payload = campaign_status(args.output_root)
    elif args.action == "audit":
        payload = audit_campaign(args.output_root)
    elif args.action == "preflight":
        if args.manifest is None:
            message = "--manifest is required for preflight"
            raise SystemExit(message)
        payload = preflight_manifest(
            args.manifest,
            device=args.device,
            ucr_data_root=args.ucr_data_root,
            external_data_root=args.external_data_root,
        )
    else:
        payload = {"code_sha256": code_sha256()}
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()

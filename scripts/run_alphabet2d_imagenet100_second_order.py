"""Run the matched one-bank second-order ImageNet-100 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_alphabet2d_imagenet100_nano as nano
import torch

VARIANTS = ("second_order_product", "second_order_pointwise")
nano.VARIANTS = VARIANTS


def _summarize(
    root: Path,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{variant}__seed{seed}.json"
        for variant in VARIANTS
        for seed in nano.SEEDS
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    means = {
        variant: sum(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == variant
        )
        / len(nano.SEEDS)
        for variant in VARIANTS
    }
    paired = [
        next(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == "second_order_product" and row["seed"] == seed
        )
        - next(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == "second_order_pointwise" and row["seed"] == seed
        )
        for seed in nano.SEEDS
    ]
    mean_delta = sum(paired) / len(paired)
    payload = {
        "schema": contract["schema"],
        "mean_final_validation_accuracy": means,
        "paired_product_minus_pointwise": paired,
        "mean_product_minus_pointwise_pp": 100.0 * mean_delta,
        "S2_product_beats_pointwise_1pp": mean_delta >= 0.01,
        "decision": (
            "retain localized pole filtering"
            if mean_delta >= 0.01
            else "second-order gain belongs to the shared feature field"
        ),
        "parameter_counts": {
            variant: sorted(
                {
                    row["parameters"]
                    for row in rows
                    if row["variant"] == variant
                }
            )
            for variant in VARIANTS
        },
    }
    nano._atomic_json(root / "summary.json", payload)  # noqa: SLF001
    return payload


def main() -> None:
    args = nano._parse_args()  # noqa: SLF001
    if not torch.cuda.is_available():
        message = "ImageNet second-order runner requires CUDA"
        raise RuntimeError(message)
    if not set(args.run_seeds) <= set(nano.SEEDS):
        message = "run seeds fall outside the second-order contract"
        raise ValueError(message)
    contract = nano._contract(args)  # noqa: SLF001
    contract["schema"] = "lnet.alphabet2d.imagenet100_second_order.v2"
    contract["evidence_status"] = "confirmatory one-bank second-order gate"
    contract["hypothesis"] = (
        "Localized product-pole Q/R measurements improve over matched "
        "pointwise Q/R measurements of the same shallow feature field."
    )
    contract["representation"] = {
        "encoder": "patch_embed+depthwise_conv+SiLU+RMSNorm",
        "banks": 1,
        "moments": "global Q and complex R at four 2D offsets",
        "head": "affine",
        "synthesis": False,
        "cascade": False,
    }
    contract["source_sha256"]["orchestrator"] = nano._digest(Path(__file__))  # noqa: SLF001
    args.root.mkdir(parents=True, exist_ok=True)
    nano._initialize(args.root, contract)  # noqa: SLF001
    if args.initialize_only:
        return
    device = torch.device("cuda")
    for variant in args.variants:
        for seed in args.run_seeds:
            nano._run_job(  # noqa: SLF001
                args.root,
                contract,
                variant=variant,
                seed=seed,
                data_root=args.data_root,
                workers=args.workers,
                device=device,
            )
    summary = _summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()

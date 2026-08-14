"""Run the matched pole-free versus terminal-pole-reader ImageNet-100 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_alphabet2d_imagenet100_nano as nano
import torch

VARIANTS = ("hybrid_reader",)
nano.VARIANTS = VARIANTS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(nano.SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def _summarize(
    root: Path,
    control_root: Path,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    hybrid_paths = [
        root / "results" / f"hybrid_reader__seed{seed}.json"
        for seed in nano.SEEDS
    ]
    control_paths = [
        control_root / "results" / f"pole_free__seed{seed}.json"
        for seed in nano.SEEDS
    ]
    if not all(path.exists() for path in (*hybrid_paths, *control_paths)):
        return None
    hybrid = [json.loads(path.read_text()) for path in hybrid_paths]
    control = [json.loads(path.read_text()) for path in control_paths]
    paired = [
        hybrid[index]["final_validation"]["accuracy"]
        - control[index]["final_validation"]["accuracy"]
        for index in range(len(nano.SEEDS))
    ]
    mean_delta = sum(paired) / len(paired)
    payload = {
        "schema": contract["schema"],
        "mean_final_validation_accuracy": {
            "hybrid_reader": sum(
                row["final_validation"]["accuracy"] for row in hybrid
            )
            / len(hybrid),
            "pole_free": sum(
                row["final_validation"]["accuracy"] for row in control
            )
            / len(control),
        },
        "paired_hybrid_minus_pole_free": paired,
        "mean_hybrid_minus_pole_free_pp": 100.0 * mean_delta,
        "H1_hybrid_beats_pole_free_1pp": mean_delta >= 0.01,
        "decision": (
            "retain terminal product-pole reader"
            if mean_delta >= 0.01
            else "reject terminal product-pole reader"
        ),
        "parameter_counts": {
            "hybrid_reader": sorted({row["parameters"] for row in hybrid}),
            "pole_free": sorted({row["parameters"] for row in control}),
        },
    }
    nano._atomic_json(root / "summary.json", payload)  # noqa: SLF001
    return payload


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "ImageNet Nano hybrid runner requires CUDA"
        raise RuntimeError(message)
    if not set(args.run_seeds) <= set(nano.SEEDS):
        message = "run seeds fall outside the ImageNet Nano hybrid contract"
        raise ValueError(message)
    contract = nano._contract(args)  # noqa: SLF001
    contract["schema"] = "lnet.alphabet2d.imagenet100_hybrid_reader.v1"
    contract["evidence_status"] = "confirmatory terminal-reader hybrid gate"
    contract["hypothesis"] = (
        "A pole-free backbone with one terminal product-pole reader improves "
        "over the matched all-pointwise pole-free control."
    )
    control_contract = args.control_root / "contract.json"
    if not control_contract.exists():
        message = f"missing pole-free control contract: {control_contract}"
        raise FileNotFoundError(message)
    control_payload = json.loads(control_contract.read_text())
    for key in ("data", "model", "recipe", "seeds"):
        if control_payload.get(key) != contract.get(key):
            message = f"pole-free control contract mismatches hybrid {key}"
            raise RuntimeError(message)
    contract["control"] = {
        "root": str(args.control_root),
        "contract_sha256": nano._digest(control_contract),  # noqa: SLF001
        "variant": "pole_free",
    }
    contract["source_sha256"]["orchestrator"] = nano._digest(Path(__file__))  # noqa: SLF001
    args.root.mkdir(parents=True, exist_ok=True)
    nano._initialize(args.root, contract)  # noqa: SLF001
    if args.initialize_only:
        return
    device = torch.device("cuda")
    for seed in args.run_seeds:
        nano._run_job(  # noqa: SLF001
            args.root,
            contract,
            variant="hybrid_reader",
            seed=seed,
            data_root=args.data_root,
            workers=args.workers,
            device=device,
        )
    summary = _summarize(args.root, args.control_root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()

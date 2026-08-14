"""Run and analyze the Q-only linear campaign on local_gpu."""

# ruff: noqa: S603

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from lnet.pac_q_only_linear_campaign import status

ROOT = Path(".omx/results/pac-q-only-linear-ucr18-local_gpu-20260724")
REFERENCE = Path(
    ".omx/results/pac-writer-reader-capacity-ablation-ucr18-local_gpu-20260724"
)


def _gpu(path: Path) -> str:
    return path.stem.split("-gpu", maxsplit=1)[1].split("-", maxsplit=1)[0]


def _run_once() -> None:
    processes: list[subprocess.Popen[bytes]] = []
    for manifest in sorted((ROOT / "final" / "manifests").glob("*.jsonl")):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = _gpu(manifest)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lnet.pac_q_only_linear_cli",
                    "--stage",
                    "worker",
                    "--device",
                    "cuda",
                    "--output-root",
                    str(ROOT),
                    "--manifest",
                    str(manifest),
                ],
                env=environment,
            )
        )
    return_codes = [process.wait() for process in processes]
    if any(return_codes):
        message = f"Q-only workers exited nonzero: {return_codes}"
        raise RuntimeError(message)


def main() -> None:
    for _ in range(3):
        _run_once()
        if status(ROOT)["final"]["done"]:
            break
    else:
        message = "Q-only campaign did not finish after three attempts"
        raise RuntimeError(message)
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_q_only_linear.py",
            str(ROOT),
            str(REFERENCE),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

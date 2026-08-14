"""Stop the ConvStem PolePyramid campaign when its first seed is clearly nonviable."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import time
from pathlib import Path

import torch


def _history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("history", [])


def _row(history: list[dict[str, float]], epoch: int) -> dict[str, float] | None:
    return next((row for row in history if int(row["epoch"]) == epoch), None)


def _descendants(pid: int) -> list[int]:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    if not children_path.exists():
        return []
    children = [int(value) for value in children_path.read_text().split()]
    return children + [descendant for child in children for descendant in _descendants(child)]


def _terminate_lanes(lanes: tuple[int, ...]) -> None:
    targets = [descendant for lane in lanes for descendant in _descendants(lane)] + list(lanes)
    for pid in reversed(targets):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lane-pids", type=int, nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    pole_path = args.root / "checkpoints" / "convstem_pole__seed501.pt"
    average_path = args.root / "checkpoints" / "convstem_average__seed501.pt"
    verdict_path = args.root / "early-gate.json"
    gates = (
        (30, 0.30, -0.01),
        (50, 0.40, 0.00),
    )
    observations = []
    for epoch, absolute_floor, relative_floor in gates:
        while True:
            pole_row = _row(_history(pole_path), epoch)
            average_row = _row(_history(average_path), epoch)
            if pole_row is not None and average_row is not None:
                break
            if not any(Path(f"/proc/{pid}").exists() for pid in args.lane_pids):
                message = "all campaign lanes ended before the early gate"
                raise SystemExit(message)
            time.sleep(args.poll_seconds)
        pole_accuracy = float(pole_row["validation_accuracy"])
        average_accuracy = float(average_row["validation_accuracy"])
        delta = pole_accuracy - average_accuracy
        passed = pole_accuracy >= absolute_floor and delta >= relative_floor
        observations.append(
            {
                "epoch": epoch,
                "pole_accuracy": pole_accuracy,
                "average_accuracy": average_accuracy,
                "pole_minus_average": delta,
                "absolute_floor": absolute_floor,
                "relative_floor": relative_floor,
                "passed": passed,
            }
        )
        verdict = {
            "schema": "lnet.convstem_pole_pyramid.early_gate.v1",
            "status": "continue" if passed else "stopped_nonviable",
            "observations": observations,
        }
        verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        if not passed:
            _terminate_lanes(tuple(args.lane_pids))
            return


if __name__ == "__main__":
    main()

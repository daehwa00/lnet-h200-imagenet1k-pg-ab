from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lnet.astronomy.quadrature import exponential_gap_bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        asdict(exponential_gap_bias(damping, gap, lag))
        for damping in (1.0 / 3000.0, 0.01, 0.1, 2.0)
        for gap in (60.0, 120.0, 180.0, 200.0)
        for lag in (1.0, 2.0, 4.0)
    ]
    payload = {
        "state_model": "z(t)=exp(-alpha*t), z(0)=1",
        "energy_comparison": (
            "right-endpoint rectangle exp(-2*alpha*H) versus "
            "exact interval mean integral"
        ),
        "lag_comparison": (
            "linear interpolation of endpoint states at H-lag versus "
            "exact exp(-alpha*(H-lag)); value is log10(interpolated/exact)"
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

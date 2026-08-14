#!/usr/bin/env python3
"""Run the compiled PGv2 smoke on the unnormalized Q4 linear variant."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import run_a2d_pgv2_h96_k3_rmsmatch_q4_linear_imagenet100 as q4_linear
import smoke_a2d_pgv2_h96_vector_input as smoke

smoke.RUNNERS["q4-linear"] = q4_linear


if __name__ == "__main__":
    smoke.main()

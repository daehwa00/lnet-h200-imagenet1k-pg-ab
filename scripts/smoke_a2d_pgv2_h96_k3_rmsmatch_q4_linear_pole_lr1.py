#!/usr/bin/env python3
"""Compiled smoke for Q4-linear with full-rate pole geometry."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import run_a2d_pgv2_h96_k3_rmsmatch_q4_linear_pole_lr1_imagenet100 as pole_lr1
import smoke_a2d_pgv2_h96_vector_input as smoke

smoke.RUNNERS["q4-linear-pole-lr1"] = pole_lr1


if __name__ == "__main__":
    smoke.main()

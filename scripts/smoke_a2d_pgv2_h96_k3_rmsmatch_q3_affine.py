#!/usr/bin/env python3
"""Run the established compiled PGv2 smoke on the Q3-only affine variant."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import run_a2d_pgv2_h96_k3_rmsmatch_q3_affine_imagenet100 as q3_affine
import smoke_a2d_pgv2_h96_vector_input as smoke

smoke.RUNNERS["q3-affine"] = q3_affine


if __name__ == "__main__":
    smoke.main()

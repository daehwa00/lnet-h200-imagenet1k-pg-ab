#!/usr/bin/env python3
"""Check the horizontal product scan against its FP32 recurrence reference."""

from __future__ import annotations

# ruff: noqa: EM101, T201, TRY003
import argparse
import json
from typing import cast

import torch
from torch import Tensor

from lnet.pac_product_scan_reference import bidirectional_product_scan_reference
from lnet.pac_triton_bidirectional_product_scan import (
    pac_triton_bidirectional_product_scan,
)

ComplexField = tuple[Tensor, Tensor]
Pole = tuple[Tensor, Tensor, Tensor, Tensor]
BidirectionalField = tuple[Tensor, Tensor, Tensor, Tensor]


def _reference(pole: Pole, source: ComplexField) -> BidirectionalField:
    return bidirectional_product_scan_reference(pole, source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    generator = torch.Generator(device="cuda").manual_seed(933 + arguments.height)
    modes = arguments.modes
    storage_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[arguments.dtype]
    pole = cast(
        "Pole",
        tuple(
            (
                torch.randn((1, 1, 1, modes), generator=generator, device="cuda") * 0.03
            ).requires_grad_()
            for _ in range(4)
        ),
    )
    pole[0].data.fill_(0.82)
    pole[2].data.fill_(0.18)
    float_source = cast(
        "ComplexField",
        tuple(
            torch.randn(
                (arguments.batch, arguments.height, arguments.height, modes),
                generator=generator,
                device="cuda",
            )
            for _ in range(2)
        ),
    )
    source = cast(
        "ComplexField",
        tuple(value.to(storage_dtype).requires_grad_() for value in float_source),
    )
    reference_source = cast(
        "ComplexField",
        tuple(value.detach().float().requires_grad_() for value in source),
    )
    expected = _reference(pole, reference_source)
    actual = pac_triton_bidirectional_product_scan(pole, source)
    maximum_forward = max(
        float(((value - reference).norm() / reference.norm().clamp_min(1.0e-8)).detach())
        for value, reference in zip(actual, expected, strict=True)
    )
    actual_grad_outputs = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(
        expected,
        (*pole, *reference_source),
        tuple(value.float() for value in actual_grad_outputs),
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (*pole, *source),
        actual_grad_outputs,
    )
    maximum_gradient = max(
        float(((value - reference).norm() / reference.norm().clamp_min(1.0e-8)).detach())
        for value, reference in zip(actual_gradients, expected_gradients, strict=True)
    )
    print(
        json.dumps(
            {
                "batch": arguments.batch,
                "dtype": arguments.dtype,
                "height": arguments.height,
                "modes": modes,
                "max_forward_relative_error": maximum_forward,
                "max_gradient_relative_error": maximum_gradient,
            },
            sort_keys=True,
        )
    )
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (*actual, *actual_gradients, *expected, *expected_gradients)
    )
    forward_tolerance = 5.0e-6 if storage_dtype == torch.float32 else 2.0e-2
    gradient_tolerance = 5.0e-5 if storage_dtype == torch.float32 else 5.0e-2
    if not finite or maximum_forward > forward_tolerance or maximum_gradient > gradient_tolerance:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

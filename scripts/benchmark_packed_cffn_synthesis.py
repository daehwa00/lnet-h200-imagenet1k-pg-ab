#!/usr/bin/env python3
"""Compare split and packed synthesis for an arbitrary-width complex CFFN."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.pac_complex_ffn import packed_cartesian_cffn
from lnet.pac_complex_layers import (
    WidelyLinear,
    packed_widely_linear_bias,
    packed_widely_linear_weight,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _weight(projection: WidelyLinear) -> Tensor:
    return packed_widely_linear_weight(
        projection.weight_real,
        projection.weight_imag,
        projection.conjugate_real,
        projection.conjugate_imag,
    )


class _SynthesisCFFN(nn.Module):
    def __init__(self, modes: int, paths: int, hidden: int, output_paths: int) -> None:
        super().__init__()
        self.paths = paths
        self.output_paths = output_paths
        self.input = WidelyLinear(paths, hidden)
        self.output = WidelyLinear(hidden, paths)
        self.scale = nn.Parameter(torch.full((paths,), 1.0e-3))
        self.synthesis_real = nn.Parameter(torch.randn(modes, output_paths, paths) * 0.1)
        self.synthesis_imag = nn.Parameter(torch.randn(modes, output_paths, paths) * 0.1)

    def _mixed(self, real: Tensor, imag: Tensor) -> Tensor:
        source = torch.cat((real, imag), dim=-1)
        hidden = functional.silu(
            functional.linear(
                source,
                _weight(self.input),
                packed_widely_linear_bias(self.input.bias_real, self.input.bias_imag),
            )
        )
        update = functional.linear(
            hidden,
            _weight(self.output),
            packed_widely_linear_bias(self.output.bias_real, self.output.bias_imag),
        )
        scale = torch.cat((self.scale, self.scale)).to(dtype=source.dtype)
        return source + scale * update


class _SplitSynthesisCFFN(_SynthesisCFFN):
    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        mixed_real, mixed_imag = self._mixed(real, imag).split(self.paths, dim=-1)
        return (
            torch.einsum("...mp,mqp->...qm", mixed_real, self.synthesis_real)
            - torch.einsum("...mp,mqp->...qm", mixed_imag, self.synthesis_imag),
            torch.einsum("...mp,mqp->...qm", mixed_real, self.synthesis_imag)
            + torch.einsum("...mp,mqp->...qm", mixed_imag, self.synthesis_real),
        )


class _PackedSynthesisCFFN(_SynthesisCFFN):
    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        output = packed_cartesian_cffn(
            torch.cat((real, imag), dim=-1),
            input_projection=self.input,
            output_projection=self.output,
            residual_scale=self.scale,
            synthesis_real=self.synthesis_real,
            synthesis_imag=self.synthesis_imag,
        )
        output_real, output_imag = output.split(self.output_paths, dim=-1)
        return output_real.transpose(-2, -1), output_imag.transpose(-2, -1)


def _step(
    module: nn.Module,
    runtime: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    real: Tensor,
    imag: Tensor,
) -> None:
    module.zero_grad(set_to_none=True)
    real.grad = None
    imag.grad = None
    output_real, output_imag = runtime(real, imag)
    (output_real.float().square().mean() + output_imag.float().square().mean()).backward()


def _runtime(
    module: nn.Module,
    compile_mode: str,
) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
    if compile_mode == "eager":
        return cast("Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]", module)
    return torch.compile(module, mode=compile_mode, fullgraph=True, dynamic=False)


def _measure(
    step: Callable[[], None],
    *,
    warmups: int,
    iterations: int,
) -> tuple[float, int]:
    for _ in range(warmups):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        step()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples), torch.cuda.max_memory_allocated()


def _benchmark(
    module: nn.Module,
    *,
    state: dict[str, Tensor],
    real: Tensor,
    imag: Tensor,
    compile_mode: str,
    warmups: int,
    iterations: int,
) -> dict[str, float | int]:
    module = module.cuda().train()
    module.load_state_dict(state)
    active_real = real.detach().clone().requires_grad_()
    active_imag = imag.detach().clone().requires_grad_()
    runtime = _runtime(module, compile_mode)
    median_ms, peak_bytes = _measure(
        lambda: _step(module, runtime, active_real, active_imag),
        warmups=warmups,
        iterations=iterations,
    )
    return {"median_step_ms": median_ms, "peak_allocated_bytes": peak_bytes}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--spatial", type=int, nargs="+", default=[28, 14, 7])
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--output-paths", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--require-speedup", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "packed CFFN synthesis benchmark requires CUDA"
        raise RuntimeError(message)
    if min(
        args.batch_size,
        *args.spatial,
        args.modes,
        args.paths,
        args.output_paths,
        args.hidden,
    ) <= 0:
        message = "all CFFN benchmark dimensions must be positive"
        raise ValueError(message)

    torch.manual_seed(501)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32 if args.precision == "float32" else torch.bfloat16
    template = _SplitSynthesisCFFN(
        args.modes,
        args.paths,
        args.hidden,
        args.output_paths,
    )
    state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    stages: dict[str, object] = {}
    passed = True
    for spatial in args.spatial:
        real = torch.randn(
            args.batch_size,
            spatial,
            spatial,
            args.modes,
            args.paths,
            device="cuda",
            dtype=dtype,
        )
        imag = torch.randn_like(real)
        split = _benchmark(
            _SplitSynthesisCFFN(
                args.modes,
                args.paths,
                args.hidden,
                args.output_paths,
            ),
            state=state,
            real=real,
            imag=imag,
            compile_mode=args.compile_mode,
            warmups=args.warmups,
            iterations=args.iterations,
        )
        packed = _benchmark(
            _PackedSynthesisCFFN(
                args.modes,
                args.paths,
                args.hidden,
                args.output_paths,
            ),
            state=state,
            real=real,
            imag=imag,
            compile_mode=args.compile_mode,
            warmups=args.warmups,
            iterations=args.iterations,
        )
        speedup = float(split["median_step_ms"]) / float(packed["median_step_ms"])
        stages[str(spatial)] = {"split": split, "packed": packed, "speedup": speedup}
        passed = passed and speedup >= args.require_speedup

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": args.precision,
        "compile_mode": args.compile_mode,
        "shape": {
            "batch_size": args.batch_size,
            "modes": args.modes,
            "paths": args.paths,
            "output_paths": args.output_paths,
            "hidden": args.hidden,
        },
        "stages": stages,
        "required_speedup": args.require_speedup,
        "pass": passed,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

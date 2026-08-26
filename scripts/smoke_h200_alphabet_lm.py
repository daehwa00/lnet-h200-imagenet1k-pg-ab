#!/usr/bin/env python3
"""Full-context H200 gate for ALPHABET-LM and matched Mamba."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import AlphabetLM, AlphabetLMConfig, FixedComplexPoleMemory1D
from lnet.alphabet_lm_mamba import (
    MAMBA_GIT_COMMIT,
    MambaLM,
    MambaLMConfig,
    build_parameter_matched_mamba,
    trainable_parameters,
)


def _runtime(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "lnet.h200.alphabet_lm.preflight.runtime.v1":
        raise RuntimeError("invalid ALPHABET-LM preflight runtime")
    expected = {
        "WANDB_API_KEY": "0" * 40,
        "WANDB_APP_URL": payload["wandb_app_url"],
        "WANDB_BASE_URL": payload["wandb_base_url"],
        "WANDB_ENTITY": payload["entity"],
        "WANDB_PROJECT": payload["project"],
        "WANDB_GROUP": payload["group"],
        "WANDB_CONSOLE": payload["console"],
    }
    if any(os.environ.get(name) != value for name, value in expected.items()):
        raise RuntimeError("ALPHABET-LM preflight W&B environment changed")
    return cast("dict[str, Any]", payload)


def _initialize_wandb(runtime: dict[str, Any], root: Path) -> Any:
    import wandb  # pyright: ignore[reportMissingImports]

    record = runtime["run"]
    run = wandb.init(
        project=runtime["project"],
        entity=runtime["entity"],
        group=runtime["group"],
        name=record["display_name"],
        id=record["id"],
        tags=record["tags"],
        resume="allow",
        dir=str(root / "wandb"),
        mode="online",
        anonymous="never",
        force=True,
        settings=wandb.Settings(
            disable_code=True,
            console="off",
            disable_git=True,
            disable_job_creation=True,
            init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "30")),
            save_code=False,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_viewer=True,
            x_extra_http_headers={"User-Agent": "Mozilla/5.0 lnet-h200-wandb-client/1"},
            x_save_requirements=False,
        ),
        config={
            **runtime["preflight"],
            "campaign_id": runtime["campaign_id"],
            "campaign_manifest_sha256": runtime["campaign_manifest_sha256"],
            "source_commit": os.environ["H200_EXPECTED_COMMIT"],
            "relay_protocol_version": runtime["relay_protocol_version"],
        },
    )
    if run is None or not run.url:
        raise RuntimeError("required ALPHABET-LM preflight W&B run was not initialized")
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    return run


def _loss(model: nn.Module, tokens: Tensor) -> Tensor:
    logits = model(tokens[:, :-1])
    return functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())


def _compiled_steps(model: nn.Module, tokens: Tensor, repeats: int) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.1, fused=True)
    compiled = cast(
        "nn.Module", torch.compile(model, mode="default", fullgraph=False, dynamic=False)
    )
    samples = []
    loss = tokens.new_zeros((), dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = _loss(compiled, tokens).float()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    if not bool(torch.isfinite(loss)) or not optimizer.state:
        raise RuntimeError("compiled LM smoke produced invalid state")
    seconds = statistics.median(samples[1:] or samples)
    result = {
        "loss": float(loss.detach()),
        "step_seconds": seconds,
        "tokens_per_second": tokens[:, 1:].numel() / seconds,
        "peak_memory_bytes": float(torch.cuda.max_memory_allocated()),
        "optimizer_state_entries": float(len(optimizer.state)),
    }
    del compiled, optimizer
    torch.compiler.reset()
    torch.cuda.empty_cache()
    return result


def _state_roundtrip(model: nn.Module, build: Any, root: Path, name: str) -> float:
    path = root / f"{name}.pt"
    torch.save(model.state_dict(), path)
    restored = build().cuda()
    restored.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
    error = max(
        float((value - restored.state_dict()[key]).abs().max())
        for key, value in model.state_dict().items()
    )
    path.unlink(missing_ok=True)
    if error != 0.0:
        raise RuntimeError(f"{name} checkpoint roundtrip changed parameters")
    return error


@torch.no_grad()
def _precision_probe(config: AlphabetLMConfig) -> float:
    fp32 = FixedComplexPoleMemory1D(
        config.pole_modes, context_length=config.context_length, scan_fp32=True
    ).cuda()
    bf16 = FixedComplexPoleMemory1D(
        config.pole_modes, context_length=config.context_length, scan_fp32=False
    ).cuda()
    bf16.load_state_dict(fp32.state_dict())
    real = torch.randn(2, config.context_length, config.pole_modes, device="cuda").bfloat16()
    imag = torch.randn_like(real)
    expected, actual = fp32(real, imag), bf16(real, imag)
    error = max(
        float((expected[0].float() - actual[0].float()).abs().max()),
        float((expected[1].float() - actual[1].float()).abs().max()),
    )
    del fp32, bf16, real, imag, expected, actual
    torch.cuda.empty_cache()
    return error


def _run(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    if runtime["preflight"] != {
        "alphabet_parameters": 34_794_496,
        "context_length": args.context_length,
        "mamba_parameter_tolerance_fraction": 0.03,
        "microbatch": args.microbatch,
        "models": ["ALPHABET-LM", "parameter-matched-Mamba"],
        "precision": "bfloat16",
        "repeats": args.repeats,
        "seed": 501,
    }:
        raise RuntimeError("ALPHABET-LM preflight arguments differ from the frozen campaign")
    config = AlphabetLMConfig(context_length=args.context_length, scan_fp32=True)
    torch.manual_seed(501)
    alphabet = AlphabetLM(config).cuda()
    alphabet_parameters = trainable_parameters(alphabet)
    torch.manual_seed(501)
    mamba, mamba_parameters, relative_error = build_parameter_matched_mamba(
        alphabet_parameters,
        MambaLMConfig(vocab_size=config.vocab_size, model_width=config.model_width),
    )
    mamba = mamba.cuda()
    tokens = torch.randint(
        config.vocab_size, (args.microbatch, config.context_length + 1), device="cuda"
    )
    precision_error = _precision_probe(config)
    alphabet_metrics = _compiled_steps(alphabet, tokens, args.repeats)
    alphabet_roundtrip = _state_roundtrip(
        alphabet, lambda: AlphabetLM(config), args.root, "alphabet"
    )
    mamba_metrics = _compiled_steps(mamba, tokens, args.repeats)
    mamba_config = mamba.config
    mamba_roundtrip = _state_roundtrip(
        mamba, lambda: MambaLM(mamba_config), args.root, "mamba"
    )
    return {
        "schema": "lnet.alphabet_lm.h200_preflight.v1",
        "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "context_length": config.context_length,
        "microbatch": args.microbatch,
        "mamba_git_commit": MAMBA_GIT_COMMIT,
        "source_commit": os.environ["H200_EXPECTED_COMMIT"],
        "campaign_id": runtime["campaign_id"],
        "alphabet": {
            "K": config.modes, "P": config.pole_modes, "D": config.layers,
            "parameters": alphabet_parameters,
            "checkpoint_max_abs": alphabet_roundtrip,
            **alphabet_metrics,
        },
        "mamba": {
            "layers": mamba_config.layers,
            "parameters": mamba_parameters,
            "relative_parameter_error": relative_error,
            "checkpoint_max_abs": mamba_roundtrip,
            **mamba_metrics,
        },
        "bf16_vs_fp32_recurrence_max_abs": precision_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=2_048)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ALPHABET-LM smoke requires one CUDA GPU")
    gpu = torch.cuda.get_device_name()
    if "H200" not in gpu.upper() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError(f"ALPHABET-LM smoke requires H200, got {gpu}")
    args.root.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(args.runtime)
    (args.root / "wandb").mkdir(parents=True, exist_ok=True)
    wandb_run = _initialize_wandb(runtime, args.root)
    succeeded = False
    try:
        payload = _run(args, runtime)
        wandb_run.log(
            {
                "alphabet/loss": payload["alphabet"]["loss"],
                "alphabet/tokens_per_second": payload["alphabet"]["tokens_per_second"],
                "alphabet/peak_memory_bytes": payload["alphabet"]["peak_memory_bytes"],
                "mamba/loss": payload["mamba"]["loss"],
                "mamba/tokens_per_second": payload["mamba"]["tokens_per_second"],
                "mamba/peak_memory_bytes": payload["mamba"]["peak_memory_bytes"],
                "bf16_vs_fp32_recurrence_max_abs": payload[
                    "bf16_vs_fp32_recurrence_max_abs"
                ],
            },
            step=0,
        )
        for key, value in {
            "status": "passed",
            "alphabet_parameters": payload["alphabet"]["parameters"],
            "mamba_parameters": payload["mamba"]["parameters"],
            "mamba_relative_parameter_error": payload["mamba"]["relative_parameter_error"],
            "source_commit": payload["source_commit"],
        }.items():
            wandb_run.summary[key] = value
        succeeded = True
    finally:
        wandb_run.finish(exit_code=0 if succeeded else 1)
    output = args.root / "preflight.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

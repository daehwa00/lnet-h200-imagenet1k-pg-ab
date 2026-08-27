#!/usr/bin/env python3
"""Frozen-checkpoint probe for direct terminal-pole LM readout."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportMissingImports=false
import argparse
import hashlib
import json
import os
import random
import signal
import threading
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import AlphabetLM, AlphabetLMConfig
from lnet.alphabet_lm_data import TokenBlockDataset, sha256_file
from scripts.train_h200_alphabet_lm_10m import DeterministicBatcher

_STOP_EVENT = threading.Event()


def _request_stop(_signum: int, _frame: object) -> None:
    _STOP_EVENT.set()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


class ModalFusionProbe(nn.Module):
    def __init__(self, input_modes: int, output_modes: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(input_modes, eps=1.0e-6, elementwise_affine=False)
        self.projection = nn.Linear(input_modes, output_modes, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, hidden: Tensor, modal: Tensor) -> Tensor:
        return hidden + self.projection(self.norm(modal))


class FrozenTerminalFeatures:
    def __init__(self, model: AlphabetLM) -> None:
        self.model = model
        self._captured: list[tuple[Tensor, Tensor]] = []
        terminal_memory = cast("nn.Module", model.blocks[-1].memory)
        self._handle = terminal_memory.register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        self._captured.append((real.detach(), imag.detach()))

    @torch.no_grad()
    def __call__(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self._captured.clear()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = self.model.hidden(inputs)
        if len(self._captured) != 1:
            raise RuntimeError("terminal pole-state capture contract changed")
        real, imag = self._captured.pop()
        signed = torch.cat((real, imag), dim=-1)
        energy = torch.log1p(real.float().square() + imag.float().square()).to(hidden.dtype)
        return hidden.detach(), signed, energy

    def close(self) -> None:
        self._handle.remove()


def _loss_sum(logits: Tensor, labels: Tensor, pad_id: int) -> Tensor:
    return functional.cross_entropy(
        logits.flatten(0, 1),
        labels.flatten(),
        ignore_index=pad_id,
        reduction="sum",
    )


def _logits(hidden: Tensor, modal: Tensor, probe: ModalFusionProbe, embedding: Tensor) -> Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return functional.linear(probe(hidden, modal), embedding)


@torch.no_grad()
def _evaluate(
    features: FrozenTerminalFeatures,
    probes: nn.ModuleDict,
    embedding: Tensor,
    dataset: TokenBlockDataset,
    *,
    sequences: int,
    microbatch: int,
    device: torch.device,
) -> dict[str, float]:
    totals = dict.fromkeys(("baseline", *probes.keys()), 0.0)
    total_tokens = 0
    for start in range(0, min(sequences, len(dataset)), microbatch):
        batch = torch.stack(
            [dataset[index] for index in range(start, min(start + microbatch, sequences))]
        ).to(device, non_blocking=True)
        inputs, labels = batch[:, :-1], batch[:, 1:]
        hidden, signed, energy = features(inputs)
        count = int((labels != dataset.manifest.pad_id).sum())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            totals["baseline"] += float(
                _loss_sum(functional.linear(hidden, embedding), labels, dataset.manifest.pad_id)
            )
        modal_inputs = {
            "energy": energy,
            "complex": signed,
            "shuffled": torch.roll(signed, shifts=997, dims=1),
        }
        for name, probe in probes.items():
            totals[name] += float(
                _loss_sum(
                    _logits(hidden, modal_inputs[name], cast("ModalFusionProbe", probe), embedding),
                    labels,
                    dataset.manifest.pad_id,
                )
            )
        total_tokens += count
    return {name: total / total_tokens for name, total in totals.items()}


def _wandb(runtime: dict[str, Any], root: Path, contract: dict[str, Any]) -> Any:
    import wandb

    run = wandb.init(
        project=runtime["project"],
        entity=runtime["entity"],
        group=runtime["group"],
        id=runtime["run"]["id"],
        name=runtime["run"]["display_name"],
        tags=runtime["run"]["tags"],
        resume="allow",
        mode="online",
        anonymous="never",
        dir=str(root / "wandb"),
        config=contract,
        settings=wandb.Settings(
            disable_code=True,
            console="off",
            disable_git=True,
            disable_job_creation=True,
            save_code=False,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_viewer=True,
            x_save_requirements=False,
        ),
    )
    if run is None or not run.url:
        raise RuntimeError("modal probe W&B initialization failed")
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runtime = cast("dict[str, Any]", json.loads(args.runtime.read_text()))
    if runtime.get("schema") != "lnet.kau.alphabet_lm.frozen_modal_probe.runtime.v1":
        raise RuntimeError("invalid frozen modal probe runtime")
    args.root.mkdir(parents=True, exist_ok=True)
    completed = args.root / "completed.json"
    if completed.is_file():
        print(f"ALPHABET_LM_MODAL_PROBE_REUSED={completed}", flush=True)
        return
    training = runtime["training"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    train = TokenBlockDataset(args.train_manifest, verify_sha256=True)
    validation = TokenBlockDataset(args.validation_manifest, verify_sha256=True)
    model = AlphabetLM(AlphabetLMConfig(vocab_size=train.manifest.vocab_size))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(requires_grad=False).eval().to(device)
    probes = nn.ModuleDict(
        {
            "energy": ModalFusionProbe(320, 512),
            "complex": ModalFusionProbe(640, 512),
            "shuffled": ModalFusionProbe(640, 512),
        }
    ).to(device)
    parameters = sum(parameter.numel() for parameter in probes.parameters())
    if parameters != int(runtime["probe_parameters"]):
        raise RuntimeError(f"modal probe parameter contract changed: {parameters}")
    contract = {
        "schema": "lnet.kau.alphabet_lm.frozen_modal_probe.v1",
        "source_commit": os.environ["KAU_EXPECTED_COMMIT"],
        "campaign_manifest_sha256": runtime["campaign_manifest_sha256"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "probe_parameters": parameters,
        "training": training,
        "probes": runtime["probes"],
    }
    contract_sha = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract["contract_sha256"] = contract_sha
    contract_path = args.root / "contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("modal probe contract changed under an existing root")
    _atomic_json(contract_path, contract)
    optimizer = torch.optim.AdamW(
        probes.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    warmup = int(training["warmup_updates"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min(1.0, (update + 1) / max(1, warmup))
    )
    batcher = DeterministicBatcher(train, seed=seed, batch_size=int(training["global_sequences"]))
    state_path = args.root / "checkpoint.pt"
    update = 0
    tokens_seen = 0
    history: list[dict[str, float | int]] = []
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("contract_sha256") != contract_sha:
            raise RuntimeError("modal probe checkpoint contract changed")
        probes.load_state_dict(state["probes"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        batcher.load_state_dict(state["batcher"])
        update = int(state["update"])
        tokens_seen = int(state["tokens_seen"])
        history = state["history"]
    run = _wandb(runtime, args.root, contract)
    feature_extractor = FrozenTerminalFeatures(model)
    target_tokens = int(training["target_tokens"])
    microbatch = int(training["microbatch"])
    started = time.perf_counter()
    signal.signal(signal.SIGTERM, _request_stop)
    torch.cuda.reset_peak_memory_stats()
    while tokens_seen < target_tokens:
        batch = batcher.next()
        valid_tokens = int((batch[:, 1:] != train.manifest.pad_id).sum())
        optimizer.zero_grad(set_to_none=True)
        losses = dict.fromkeys(probes.keys(), 0.0)
        baseline = 0.0
        for micro_cpu in batch.split(microbatch):
            micro = micro_cpu.to(device, non_blocking=True)
            inputs, labels = micro[:, :-1], micro[:, 1:]
            hidden, signed, energy = feature_extractor(inputs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                baseline += float(
                    _loss_sum(
                        functional.linear(hidden, model.embedding.weight),
                        labels,
                        train.manifest.pad_id,
                    )
                )
            modal_inputs = {
                "energy": energy,
                "complex": signed,
                "shuffled": torch.roll(signed, shifts=997, dims=1),
            }
            for name, probe in probes.items():
                loss_sum = _loss_sum(
                    _logits(
                        hidden,
                        modal_inputs[name],
                        cast("ModalFusionProbe", probe),
                        model.embedding.weight,
                    ),
                    labels,
                    train.manifest.pad_id,
                )
                (loss_sum / valid_tokens).backward()
                losses[name] += float(loss_sum.detach())
        torch.nn.utils.clip_grad_norm_(probes.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        update += 1
        tokens_seen += valid_tokens
        elapsed = time.perf_counter() - started
        row: dict[str, float | int] = {
            "update": update,
            "tokens_seen": tokens_seen,
            "baseline_loss": baseline / valid_tokens,
            "energy_loss": losses["energy"] / valid_tokens,
            "complex_loss": losses["complex"] / valid_tokens,
            "shuffled_loss": losses["shuffled"] / valid_tokens,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "tokens_per_second": tokens_seen / max(elapsed, 1.0e-9),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        history.append(row)
        if (
            update % int(training["checkpoint_updates"]) == 0
            or tokens_seen >= target_tokens
            or _STOP_EVENT.is_set()
        ):
            _atomic_torch(
                state_path,
                {
                    "contract_sha256": contract_sha,
                    "probes": probes.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "batcher": batcher.state_dict(),
                    "update": update,
                    "tokens_seen": tokens_seen,
                    "history": history,
                },
            )
            _atomic_json(args.root / "history.json", history)
        run.log(
            {
                "train/baseline_loss": row["baseline_loss"],
                "train/energy_loss": row["energy_loss"],
                "train/complex_loss": row["complex_loss"],
                "train/shuffled_loss": row["shuffled_loss"],
                "training/tokens": tokens_seen,
                "performance/tokens_per_second": row["tokens_per_second"],
                "performance/peak_memory_bytes": row["peak_memory_bytes"],
            },
            step=update,
        )
        print("ALPHABET_LM_MODAL_PROGRESS=" + json.dumps(row, sort_keys=True), flush=True)
        if _STOP_EVENT.is_set():
            run.finish(exit_code=143)
            raise SystemExit(143)
    validation_losses = _evaluate(
        feature_extractor,
        probes,
        model.embedding.weight,
        validation,
        sequences=int(training["validation_sequences"]),
        microbatch=microbatch,
        device=device,
    )
    receipt = {
        "schema": "lnet.kau.alphabet_lm.frozen_modal_probe.receipt.v1",
        "status": "completed",
        "tokens_seen": tokens_seen,
        "updates": update,
        "validation_losses": validation_losses,
        "improvements": {
            name: validation_losses["baseline"] - loss
            for name, loss in validation_losses.items()
            if name != "baseline"
        },
        "probe_parameters": parameters,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
        "wandb_url": run.url,
        "contract_sha256": contract_sha,
    }
    _atomic_json(completed, receipt)
    run.summary.update(receipt)
    run.finish()
    feature_extractor.close()
    print("ALPHABET_LM_MODAL_COMPLETE=" + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

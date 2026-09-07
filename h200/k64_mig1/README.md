# Lee-Wonwoo1: LNet-K64 on one H200 MIG slice

Independent ImageNet-1K confirmation run. This does not restart or retarget the
active K96 campaign, the dormant K128 campaign, or the qlab queue.

## Frozen experiment

- Model: `L-K64-U125`, K=(64,64,64,64), P=(80,80,80,80), depth=(2,2,6,2).
- Parameters with the 1000-class primary Q4 head: 1,562,760.
- Seeds: 501, 509, 521, sequentially on exactly one visible H200 MIG device.
- 100 epochs per seed; physical/effective batch 256; no gradient accumulation.
- Shared matched-baseline recipe: AdamW LR 0.003, weight decay 0.05,
  five-epoch warmup, BF16, 224-pixel inputs and the existing augmentation recipe.
- Prepared compiled training, compile mode `default`, GPU Mixup, yield-first
  input prefetch, eight spawn-context loader workers, persistent validation workers.
- No silent batch reduction or model/recipe fallback on OOM.

## Isolation and launch

Use the official GPU request form while logged in as **Lee-Wonwoo1**. Select
`pytorch/pytorch:latest`, Python, GPU allocation **1**. The existing issue #386
contains a TinyNeXt command and must not be mistaken for this K64 request.
Account registration does not replace a valid approved usage period; the owner
reported approval from September 7 through December 31, 2026.

Repository: `https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab.git`.
Deployment branch: `codex/imagenet1k-k64-mig1-lee`.
Launch with an exact 40-character deployment SHA:

```bash
git fetch origin codex/imagenet1k-k64-mig1-lee && git checkout --detach <deployment-sha> && H200_EXPECTED_COMMIT=<deployment-sha> H200_ALLOW_NOASSERTION_SOURCES=research-only H200_LNET_K64_ONLY=1 H200_OUTPUT_USER=Lee-Wonwoo1 bash h200/run_baselines.sh
```

Dedicated control ref: `control/imagenet1k-k64-mig1-lee`,
`h200/baselines/control.json`; the target SHA must match the deployment checkout.
Run artifacts and control state are under `/app/output/Lee-Wonwoo1/`, partitioned
by immutable source SHA and dataset identity. Each seed retains its checkpoint,
contract, local telemetry, result JSON, and queue status. Resume uses the exact
same source, task and dataset contract.

Before training, the entrypoint runs `scripts/smoke_lnet_k64_mig1.py` in a separate
process. It limits the allocator to min(16 GiB, 90% of the visible device), checks
two BF16 forward/backward/optimizer and train/eval cycles, including a short final
evaluation batch. OOM or nonfinite values abort the launch. The smoke never
writes a production checkpoint. Its log is retained at `k64-mig1-smoke.log`.
The actual H200/MIG smoke is mandatory even when another GPU passed locally.

W&B project: `daehwa/alphabet2d-imagenet1k-h200-baselines`.
The existing secret/IP-scoped relay accepts the three new K64 IDs; all previous
run records, access restrictions and secrets are preserved. No API credentials
are embedded in the checkout or request command.

## Verification

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 python -m pytest -q tests/test_lnet_k64_mig1.py tests/test_lnet_k128_h200_queue.py tests/test_lnet_k96_imagenet1k_runner.py tests/test_h200_baseline_entrypoint.py
bash -n h200/run_baselines.sh
python h200/baselines/generate_wandb_contract.py --check
```

The relay's `npm run check` and deployment dry run must pass before its additive
allowlist update. Local GPU smoke evidence is separate from actual MIG training
evidence; no full ImageNet epoch is claimed until the managed job reports it.

Preparation evidence (2026-09-07): 21 Python tests and 9 relay tests passed;
shell syntax, generated-contract drift check and relay deployment dry run passed.
The local PyTorch 2.9.1+cu128 / RTX PRO 6000 Blackwell smoke passed batch 256 with
7.7163 GiB peak allocated and 9.0879 GiB peak reserved under the 16 GiB limit.
Both train/eval cycles had finite losses, gradients and logits. First compilation
plus the smoke took 372.55 seconds; this is not a training-throughput benchmark.
Relay deployment `c12b7063-e8ff-4b72-b146-a8c80edda9cd` adds exactly three IDs
to the previous 65 allowed records, removing none; its health check passed.

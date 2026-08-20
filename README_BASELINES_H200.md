# H200 ImageNet-1K matched baseline queue

This campaign trains the requested 20 compact vision models from scratch under
one matched 224px recipe. Each model first receives the same three-learning-rate
calibration budget, then runs 300 epochs for seeds 501, 509, and 521.

```text
ParC-Net XS / S
MobileViTv2 0.50 / 0.75 / 1.00
SReT-Tiny
MogaNet-XT
UniConvNet-A
ConvNeXt V2 Atto
EfficientMod-XXS
EMOv2-1M / 2M
MobileOne-S0 / S1
EfficientFormerV2-S0
SwiftFormer-XS
FastViT-T8
TinyNeXt-T / S / M
```

Run on one full H200 with the exact pushed revision:

```bash
H200_ALLOW_NOASSERTION_SOURCES=research-only \
H200_EXPECTED_COMMIT=<40-character-commit> \
bash h200/run_baselines.sh
```

## Queue protocol

- LR calibration: `3e-4`, `1e-3`, `3e-3`; seed 501; 3 epochs each.
- Full training: selected LR; 300 epochs; seeds 501, 509, 521.
- AdamW, weight decay 0.05, warmup plus cosine, Mixup 0.8, label
  smoothing 0.1, RandAugment N2/M9, random erasing 0.25.
- Effective batch size 256, two DataLoader workers per process, prefetch one.
- BF16 on 19 models. UniConvNet-A is explicitly FP32 because its pinned DCNv3
  CUDA kernel does not dispatch BFloat16.
- No pretrained weights and no distillation head.

The entrypoint profiles one, two, and four concurrent workers and selects the
highest measured aggregate throughput. It uses NVIDIA MPS when available; when
MPS is unavailable it benchmarks CUDA's default multi-process mode and keeps
multiple workers only if that is actually faster. Four-worker clients are
capped at 22% of H200 memory and, under MPS, 50% active threads each. Worker
completion is dynamic: a new task starts as soon as any slot becomes free.

Every task owns an immutable contract, atomic checkpoint, result JSON, progress
JSON, log, and durable telemetry spool. A failed model does not terminate other
workers. Relaunching the same command resumes checkpoints and skips validated
results. `run/summary.json` reports each model's three-seed mean and population
standard deviation, and records missing seeds when the campaign is incomplete.

## Implementations and source policy

Nine models use pinned `timm==1.0.26`. Eleven use exact, clean official Git
checkouts declared in `h200/baselines/sources.json`. ParC-Net and EMOv2 have no
declared upstream license (`NOASSERTION`), so their sources are never copied
into this repository or durable output. They are checked out only into the
ephemeral Pod scratch tree, removed on exit, and marked
`redistribution_allowed=false`.

MobileOne and FastViT train their unfused multi-branch graphs. EfficientFormerV2
and SwiftFormer use only their primary, non-distillation heads. EMOv2 dict output
is normalized to logits. UniConvNet-A builds its MIT-licensed DCNv3 extension
from a disposable source copy using the recorded Torch 2.9 compatibility patch,
then must pass an isolated CUDA forward/backward probe before model import.

## Telemetry

Calibration and concurrency preflight never contact W&B. Full runs use a
separate baseline relay with exactly 60 production run IDs and one permanent
canary. Console capture, code, Git metadata, requirements, machine metadata, and
system statistics are disabled. W&B is non-authoritative; a relay failure cannot
stop training and retained local telemetry is replayed after recovery.

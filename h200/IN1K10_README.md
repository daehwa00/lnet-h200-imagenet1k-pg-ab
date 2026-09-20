# ImageNet-1K 10% — single-request campaign

One full H200 request runs all 15 jobs sequentially. The user submits the request;
the repository does not create a GitHub issue or allocate resources itself.

## Scientific protocol

| Item | Fixed setting |
| --- | --- |
| Models | VA-K96, VA-K128, ConvNeXtV2-Atto, TinyViM-S, ParC-Net-S |
| Seeds | 501, 509, 521 |
| Order | All five models at seed 501, then 509, then 521 |
| Initialization | Scratch; no pretrained weights |
| Train subset | Official SimCLR `imagenet_subsets/10percent.txt` |
| Train images | 128,116 images, all 1,000 classes; 128–129 images/class |
| Validation | Original ImageNet-1K 50,000 images |
| Resolution / epochs | 224 × 224 / 100 |
| Batch / updates | One common physical/effective 512 or 1024 selected before training; 250 or125 updates/epoch |
| Optimizer | Main recipe: AdamW, LR 0.003, WD 0.05, five-epoch warmup, cosine |
| Augmentation | Existing Main worker's RandAugment, mixup, random erasing and smoothing |
| Evaluation / reporting | Full validation only at epoch100; final Top-1 mean ± sample SD across three seeds |

The identical public subset is used for every model and seed. It is **not** an
independently sampled subset per seed, nor exactly 10% of each original class.
Training keeps the Main worker's shuffled `drop_last=True` convention (116 images
are omitted from each epoch's shuffled order). Validation never drops images.

Pinned upstream list: [SimCLR](https://github.com/google-research/simclr/blob/be133b38fb21abb77403391dc6db3412edfe92f2/imagenet_subsets/10percent.txt).
SHA256: `6d09de11e7bdaf5b1f3b1f249b6183695f97310cdd20f0c03e7235b6b9392091`.
No ImageNet images are redistributed. The existing dataset is read from
`/app/data/ImageNet-2012/{train,val}/<synset>/*.JPEG`.

## Launch and gates

Run `bash h200/run_in1k10.sh` from a clean checkout of the approved commit.
Request GPU count **7 (one full GPU)**; the launcher rejects a small MIG slice.
Use Python and `pytorch/pytorch:latest` in the platform request form. The launcher
installs a private pinned Python 3.13.11 / Torch 2.9.1+cu128 environment and verifies
the existing native-extension wheel checksum. It copies committed code out of
the scratch checkout before installation or training.

Restart v3 is explicitly authorized to start from scratch and increase the common
batch for all five models. This differs from Main ImageNet batch256 and must not
be described as an otherwise identical training recipe. LR stays0.003; no linear
LR scaling is silently introduced. Total updates are25,000 at batch512, or12,500
at batch1024, for all models/seeds. All still see128,000 shuffled images/epoch.

A bounded pre-training probe compares old NumPy-pipe input with lossless memfd
input, chooses2/4/8 workers within detected CPU/RAM limits (or1 if constrained),
and measures VA-K96 GPU-resident compute at batch256/512/1024. The smaller batch
within5% of the fastest successful expanded-batch compute result is preferred.
All five models must then pass full-input preflight at that common batch before
any production run. An actual batch1024 CUDA OOM permits a common fallback512;
an insufficient batch-aware host-RAM budget also permits fallback512. Worker
counts are clamped again for the selected batch, and train workers are closed
before final validation to avoid overlapping two persistent worker pools.
Other errors stop the campaign. Probe weights are discarded. This is a bounded
VA-K96-guided choice with an all-model memory gate, not a proof of globally
optimal throughput for all architectures. Input profiles include a repeated
baseline to expose cache/order effects.

Each job then follows:
two-update/512-validation-image GPU preflight → owner-side W&B readiness
acknowledgment → full training → final W&B metrics acknowledgment → next job.
There is no external checkpoint upload, restore, or backup-failure stop condition.
Preflight weights are never reused. Local resume is possible only if a checkpoint
is visible in this same output root and its strict contract still matches.
Failed setup, nonfinite evaluation, missing data, changed source/recipe, or failed
gates stop the campaign rather than silently skipping a model or changing batch.

The selected implementation uses BF16, fused AdamW, GPU mixup/prefetch,
channels-last, torch.compile, persistent loaders, bounded compilation threads,
and the numerically checked VA spill cleanup. Failed state-cache/split-backward
prototypes are excluded. Lossless Linux memfd collation transfers small FD
descriptors rather than147MiB FP32 batch payloads through a pipe. Standard
DataLoader pin-memory threads materialize parent-owned tensors asynchronously.
The actual transformed FP32 values and labels are preserved bit-for-bit; there
is no FP16/uint8 compression or changed augmentation. memfd consumes RAM under
the cgroup limit without relying on the small /dev/shm mount. Actual H200 speed
and real pinned allocation remain to be measured after submission. GPU
clocks/occupancy are not hardcoded.

Host loader wait, pin/materialization time, and CUDA step spans are logged each
epoch. These overlap and must not be summed as GPU idle time. CUDA spans include
launch gaps and first-epoch compilation effects, not solely kernel busy time.
The guard also records CPU quota/throttling and sampled GPU utilization/clock.

## Logs, W&B, stop, and checkpoint recovery

The independent stdlib watchdog starts before environment setup. Console output
goes both to the platform and an authenticated relay; progress is sampled every
20 optimizer updates. A qlab observer records logs and forwards real metrics to
15 stable runs in `daehwa/alphabet2d-imagenet1k-10pct`, group
`simclr10-finaleval-v3`. Run IDs differ from both interrupted prior attempts, so new
epoch-zero training does not get appended to either old learning curve. The GPU
process receives no W&B API key.

Owner commands (key supplied through a private file, never through the form):

```bash
python scripts/in1k10_control.py --secrets /private/owner-secrets.json status
python scripts/in1k10_control.py --secrets /private/owner-secrets.json stop
python scripts/in1k10_control.py --secrets /private/owner-secrets.json force
```

`stop` requests an epoch-boundary save, with a 15-minute grace period; `force`
kills the supervised process group at the next successful control poll. A
disconnected controller/observer triggers a stop after 10 minutes; no observed
training progress triggers one after 20 minutes. These controls do not guarantee
Kubernetes resource release or save a checkpoint after a whole-container SIGKILL.
Do not use `release`/`arm` while an old container is still running.

Checkpoint files are atomically saved after each validated epoch under
`/app/output/in1k10-finaleval-v3`, **on H200 only**. The user explicitly disabled
external weight backups after build802 failed with RemoteDisconnected during
its epoch10 checkpoint upload. The watchdog stopped that run after epoch12.
The observer now transfers only logs and metrics, never model/optimizer weights.
The campaign still waits for final metrics to reach W&B before the next model;
that acknowledgment is a small control message, not a checkpoint transfer.

The H200 platform mounts `/app/output` **per request**. Saving there alone does
not make old files visible in the next request. There is **no cross-request
automatic checkpoint recovery** in v3. If a container is returned and its volume
is not exposed again, restarting from scratch may be required. Logs and W&B
metrics survive independently, but they cannot reconstruct model weights.
Same-volume resume is epoch-boundary recovery, not a promise of bitwise-identical
augmentation RNG after a worker-process restart.

The existing H200 egress-IP allowlist is reused without widening it. Enrollment
uses a runtime-generated agent token, and privileged stop/ack commands require a
separate owner token. Shared H200 NAT is an enrollment trust boundary, **not
cryptographic per-person identity**. Keep owner secrets and controller files
private. External baseline sources are cloned at pinned revisions listed in
`in1k10_sources.json`, not redistributed; their upstream terms still apply.

## Verification and limits

Verified before submission: pinned list/counts, all five CPU model constructions
and parameter counts, collation value preservation, watchdog forced termination,
control/artifact tests, Python/shell/TypeScript checks, and an isolated live
transport/stop/W&B canary. That small v1 canary did not establish reliable transfer
of actual H200 checkpoints; v2 deliberately removes weight transfer. Additional
tests verify all15 jobs advance without backup files and the guard never calls
artifact endpoints. The canary is explicitly not a training run.

Not yet verified: GPU preflight on the requested H200 image, H200-specific speed,
100-epoch completion, or all-15-run duration. Platform time limits may require a
new user-submitted request; without a visible old volume this means another fresh
start. No automatic resubmission or external checkpoint recovery.

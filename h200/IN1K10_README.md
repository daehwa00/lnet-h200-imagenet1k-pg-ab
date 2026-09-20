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
| Batch / updates | Physical and effective 256; 500/epoch; 50,000 total |
| Optimizer | Main recipe: AdamW, LR 0.003, WD 0.05, five-epoch warmup, cosine |
| Augmentation | Existing Main worker's RandAugment, mixup, random erasing and smoothing |
| Reporting | Final epoch Top-1, mean ± sample SD across three seeds |

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

Restart v2 is explicitly authorized to start from scratch. Each job follows:
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
prototypes are excluded. NumPy pipe collation avoids large Torch shared-memory
IPC buffers without changing image transforms. Worker count is two, or zero for
a small cgroup memory allowance. This is a conservative configuration, **not a
claim of measured H200-optimal throughput**; H200 preflight and actual timing
remain to be measured after submission. GPU clocks/occupancy are not hardcoded.

## Logs, W&B, stop, and checkpoint recovery

The independent stdlib watchdog starts before environment setup. Console output
goes both to the platform and an authenticated relay; progress is sampled every
20 optimizer updates. A qlab observer records logs and forwards real metrics to
15 stable runs in `daehwa/alphabet2d-imagenet1k-10pct`, group
`simclr10-local-v2`. Run IDs differ from the interrupted v1 attempt, so new
epoch-zero training does not get appended to the old 12-epoch curve. The GPU
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
`/app/output/in1k10-local-v2`, **on H200 only**. The user explicitly disabled
external weight backups after build802 failed with RemoteDisconnected during
its epoch10 checkpoint upload. The watchdog stopped that run after epoch12.
The observer now transfers only logs and metrics, never model/optimizer weights.
The campaign still waits for final metrics to reach W&B before the next model;
that acknowledgment is a small control message, not a checkpoint transfer.

The H200 platform mounts `/app/output` **per request**. Saving there alone does
not make old files visible in the next request. There is **no cross-request
automatic checkpoint recovery** in v2. If a container is returned and its volume
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

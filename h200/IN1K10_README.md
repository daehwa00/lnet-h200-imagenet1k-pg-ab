# ImageNet-1K 10% — single-request campaign

The v4 request runs the remaining14 jobs, starting with VA-K128seed501. Completed
VA-K96seed501 is retained from v3 (100epochs/25,000updates/55.904% Top1), verified
against W&B and recorded in in1k10_completed_v3.json. It is not trained again.
The user submits the request;
the repository does not create a GitHub issue or allocate resources itself.

## Scientific protocol

| Item | Fixed setting |
| --- | --- |
| Models | VA-K96, VA-K128, ConvNeXtV2-Atto, TinyViM-S, ParC-Net-S |
| Seeds | 501, 509, 521 |
| Order | Remaining four models at seed501, then all five at509, then521 |
| Initialization | Scratch; no pretrained weights |
| Train subset | Official SimCLR `imagenet_subsets/10percent.txt` |
| Train images | 128,116 images, all 1,000 classes; 128–129 images/class |
| Validation | Original ImageNet-1K 50,000 images |
| Resolution / epochs | 224 × 224 / 100 |
| Batch / updates | Fixed physical/effective512,250updates/epoch,25,000total; workers8 |
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

The user previously approved a common larger batch for all five models. v3
measured256/512/1024 and selected512/workers8/memfd; all five GPU preflights passed.
v4 freezes that measured profile, so the carried K96 result and remaining jobs
keep the same scientific recipe. It does not rerun a broad batch sweep or silently
select1024. It checks hostRAM/CPU suitability before launch. Main ImageNet used
batch256: this enlarged-batch panel must not be described as identical to Main.
LR stays0.003; no linear scaling. Train workers close before final validation.

Each job then follows:
two-update/512-validation-image GPU preflight → owner-side W&B readiness
acknowledgment → full training → final W&B metrics acknowledgment → next job.
There is no external checkpoint upload, restore, or backup-failure stop condition.
Preflight weights are never reused. Local resume is possible only if a checkpoint
is visible in this same output root and its strict contract still matches.
Failed setup, nonfinite evaluation, missing data or changed source/recipe stop
the campaign rather than silently skipping a model or changing batch. Network
readiness/final-metrics gates wait without a deadline and honor STOP: the container
is retained with local results, but the GPU can idle until telemetry recovers.

The selected implementation uses BF16, fused AdamW, GPU mixup/prefetch,
channels-last, torch.compile, persistent loaders, bounded compilation threads,
and the numerically checked VA spill cleanup. Failed state-cache/split-backward
prototypes are excluded. Lossless Linux memfd collation transfers small FD
descriptors rather than147MiB FP32 batch payloads through a pipe. Standard
DataLoader pin-memory threads materialize parent-owned tensors asynchronously.
The actual transformed FP32 values and labels are preserved bit-for-bit; there
is no FP16/uint8 compression or changed augmentation. memfd consumes RAM under
the cgroup limit without relying on the small /dev/shm mount. v3 H200 K96 training
reached approximately1,400images/s; v4's new container still needs its preflight. GPU
clocks/occupancy are not hardcoded.

Host loader wait, pin/materialization time, and CUDA step spans are logged each
epoch. These overlap and must not be summed as GPU idle time. CUDA spans include
launch gaps and first-epoch compilation effects, not solely kernel busy time.
The guard also records CPU quota/throttling and sampled GPU utilization/clock.

## Logs, W&B, stop, and checkpoint recovery

The independent stdlib watchdog starts before environment setup. Console output
goes both to the platform and an authenticated relay; progress is sampled every
20 optimizer updates. A qlab observer records logs and forwards real metrics to
14 new stable runs in `daehwa/alphabet2d-imagenet1k-10pct`, group
`simclr10-networkfix-v4`. The carried K96 result retains its original v3 run URL.
New run IDs differ from interrupted prior attempts, so new
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
disconnected controller/observer now causes a warning, NOT a training stop.
The last valid control response is retained. The former empty-response→timestamp0
false observer-timeout path is removed. Genuine local compute stagnation still
stops after20minutes; intentional W&B/metrics waiting is exempt. Signals and local
STOP remain active. Remote stop cannot be delivered while the network is down.
Control/events poll15seconds, observer snapshots15seconds/heartbeat60seconds,
with bounded exponential backoff honoring Retry-After. This reduces this campaign's
nominal relay load from about69k to19k requests/day (other clients add usage).
Every stop reason and HTTP status/redacted error excerpt is printed to platform
stdout and a local diagnostic file, independently of relay delivery.
These controls do not guarantee
Kubernetes resource release or save a checkpoint after a whole-container SIGKILL.
Do not use `release`/`arm` while an old container is still running.

Checkpoint files are atomically saved after each validated epoch under
`/app/output/in1k10-networkfix-v4`, **on H200 only**. The user explicitly disabled
external weight backups after build802 failed with RemoteDisconnected during
its epoch10 checkpoint upload. The watchdog stopped that run after epoch12.
The observer now transfers only logs and metrics, never model/optimizer weights.
The campaign still waits for final metrics to reach W&B before the next model;
that acknowledgment is a small control message, not a checkpoint transfer.

The H200 platform mounts `/app/output` **per request**. Saving there alone does
not make old files visible in the next request. There is **no cross-request
automatic checkpoint recovery** in v4. If a container is returned and its volume
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
tests verify the14remaining jobs start atK128 and preserve the prior K96 endpoint,
the guard never calls artifact endpoints, long telemetry outages do not kill
progressing training, real stalls/explicit stop still stop, and HTTP429 readiness
and final acknowledgments can recover. Tests use bounded CPU fixtures/mocked HTTP,
not fake production results. Training math/worker/kernels are unchanged from v3.

Not yet verified: GPU preflight in the new requested container or remaining14-run
completion. Platform time limits may require a
new user-submitted request; without a visible old volume this means another fresh
start. No automatic resubmission or external checkpoint recovery.

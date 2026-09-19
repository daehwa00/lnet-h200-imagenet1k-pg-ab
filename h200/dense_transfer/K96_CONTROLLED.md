# K96 COCO controlled campaign — preparation, not submitted

Fixed ImageNet-1K seed501/epoch100 K96 checkpoint; downstream seeds501/509/521.
Original E/ReIm interface, channels192 at strides4/8/16/32, shared Mask R-CNN/FPN
source unchanged. Physical2/effective16, BF16, 800/max1333, AdamW1e-4 WD.05,
warmup1000, drops8/11, 12epochs (88,716updates each). Workers0 avoids previous
IPC failures. It is an execution difference, not a claim of bitwise-equivalent
data-loader trajectories. GPU full-input validation is required for each seed.

## Controls, independent of GitHub

Run `bash h200/run_k96_coco_controlled.sh` only after privately provisioning:

- H200_AGENT_TOKEN: ingest/control-read credential, not owner credential.
- H200_K96_CHECKPOINT: visible fixed pretrained checkpoint file.
- H200_K96_CHECKPOINT_SHA256: verified digest of that exact file.

Never paste transport credentials into GitHub source, issues or prefilled URLs.
The existing request form has no confirmed private-secret injection field;
that delivery channel must be resolved before a submission-ready request.

The deployed dedicated Worker is `lnet-h200-k96-control-v1`. The old W&B relay
is untouched. Independent stdlib guard checks stop every5seconds, sends SIGTERM
for cooperative checkpointing, then SIGKILL after120seconds if needed. A
continuous control outage of600seconds also requests stop. None of this can
guarantee Kubernetes resource release or survival of the entire container.

Full stdout/stderr is retained at `/app/output/daehwa00/dense-k96-coco-v1/console.log`.
HTTP event batches include progress and cgroup status. The remote tail holds
2000 events; an owner receiver must run to retain the entire streamed history.
During a network outage local logs are retained and the byte cursor advances
only on successful upload. Delivery is at-least-once (duplicates are possible).
Hard OOM/SIGKILL of the entire container can prevent the final report.
The control's stop latch is permanent for this campaign; a stopped campaign is
not silently rearmed or resumed.

## Current evidence / blockers

- K96 ImageNet seed501 W&B run4466b7509edbf9a5 completed100epochs, Top1 72.374%.
- That run's accessible W&B files/artifacts contain history, not model weights.
- No K96 seed501 checkpoint was located in the checked local stores or GitHub
  Releases. Do not substitute K128, initialize randomly, or claim readiness.
- K96 constructor count3,253,224 and rectangular64x96 feature outputs verified
  on CPU. Checkpoint-exact identity cannot be verified until weights are found.
- Guard tests cover log/exit capture, refusal before launch, forced stop.
- No H200 job was submitted or launched by this preparation.

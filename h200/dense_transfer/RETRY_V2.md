# VA COCO seed501 retry v2 —2026-09-15

User submits the GPU request; this branch does not create/close an issue.

## Why build749 failed

Issue440 ended2026-09-15 20:13KST after `rebuild_storage_fd → resource_sharer → SocketClient.connect` raised FileNotFoundError. The pin-memory thread then exited and the DataLoader aborted. This directly identifies a missing multiprocessing IPC socket, **not a missing COCO image or an observed CUDA/NVML error**.

The supplied log does not establish why the socket disappeared. Worker termination/OOM and externally cleaned temporary paths are possibilities; no exit/memory counters were recorded. Missing source snippets for scratch-path frames are also not conclusive proof of deletion. Do not claim a confirmed root cause beyond the missing socket.

## Changes

- Archive the exact Git commit into `/app/output/daehwa00/dense-transfer/code/<SHA>` and execute there, not under `/app/scratch/input`.
- Use a short private per-job TMPDIR under persistent output, before spawning Python, DataLoader, compiler or W&B service processes.
- Use `file_system` CPU tensor sharing in parent AND spawn workers, avoiding the observed file-descriptor/resource_sharer transfer path. Keep pin-memory acceleration.
- Use2 workers/prefetch1 only with sufficient cgroup RAM and shared memory. Otherwise use an explicitly reported0-worker profile with no multiprocessing tensor IPC. Recipe/seed/batch are unchanged; worker RNG trajectory is not claimed identical to v1.
- Include the validated4090 hybrid optimization: fused sequential FP32 odd-grid vertical recurrence, grouped finite checks, and shape-adaptive dense tiles. No resolution/batch/epoch/pole reduction. The previous4090 paired benchmark was1.56x; H200 speed is unmeasured and not guaranteed to match.
- H200 runs its own full-input finite train/eval/exact-restore/continuation gate before production.

## W&B

Project: `daehwa/alphabet2d-dense-transfer`.
Production ID: `e3ec67e73d61028d`, name `H200-VA-COCO-ft501-v2` (created when the user starts the job).
Permanent non-training canary ID: `a3f5d2756fbc5127`.

Long-lived SDK logging: loss and optimizer progress every20updates; health heartbeat every60seconds; checkpoint position every save; final AP or failure diagnostics. Console progress is limited to1000-update intervals to avoid overflowing the GitHub result comment. Local telemetry JSONL and SDK spool also remain under output.

Logging canary must succeed before training/data preparation proceeds. No visible checkpoint uses native SDK `resume=never`, preventing accidental reuse of an old W&B history. A visible checkpoint uses `resume=allow`. A custom Python resume query was rejected by the strict traced relay and removed in favor of the native SDK protocol; the native collision guard was verified.

Relay base: `https://lnet-h200-baseline-relay-v1.gpupulse-monitor.workers.dev/dense-v2`.
It is an additive prefix with only the dense project and two IDs. The original69 IDs/routes, real key, and egress-IP secret are unchanged. Real credentials are never in this checkout or request command. Existing H200 IP restrictions intentionally reject control/qlab/KAU egress; the restriction was not widened for testing.

Protocol end-to-end validation used a loopback-only local Worker and an authorized local credential held in a private temporary environment file, then removed. Actual W&B history/summary and `resume=never` collision refusal passed. The cloud H200-origin handshake remains a mandatory deployment-time check because no direct H200 session is available before submission. The relay remains IP-scoped/non-authoritative: shared-NAT clients cannot be cryptographically distinguished.

## Recovery and storage

Output: `/app/output/daehwa00/dense-transfer/runs/coco-va_k128-seed501-v2`.
Looks first for its own checkpoint, then the v1 seed501 checkpoint. If visible, validates model/data/recipe and migrates only execution/loader/output metadata, preserving model/optimizer/scheduler/saved RNG/progress and retaining the original file. Wrong seed/model/recipe is refused. If absent, prints that previous data is **not visible**, then starts fresh from the approved ImageNet checkpoint; absence does not prove deletion.

Provider volumes have previously been invisible across jobs. `/app/output` storage alone does not guarantee that a new container can recover the previous checkpoint. No trained checkpoint is publicly uploaded by this change.

Failure logs include last observed and last saved progress, checkpoint existence, cgroupv1/v2 memory counters, worker exit states, TMP/code existence, storage and GPU diagnostics. SIGTERM during training requests an update-boundary checkpoint. A hard SIGKILL cannot record its own postmortem; the prior checkpoint/heartbeat remains the evidence.

## Verification

-43 Python tests passed (engine/data/readiness plus17 v2 loader, recovery, logging-guard and orchestration tests).
-15 relay tests, generated type checks, TypeScript checks, drift checks and deployment dry run passed.
-Live root/dense health checks passed; original69 allowed IDs matched the prior deployed source.
-Local Worker→real W&B canary and native existing-run refusal passed without changing deployed access restrictions.
-Actual H200 GPU/IPC runtime proof is intentionally automatic after user submission; no H200 training start is claimed during preparation.

Cloud relay prior version: `8f9c3818-ab72-4712-a661-a54a83a7838d`.
Current additive version: `0e21fe5a-5ffe-4592-9e92-058ea9fdf9ba`.
Do not redeploy an older baseline entrypoint while the dense prefix is in use.

## Request fields

- User: `daehwa00`
- Repository: `https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab.git`
- Branch: `codex/h200-dense-retry-20260915`
- Entrypoint: `h200/run_dense_va_coco_v2.sh`, with exact deployment SHA and H200_EXPECTED_COMMIT.
- Image: `pytorch/pytorch:latest`; language Python; allocation7 (one whole GPU).

No API key is required in the issue form. The user must click Create.

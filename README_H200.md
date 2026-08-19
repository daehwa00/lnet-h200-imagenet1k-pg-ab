# One-H200 ImageNet-1K PG ablation

This frozen deployment bundle runs one controlled comparison:

- `PGv2-H96-K3-RMSMatch-PGNoWD`
- `PGv2-H96-K3-RMSMatch-NoPG-All`

Only the Stage 1–3 phase-gated mode residual differs. The seed, K3 RMS-matched
reader, D4 scan, Q descriptor, path projection, carry, classifier, optimizer,
schedule, augmentation, precision, and batch size stay fixed.

Run on Aerodrone with a whole H200 (`MIG=7`):

```bash
H200_EXPECTED_COMMIT=<pushed-40-character-commit> bash h200/run.sh
```

The entrypoint rejects a different commit or dirty checkout, regenerates
nothing, installs Python `3.13.11` from the hash-locked environment, verifies
the exact H200/CUDA/package versions, and disables persistent DataLoader workers
so sampler, augmentation-worker, process, CUDA, and mixup RNG streams remain
continuous across epoch-boundary resume. Bitwise CUDA-kernel identity is not
claimed because the measured compile path retains cuDNN benchmarking. It validates exactly
1,281,167 train images, 50,000 validation images, and 1,000 matching classes,
then freezes the sorted relative-path/size/content digest in the immutable experiment
contract. Both models must pass a production-batch (`256`) compiled BF16
forward/backward smoke before training.

The output path is derived from the v3 campaign-manifest digest:

```text
/app/output/daehwa00/lnet-h200-imagenet1k-pg-ab-v3-<manifest-prefix>/
```

Local artifacts are authoritative. Every epoch is committed in this order:
scheduler state, fsync'd atomic checkpoint, fsync'd telemetry spool,
`H200_PROGRESS_JSON`, and finally an optional W&B upload. A W&B init, log, or
finish failure emits `H200_WANDB_DEGRADED_JSON` and cannot terminate training.
On resume, checkpoint history repairs a missing spool record; an existing final
result is never retrained and can still be reconciled to W&B.

W&B is a best-effort, non-authoritative mirror. The H200 process always sends a
40-zero placeholder to the IP/run/protocol-scoped Cloudflare relay. The real
key and H200 egress allowlist exist only as Worker secrets. Console capture is
off, so the relay rejects `output.log`; code, Git metadata, requirements,
machine metadata, and system statistics are also disabled. The first W&B run is
created only after epoch 1 has a durable checkpoint. Run URLs are printed as
`WANDB_RUN_URL=https://wandb.ai/...` when the mirror is available.

`h200/campaign.json` is the single editable campaign contract. Verify generated
files with `python h200/generate_campaign.py --check`. The vendored relay,
deployment instructions, and permanent non-production canary live under
`cloudflare/relay/`. Because the H200 service provides no per-job secret, users
behind the same NAT cannot be cryptographically distinguished; rate limiting,
exact request allowlists, immutable run metadata, and non-authoritative W&B
semantics bound that residual risk.

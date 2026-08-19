# One-H200 ImageNet-1K PG ablation

This frozen deployment bundle runs one controlled comparison:

- `PGv2-H96-K3-RMSMatch-PGNoWD`
- `PGv2-H96-K3-RMSMatch-NoPG-All`

Only the Stage 1–3 phase-gated mode residual differs. The seed, K3 RMS-matched
reader, D4 scan, Q descriptor, path projection, carry, classifier, optimizer,
schedule, augmentation, precision, and batch size stay fixed.

Run on Aerodrone with a whole H200 (`MIG=7`):

```bash
bash h200/run.sh
```

The entrypoint checks the Hopper GPU, exact Python/Torch/Triton environment,
the 1000-class dataset layout, compiled BF16 forward/backward for both models,
finite gradients, and strict checkpoint reload before training. Checkpoints,
compiler caches, W&B files, and final JSON results are under
`/app/output/daehwa00/lnet-h200-imagenet1k-pg-ab-v1`.

W&B is mandatory and online. The H200 job sends a non-secret dummy credential
to the IP- and run-scoped Cloudflare relay configured in `h200/run.sh`. The
relay replaces it with the real `WANDB_API_KEY`, which exists only as an
encrypted Worker secret. The public repository, issue body, job command, and
H200 environment never contain the account credential. Run URLs are printed as
`WANDB_RUN_URL=https://wandb.ai/...`.

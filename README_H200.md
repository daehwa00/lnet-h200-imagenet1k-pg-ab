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

W&B is mandatory and online. If the provider securely mounts `WANDB_API_KEY`,
the runs go to that account. Otherwise W&B anonymous online mode is used and
the claimable run URLs are printed as `WANDB_RUN_URL=...`; anonymous runs must
be claimed within W&B's retention window.

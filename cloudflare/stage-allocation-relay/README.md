# H200 ImageNet-100 stage-allocation W&B relay

This is the dedicated non-authoritative W&B relay for the 20-model, three-seed
ImageNet-1K baseline campaign. It is generated from
`h200/stage_allocation/campaign.json` and the already traced W&B 0.22.3 protocol.
Exactly 60 production run IDs and one permanent canary ID are accepted.

The real W&B key and H200 egress allowlist are Worker secrets:

```sh
cd cloudflare/baseline-relay
npx wrangler secret put WANDB_API_KEY
npx wrangler secret put ALLOWED_EGRESS_IPS
npm run check
npm run deploy:dry
npx wrangler deploy
```

Regenerate after editing the baseline campaign:

```sh
python h200/stage_allocation/generate_contract.py
```

W&B remains a best-effort mirror. Durable local checkpoints, progress JSON,
and result JSON are the experiment authority. Shared-NAT clients cannot be
cryptographically distinguished because the H200 provider exposes no per-job
secret channel.

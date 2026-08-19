# H200 W&B secret relay

This Worker is a narrow credential relay for the exact W&B `0.22.3` protocol
recorded for the v3 H200 campaign. `h200/campaign.json` is the only editable
campaign contract; run `python h200/generate_campaign.py` from the repository
root after changing it, and commit every generated file together.

The production W&B key and the H200 egress IP allowlist are Worker secrets. They
must never be added to the manifest, Wrangler config, source, issue form, or
logs:

```sh
cd cloudflare/relay
npx wrangler secret put WANDB_API_KEY
npx wrangler secret put ALLOWED_EGRESS_IPS
npm run check
npm run deploy:dry
npx wrangler deploy
```

After deployment, run the permanent protocol canary with Python containing the
manifest-pinned `wandb==0.22.3`:

```sh
PYTHONPATH=src python cloudflare/relay/canary.py
```

The canary uses only the public zero placeholder credential, must print
`H200_RELAY_CANARY_OK=...`, and must never be deleted from W&B. A production
run ID must never be used for relay probing.

The Worker overwrites client authentication, fixes the only upstream origin to
`https://api.wandb.ai`, rejects redirects and compressed input, caps request
bodies, checks exact GraphQL hashes and variable shapes, and allows only
`config.yaml`, `wandb-history.jsonl`, and `wandb-summary.json`. W&B console
capture must remain off, so `output.log` is deliberately rejected. Rejections
are logged as structured reason codes without request bodies, headers, source
addresses, or credentials. Validation deployments use 100% log and trace
sampling.

The permanent canary ID in the manifest is not a production run ID and must not
be deleted. Use it for a full SDK canary before changing an allowlist. Because
the H200 service exposes no per-job secret channel, the IP gate cannot
distinguish two users behind the same NAT. Rate limiting and immutable run
metadata reduce abuse but do not make W&B authoritative; local signed campaign
contracts, checkpoints, progress records, and result files remain the source of
truth for research results.

# K96 seed521 input-only resume — dormant until recovery is verified

Do not stop issue384 merely because this branch exists. At preparation time,
W&B exposed only config.yaml and wandb-summary.json, not a model checkpoint.
The managed service documents server-side /app/output storage but does not
guarantee that a new Pod can read a previous Pod's output. Earlier recovery
attempts encountered a missing /app/output/daehwa00 directory.

`h200/run_k96_input_resume.sh` therefore requires the original checkpoint,
contract, dataset manifest, and pinned Python environment to be readable at the
exact recorded paths. Missing artifacts cause an immediate refusal; there is
no from-scratch fallback and no package/driver install or upgrade.

After that gate succeeds:

1. Verify the exact original worker/runner fingerprints, seed521, epoch100
   horizon, batch256, workers8, optimizer/LR/precision contract and checkpoint.
2. Compare old and GPU-Mixup/yield-first input pipelines on H200 using isolated
   copies of the model/optimizer; do not log benchmark steps as training.
3. Enable the new pipeline only with at least a 5% measured gain, otherwise use
   the original pipeline. Copy the old checkpoint into a separate namespace.
4. Resume the same W&B run `26f109ae80471f86` through the existing relay. No new
   allowlist, token publication, or duplicate completed seed is needed.
5. Use separate owner control `control/imagenet1k-k96-input-resume`. A stop
   marker is checked after the epoch checkpoint; supervisor grace is 40 minutes,
   with forced termination fallback if the process is unresponsive. The last
   atomic checkpoint remains authoritative in that fallback.

The currently running issue384 control branch is not changed by this branch.
The new control defaults to stop. Do not arm it or submit a new managed request
until a recoverable checkpoint path is confirmed. CPU contract tests do not
substitute for proving cross-Pod storage access or an H200 speed measurement.

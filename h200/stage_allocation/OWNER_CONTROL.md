# H200 owner kill switch

The stage-allocation launcher is supervised from the first expensive setup step
through the final variant. It polls the dedicated
`control/imagenet100-stage-allocation` branch every 15 seconds. The mutable
control record is deliberately separate from the immutable deployment branch.

## Stop a running request

1. Open [`control.json` on the control branch](https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab/edit/control/imagenet100-stage-allocation/h200/stage_allocation/control.json).
2. Increase `generation` by one.
3. Change `action` to `stop`, set `updated_at` to the current timezone-aware
   timestamp, and describe the reason.
4. Commit directly to `control/imagenet100-stage-allocation`.

The healthy path exits after the next completed optimizer update. The supervisor
allows two minutes for that cooperative exit, then sends `SIGTERM` to the whole
process group and escalates to `SIGKILL` after another 30 seconds. It never
replaces an epoch checkpoint with partial-epoch state.

## Resume or submit another request

Increase `generation` again, change `action` back to `run`, update `updated_at`,
and keep `target_commit` equal to the exact deployment commit used by
`H200_EXPECTED_COMMIT`. A stale or same-generation `run` cannot clear a recorded
stop.

No W&B key, GitHub token, or other secret belongs in this file. Repository write
permission is the authorization boundary.

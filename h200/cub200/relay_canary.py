"""CPU-only scoped W&B connection/log canary; never starts training."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from cub200_run_ids import run_id
import wandb

os.environ['WANDB_BASE_URL'] = 'https://lnet-h200-baseline-relay-v1.gpupulse-monitor.workers.dev/cub-v1'
os.environ['WANDB_API_KEY'] = '0' * 40  # SDK placeholder, never an upstream key.
campaign = 'cub200-scratch-100ep-lee-v1'
run = wandb.init(entity='daehwa', project='alphabet2d-cub200', group=campaign,
    id=run_id(campaign, 'connectivity'), name='connectivity', resume='allow', mode='online',
    settings=wandb.Settings(init_timeout=45, console='off', disable_code=True,
                           disable_git=True, x_disable_stats=True))
assert not run.offline
run.log({'connectivity_ok': 1, 'training_started': False})
print('CUB_RELAY_CANARY_OK run=' + run.id, flush=True)
run.finish()

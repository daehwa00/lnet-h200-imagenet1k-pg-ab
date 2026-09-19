"""Explicit isolated live CPU canary; never used by automated unit tests."""
import argparse
import json
from pathlib import Path
import sys
import time
import wandb
from cub200_supervisor import supervise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--confirm-live-canary',action='store_true');args=p.parse_args()
    if not args.confirm_live_canary:p.error('Explicit --confirm-live-canary required')
    args.root.mkdir(parents=True,exist_ok=False)
    run=wandb.init(entity='daehwa',project='alphabet2d-cub200',group='infrastructure-canary',
        name='CPU-only-stop-log-canary',job_type='infrastructure-test',
        tags=['not-a-scientific-result','no-gpu'],mode='online',
        settings=wandb.Settings(init_timeout=45,console='off',disable_code=True,disable_git=True,x_disable_stats=True))
    (args.root/'run.json').write_text(json.dumps({'id':run.id,'path':run.path,'url':run.url}))
    run.log({'canary/started':1})
    command=[sys.executable,'-u','-c','import time; print("CUB_CANARY_LOG_VISIBLE",flush=True); time.sleep(180)']
    ok=supervise(command,args.root,run,args.root/'STOP',grace=3,stall=150)
    run.summary['canary_stop_received']=not ok and (args.root/'STOP').exists()
    run.finish()


if __name__=='__main__':main()

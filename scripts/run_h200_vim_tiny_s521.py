"""Matched original Vim-Tiny seed521 on one whole H200; no pretrained weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import run_h200_baseline_worker as worker

MODEL_KEY='vision_mamba_tiny'
GROUP='h200-imagenet1k-moga-emo-100ep-s501-v2'
RUN_ID=hashlib.sha256(f'{GROUP}:{MODEL_KEY}:seed521'.encode()).hexdigest()[:16]
TAGS=['H200','ImageNet-1K','Vim-Tiny','100ep','seed521']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    os.environ.update(H200_BASELINE_RUN_ID=RUN_ID,H200_BASELINE_DISPLAY_NAME='H200-Vim-Tiny-s521',
        H200_BASELINE_TAGS_JSON=json.dumps(TAGS),WANDB_GROUP=GROUP,
        LNET_GPU_MIXUP='1',LNET_YIELD_BEFORE_FETCH='1',LNET_RETAIN_EXTERNAL_IMPORTS='1',
        LNET_LOADER_CONTEXT='spawn',LNET_VALIDATION_PERSISTENT='1',LNET_BATCH_PROGRESS='1')
    os.environ.pop('H200_BASELINE_TORCH_COMPILE_MODE',None)
    out=a.output_root/('preflight' if a.preflight else 'seed_521')
    ck=out/'checkpoint.pt'
    task=worker.BaselineTask(phase='preflight' if a.preflight else 'full',model_key=MODEL_KEY,
        seed=521,learning_rate=.003,epochs=1 if a.preflight else 100,
        data_root=a.data_root,output_dir=out,result_path=out/'result.json',checkpoint_path=ck,
        source_root=a.source_root,batch_size=256,workers=8,
        wandb_mode='disabled' if a.preflight else 'online',resume=ck.exists(),
        max_steps=2 if a.preflight else None)
    print('H200_VIM_STAGE='+('preflight' if a.preflight else 'training'),flush=True)
    worker.run_task(task)


if __name__=='__main__': main()

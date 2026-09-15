"""H200 VA COCO orchestration with exact state recovery and live W&B logging."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

import torch
import run_dense_transfer as training
import validate_dense_transfer_runtime as validator
from dense_transfer.engine import atomic_json, atomic_torch, contract_hash
from dense_transfer.telemetry import Telemetry, diagnostics, canary, relay_rejection_probe


def loader_profile(memory_limit: int|None, shm_free: int) -> dict:
    # Avoid multiplying the large COCO annotation graph in small containers.
    workers=2 if (memory_limit is None or memory_limit>=24*1024**3) and shm_free>=1024**3 else 0
    return {'workers':workers,'prefetch_factor':1,'sharing_strategy':'file_system',
            'pin_memory':True,'reason':'bounded multiprocessing' if workers else 'single-process fallback: low RAM or shared memory; no tensor IPC'}


def read_limits():
    limit=0 # Unknown limit uses single-process loading rather than guessing RAM.
    for path in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value=Path(path).read_text().strip();limit=int(value) if value.isdigit() else None;break
        except OSError:continue
    try:shm=shutil.disk_usage('/dev/shm').free
    except OSError:shm=0
    return loader_profile(limit,shm)


def verify_science(old,new):
    old=json.loads(json.dumps(old));new=json.loads(json.dumps(new))
    allowed={'workers','output_root','pin_memory','prefetch_factor','sharing_strategy'}
    for key in ('optimizer','dataset','checkpoint_sha256','checkpoint_prevalidation','runtime_versions','schema'):
        if old.get(key)!=new.get(key):raise ValueError('resume changed '+key)
    a={k:v for k,v in old['settings'].items() if k not in allowed}
    b={k:v for k,v in new['settings'].items() if k not in allowed}
    if a!=b:raise ValueError('resume changed scientific settings')
    if (b['task'],b['model_key'],b['seed'])!=('coco','va_k128',501):raise ValueError('wrong prior run identity')
    for key in ('models','metrics'):
        if old['runtime_source_sha256'][key]!=new['runtime_source_sha256'][key]:raise ValueError('resume changed head/evaluation')
    a,b=copy.deepcopy(old['backbone']),copy.deepcopy(new['backbone'])
    for value in (a,b):
        for key in ('dense_kernel_sha256','dense_performance'):
            value.get('pretrained_provenance',{}).pop(key,None)
    if a!=b:raise ValueError('resume changed pretrained backbone')


def recover(source: Path,ready: dict,output: Path):
    payload=torch.load(source,map_location='cpu',weights_only=False)
    if payload.get('contract_hash')!=contract_hash(payload['contract']):raise ValueError('old checkpoint binding invalid')
    if not 0<=payload['progress']['optimizer_updates']<=88716:raise ValueError('invalid old update budget')
    if payload['contract_hash']==ready['contract_hash']:return source,payload['progress']
    verify_science(payload['contract'],ready['contract'])
    archive=output/'recovery';archive.mkdir(parents=True,exist_ok=True)
    preserved=archive/'source.pt'
    if preserved.exists():raise FileExistsError('recovery archive already exists; inspect before retry')
    shutil.copy2(source,preserved)
    migrated={**payload,'contract':ready['contract'],'contract_hash':ready['contract_hash']}
    with preserved.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    migrated['recovery']={'source':str(source),'source_sha256':sha,'checkpoint_state_preserved':True,
                         'loader_rng_trajectory_exact':False,'reason':'optimized execution + safe loader profile'}
    target=archive/'migrated.pt';atomic_torch(target,migrated)
    verify=torch.load(target,map_location='cpu',weights_only=False)
    for key in payload:
        if key not in ('contract','contract_hash') and not validator._same(payload[key],verify[key]):
            raise ValueError('state changed during recovery: '+key)
    atomic_json(archive/'receipt.json',migrated['recovery'])
    return target,payload['progress']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--canary-only',action='store_true')
    args=parser.parse_args()
    if args.canary_only:
        try:canary()
        except Exception as error:
            payload={'error_type':type(error).__name__,'message':str(error)[:1500],
                     'relay_probe':relay_rejection_probe(),'diagnostics':diagnostics()}
            atomic_json(args.root/'logging-preflight-failure.json',payload)
            print('H200_DENSE_LOGGING_PREFLIGHT_FAILED='+json.dumps(payload),flush=True)
            raise
        return
    if args.data_root is None:parser.error('--data-root is required')
    root=args.root;output=root/'runs/coco-va_k128-seed501-v2';output.mkdir(parents=True,exist_ok=True)
    profile=read_limits();print('H200_DENSE_LOADER='+json.dumps(profile),flush=True)
    atomic_json(output/'startup-diagnostics.json',diagnostics())
    sources=[output/'checkpoints/last.pt',root/'runs/coco-va_k128-seed501/checkpoints/last.pt']
    source=next((p for p in sources if p.is_file()),None)
    print('H200_DENSE_RECOVERY='+json.dumps({'visible_checkpoint':None if source is None else str(source),
          'action':'validate-and-resume' if source else 'fresh: previous job checkpoint is not visible; not proof of deletion'}),flush=True)
    common=['--task','coco','--model','va_k128','--seed','501','--checkpoint',str(root/'checkpoints/va_k128_seed501_ep100.pt'),
            '--data-root',str(args.data_root),'--output-root',str(output),'--physical-batch-size','2','--effective-batch-size','16',
            '--workers',str(profile['workers']),'--prefetch-factor','1','--sharing-strategy','file_system']
    telemetry=None;success=False;final=None
    try:
        telemetry=Telemetry(output,has_checkpoint=source is not None,loader_profile=profile)
        telemetry.stage('validating')
        validation=validator.run_validation(validator.parser().parse_args(common+['--mode','smoke','--max-probe-updates','2']))
        print('H200_DENSE_VALIDATION='+json.dumps({'status':validation['status'],'evidence':{k:v for k,v in validation['evidence'].items() if k!='source_dataset_pretrain_hashes'}}),flush=True)
        if validation['status']!='ready' or not validation['evidence'].get('continuation_parity_verified'):
            raise RuntimeError('H200 full-input runtime validation failed; training not started')
        ready=json.loads((output/'queue/readiness.json').read_text())
        resume=[]
        if source:
            source,progress=recover(source,ready,output);resume=['--resume',str(source)]
            print('H200_DENSE_RESUME='+json.dumps(progress),flush=True)
        gc.collect();torch.cuda.empty_cache()
        telemetry.stage('training')
        train_args=training.parser().parse_args(common+['--mode','train','--confirm-training','DENSE_TRANSFER_TRAIN']+resume)
        train_args.telemetry=telemetry
        result=training.run(train_args)
        final=json.loads((output/'status/final.json').read_text())
        success=final.get('state')=='completed' and final.get('progress',{}).get('optimizer_updates')==88716
        print('H200_DENSE_FINAL='+json.dumps({'state':final.get('state'),'progress':final.get('progress'),'evaluation':final.get('evaluation'),'checkpoint':final.get('checkpoint')}),flush=True)
        if not success:raise RuntimeError('training paused/stopped before full12epochs; saved state retained')
    except BaseException as error:
        if telemetry:telemetry.fail(error)
        else:
            failure={'error_type':type(error).__name__,'message':str(error)[:1500],'diagnostics':diagnostics()}
            atomic_json(output/'startup-failure.json',failure);print('H200_DENSE_STARTUP_FAILURE='+json.dumps(failure),flush=True)
        raise
    finally:
        if telemetry:telemetry.close(success,final)


if __name__=='__main__':main()

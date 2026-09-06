"""Strict same-run K96 seed521 input-only upgrade. Never starts from scratch."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

LEGACY_BASE = Path('/app/output/daehwa00/lnet-h200-imagenet1k-baselines-v1-2db00309e8e8-17e0ad410448')
MODEL = 'lnet_k96_p128x4_d2262_clean_restart_v3'
LEGACY_RUN = LEGACY_BASE / 'run-data992f76a6fb09' / 'lnet-k96-p128x4-d2262-3seed' / MODEL / 'seed_521'
LEGACY_WORKER_SHA = '99f4b4893fca91a7f947032ec304e1748a7371f99d9934620c84c771f1d2589e'
LEGACY_RUNNER_SHA = '9df60639e771fa5a9ff176f66a8494334b00eaeb9b4f8b8929bd2af13bfbcafa'
STEM = MODEL + '__full__seed521__lr0p003'


def locate():
    for p in (LEGACY_RUN/'checkpoint.pt', LEGACY_RUN/'contracts'/f'{STEM}.json', LEGACY_BASE/'dataset_manifest.json'):
        if not p.is_file():
            raise RuntimeError(f'RESUME_BLOCKED: original artifact is inaccessible: {p}; refusing a fresh run')
    candidates = list(LEGACY_BASE.glob('environment-py3.13.11-*/bin/python'))
    if len(candidates) != 1 or not os.access(candidates[0], os.X_OK):
        raise RuntimeError('RESUME_BLOCKED: original pinned environment is inaccessible; no package changes allowed')
    return candidates[0]


def validate_legacy(contract, checkpoint):
    expected = {'model_key':MODEL,'seed':521,'epochs':100,'batch_size':256,'workers':8,'learning_rate':0.003}
    for key,value in expected.items():
        if contract['task'].get(key) != value:
            raise RuntimeError(f'legacy contract mismatch: {key}')
    if contract['source_sha256'].get('worker') != LEGACY_WORKER_SHA or contract['source_sha256'].get('imagenet1k_runner') != LEGACY_RUNNER_SHA:
        raise RuntimeError('legacy source fingerprint does not match issue384')
    canonical = json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=True)
    if checkpoint.get('contract_sha256') != hashlib.sha256(canonical.encode()).hexdigest():
        raise RuntimeError('checkpoint is not bound to the verified legacy contract')
    if not 1 <= checkpoint.get('completed_epochs',0) < 100:
        raise RuntimeError('expected an unfinished checkpoint with at least one completed epoch')


def main():
    locate()
    if '--locate-python' in sys.argv:
        print(locate())
        return
    import importlib.metadata
    import platform
    import subprocess
    import time
    import torch
    import run_lnet_k96_p128_d2262_imagenet1k as k96
    import run_h200_baseline_worker as worker
    expected = {'torch':'2.9.1+cu128','torchvision':'0.24.1+cu128','numpy':'2.4.6','timm':'1.0.26','wandb':'0.22.3'}
    if platform.python_version() != '3.13.11' or any(importlib.metadata.version(k)!=v for k,v in expected.items()):
        raise RuntimeError('pinned runtime changed; refusing resume')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name():
        raise RuntimeError('this entrypoint requires H200')
    legacy = json.loads((LEGACY_RUN/'contracts'/f'{STEM}.json').read_text())
    checkpoint = torch.load(LEGACY_RUN/'checkpoint.pt',map_location='cpu',weights_only=True)
    validate_legacy(legacy, checkpoint)
    sha=os.environ['H200_EXPECTED_COMMIT']
    out=Path('/app/output/daehwa00/k96-input-resume-v1')/sha/'seed_521'
    out.mkdir(parents=True,exist_ok=True)
    runtime=json.loads((Path('h200/baselines/wandb.runtime.json')).read_text())
    os.environ.update(
        WANDB_BASE_URL=runtime['wandb_base_url'], WANDB_API_KEY='0'*40,
        WANDB_APP_URL='https://wandb.ai', WANDB_CONSOLE='off',
        H200_BASELINE_TORCH_COMPILE_MODE='default', H200_BASELINE_COMPILED_TRAINING_PREPARATION='1',
        LNET_DATASET_MANIFEST_PATH=str(LEGACY_BASE/'dataset_manifest.json'),
        LNET_DATASET_IDENTITY_SHA256=legacy['dataset']['identity_sha256'],
        TORCHINDUCTOR_CACHE_DIR=str(LEGACY_BASE/'cache/torchinductor'),
        TRITON_CACHE_DIR=str(LEGACY_BASE/'cache/triton'),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', H200_GPU_MEMORY_FRACTION='1.0',
        H200_GPU_DRIVER_VERSION=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0].strip(),
    )
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=True
    torch.set_float32_matmul_precision('high')
    k96._install_worker_contract()
    k96._configure_wandb('online',521)
    if os.environ['H200_BASELINE_RUN_ID'] != '26f109ae80471f86':
        raise RuntimeError('W&B identity changed')
    task=worker.BaselineTask('full',MODEL,521,0.003,100,Path(legacy['task']['data_root']),out,
         out/'result.json',out/'checkpoint.pt',source_root=Path.cwd(),batch_size=256,workers=8,wandb_mode='online',resume=True)
    decision_path=out/'input-benchmark.json'
    marker=os.environ.get('LNET_EPOCH_STOP_MARKER')
    if marker and Path(marker).exists():
        print('Stopped before benchmark; legacy checkpoint preserved')
        return
    if not decision_path.exists():
        measurements=[]
        for enabled in ('0','1'):
            worker._seed_everything(521)
            os.environ.update(LNET_GPU_MIXUP=enabled,LNET_YIELD_BEFORE_FETCH=enabled)
            model=k96._build_model(MODEL,None,1000).cuda().to(memory_format=torch.channels_last)
            model.load_state_dict(checkpoint['model'])
            optimizer=worker._build_optimizer(model,0.003,torch.device('cuda'))
            optimizer.load_state_dict(checkpoint['optimizer'])
            active=worker._compiled_runtime(model,torch.device('cuda'))
            loaders=worker._build_loaders(task,torch.device('cuda'))
            batches=worker._training_batches(loaders.train,worker._make_mixup(),torch.device('cuda'))
            active.train()
            for index in range(25):
                if marker and Path(marker).exists():
                    raise RuntimeError('owner stop requested during isolated benchmark')
                if index==5:
                    torch.cuda.synchronize(); started=time.monotonic()
                x,hard,soft=next(batches)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=worker._soft_target_cross_entropy(active(x).float(),soft.float())
                loss.backward(); optimizer.step()
            torch.cuda.synchronize()
            if not torch.isfinite(loss): raise RuntimeError('benchmark loss nonfinite')
            measurements.append(20*256/(time.monotonic()-started))
            if loaders.train._iterator is not None: loaders.train._iterator._shutdown_workers()
            del active,model,optimizer,loaders,batches,x,hard,soft,loss
            torch._dynamo.reset(); torch.cuda.empty_cache()
        worker._atomic_json(decision_path, {'old_images_per_second':measurements[0],
            'new_images_per_second':measurements[1],'enable':measurements[1]>=1.05*measurements[0]})
    decision=json.loads(decision_path.read_text())
    enabled='1' if decision['enable'] else '0'
    os.environ.update(LNET_GPU_MIXUP=enabled,LNET_YIELD_BEFORE_FETCH=enabled)
    if not task.checkpoint_path.exists():
        temporary=out/'checkpoint.copy.tmp'
        shutil.copyfile(LEGACY_RUN/'checkpoint.pt',temporary)
        with temporary.open('rb') as stream: os.fsync(stream.fileno())
        temporary.replace(task.checkpoint_path)
        worker._fsync_directory(out)
        worker._atomic_json(out/'contracts'/f'{STEM}.json',legacy)
    worker._atomic_json(out/'upgrade.json',{'legacy_checkpoint':str(LEGACY_RUN/'checkpoint.pt'),
        'source_commit':sha,'resume_epoch':checkpoint['completed_epochs'],'wandb_run_id':'26f109ae80471f86',
        'benchmark':decision})
    print('H200_K96_INPUT_UPGRADE='+json.dumps(decision),flush=True)
    result=worker.run_task(task,model_builder=k96._build_model,previous_contract=legacy)
    print(json.dumps({'status':result['status'],'checkpoint':str(task.checkpoint_path)}),flush=True)


if __name__=='__main__':
    main()

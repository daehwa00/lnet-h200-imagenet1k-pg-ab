"""Bounded H200 input/compute probe; no production weights or dataset changes."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from torch.utils.data import DataLoader
from timm.data import create_transform
import run_h200_baseline_worker as worker
from in1k10_worker import FixedTrainingSubset,ArrayLoader,numpy_collate,build_model
from in1k10_ipc import BatchLoader,memfd_collate
from in1k10_transport import atomic


def resource_budget(batch_size=256):
    cores=float(len(os.sched_getaffinity(0)))
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':cores=min(cores,int(quota)/int(period))
    except OSError:pass
    available=os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    try:
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:'))
    except (OSError,StopIteration,ValueError):pass
    try:
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit!='max':
            stats=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
            available=min(available,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text())+int(stats.get('inactive_file',0)))
    except OSError:pass
    # Allow worker runtime + raw/transformed/memfd/pinned buffers, not merely one
    # tensor. Production shuts train workers down before final validation.
    per_worker=1024**3+5*batch_size*3*224*224*4
    maximum=min(8,max(1,int(cores)),int((available-3*1024**3)/per_worker))
    if maximum<1:raise RuntimeError('Insufficient host RAM for a bounded worker and batch buffer')
    return {'effective_cpu_cores':cores,'available_memory_bytes':available,'max_workers':maximum,
        'batch_size':batch_size,'host_bytes_per_worker_budget':per_worker}


def input_probe(dataset,workers,backend,batch_size=256):
    generator=torch.Generator().manual_seed(501)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=workers,
        prefetch_factor=1,persistent_workers=True,multiprocessing_context='spawn',generator=generator,
        pin_memory=backend=='memfd',collate_fn=memfd_collate if backend=='memfd' else numpy_collate)
    wrapped=BatchLoader(loader,True) if backend=='memfd' else ArrayLoader(loader,True)
    iterator=iter(wrapped)
    try:
        for _ in range(4):images,labels=next(iterator)
        torch.cuda.synchronize();started=time.perf_counter()
        before=dict(wrapped.timing)
        for _ in range(20):
            images,labels=next(iterator)
            assert images.shape==(batch_size,3,224,224) and labels.shape==(batch_size,)
        seconds=time.perf_counter()-started
        return {'backend':backend,'workers':workers,'batch_size':batch_size,'images_per_second':batch_size*20/seconds,
            'measured_seconds':seconds,'loader_wait_seconds':wrapped.timing['loader_wait_seconds']-before['loader_wait_seconds'],
            'pin_seconds':wrapped.timing['pin_seconds']-before['pin_seconds']}
    finally:
        iterator.close()
        if loader._iterator is not None:loader._iterator._shutdown_workers()
        del iterator,wrapped,loader
        gc.collect()


def compute_probe(sources,batch_size=256):
    worker._seed_everything(501);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda');model=build_model('in10_va_k96',sources,1000).to(device,memory_format=torch.channels_last)
    model.train();optimizer=worker._build_optimizer(model,.003,device)
    active=worker._compiled_runtime(model,device)
    inputs=torch.randn(batch_size,3,224,224,device=device).contiguous(memory_format=torch.channels_last)
    labels=torch.randint(0,1000,(batch_size,),device=device);mixup=worker._make_mixup()
    update=0
    def step():
        nonlocal update
        rate=worker._learning_rate(.003,update,128116//batch_size,100,5)
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        images,targets=mixup(inputs,labels)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=worker._classification_logits(active(images),batch_size)
        loss=worker._soft_target_cross_entropy(logits.float(),targets.float())
        loss.backward();optimizer.step()
        update+=1
        return loss
    for _ in range(10):step()
    torch.cuda.synchronize();started=time.perf_counter()
    for _ in range(20):loss=step()
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    if not torch.isfinite(loss):raise FloatingPointError('Resident compute probe is nonfinite')
    return {'model':'va_k96','batch_size':batch_size,'warmup_steps':10,'measured_steps':20,
        'seconds_per_step':seconds/20,'images_per_second':batch_size*20/seconds,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scope':'resident GPU input incl mixup, forward, backward, fused AdamW; no dataset IO',
        'production_training_started':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    args=p.parse_args();budget=resource_budget()
    os.environ.update(LNET_GPU_MIXUP='1',LNET_RETAIN_EXTERNAL_IMPORTS='1',LNET_DISABLE_LAUNCH_AUTOTUNE='1',
        LNET_COMPILE_RECOMPILE_LIMIT='32',H200_BASELINE_TORCH_COMPILE_MODE='default')
    def stage(name):
        atomic(args.root/'current.json',{'stage':'input_compute_probe','probe_step':name,'production_training_started':False})
        print('IN10_PROBE_STAGE='+name,flush=True)
        if (args.root/'STOP').exists():raise InterruptedError('Stop requested')
    stage('input')
    transform=create_transform(input_size=(3,224,224),is_training=True,auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',re_prob=worker.RANDOM_ERASING_PROBABILITY,re_mode='pixel',re_count=1)
    dataset=FixedTrainingSubset(args.data_root/'train',args.root/'dataset/subset-manifest.json',transform)
    reports=[]
    baseline=('numpy',min(2,budget['max_workers']))
    profiles=[baseline]+[('memfd',n) for n in sorted({min(n,budget['max_workers']) for n in (2,4,8)})]+[baseline]
    for backend,count in profiles:
        stage(f'input-{backend}-{count}-{len(reports)}')
        report=input_probe(dataset,count,backend);reports.append(report)
        print('IN10_INPUT_PROBE='+json.dumps(report),flush=True)
        atomic(args.root/'performance.json',{'resources':budget,'input':reports})
    # Prefer the smaller worker count within 5% of the fastest measured input.
    candidates=[r for r in reports if r['backend']=='memfd']
    best=max(r['images_per_second'] for r in candidates)
    chosen=min((r for r in candidates if r['images_per_second']>=.95*best),key=lambda r:r['workers'])
    compute=[]
    for batch_size in (256,512,1024):
        stage(f'compute-b{batch_size}')
        torch._dynamo.reset();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:result=compute_probe(args.sources,batch_size)
        except torch.cuda.OutOfMemoryError:
            result={'batch_size':batch_size,'error':'CUDA out of memory','production_training_started':False}
        compute.append(result)
        print('IN10_COMPUTE_PROBE='+json.dumps(result),flush=True)
    successful=[r for r in compute if r['batch_size'] in (512,1024) and 'error' not in r]
    if not successful:raise RuntimeError('Neither expanded common batch passed compute probe')
    peak=max(r['images_per_second'] for r in successful)
    batch_size=min(r['batch_size'] for r in successful if r['images_per_second']>=.95*peak)
    # Free the probe process's CUDA caches before child processes test all models.
    torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
    from in1k10_worker import MODELS
    log_root=args.root/'performance';log_root.mkdir(exist_ok=True)
    while True:
        try:batch_budget=resource_budget(batch_size)
        except RuntimeError:
            if batch_size!=1024:raise
            print('IN10_BATCH_FALLBACK=512 reason=host_memory_budget',flush=True)
            batch_size=512;continue
        selected_workers=min(chosen['workers'],batch_budget['max_workers'])
        retry_smaller=False
        for model in MODELS:
            stage(f'common-b{batch_size}-{model}-preflight')
            log=log_root/f'{model}-b{batch_size}.log'
            command=[sys.executable,'-u',str(Path(__file__).with_name('in1k10_worker.py')),
                '--root',str(args.root),'--data-root',str(args.data_root),'--sources',str(args.sources),
                '--model',model,'--seed','501','--phase','preflight','--workers',str(selected_workers),
                '--batch-size',str(batch_size)]
            with log.open('w') as stream:
                process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
            if process.returncode:
                tail=log.read_text()[-24000:]
                print(tail,flush=True)
                if batch_size==1024 and ('CUDA out of memory' in tail or 'torch.OutOfMemoryError' in tail):
                    batch_size=512;retry_smaller=True;break
                raise RuntimeError(f'Common-batch preflight failed: {model}; see {log}')
        if not retry_smaller:break
    stage(f'selected-input-b{batch_size}')
    selected_input=input_probe(dataset,selected_workers,'memfd',batch_size)
    print('IN10_SELECTED_INPUT='+json.dumps(selected_input),flush=True)
    profile={'resources':budget,'batch_resources':batch_budget,'input':reports,'compute':compute,'workers':selected_workers,
        'batch_size':batch_size,'selected_input':selected_input,'all_models_preflight_passed':True,
        'ipc':'memfd','external_backup':False,'validation_policy':'final_epoch_only',
        'production_weights_reused':False}
    atomic(args.root/'performance.json',profile)
    atomic(args.root/'input-profile.json',{'workers':selected_workers,'ipc':'memfd','prefetch_factor':1,'batch_size':batch_size})
    print('IN10_PERFORMANCE='+json.dumps(profile),flush=True)


if __name__=='__main__':main()

"""CUB scratch worker: shared ImageNet recipe; final-test-only reporting."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import run_h200_baseline_worker as common
from cub200_data import CUB, manifest
from cub200_models import MODELS, build


def atomic(path, value):
    common._atomic_json(path, value)


def emit(root, kind, **fields):
    record = dict(kind=kind,time=time.time(),**fields)
    with (root/'telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
        stream.flush()
    print('CUB_EVENT='+json.dumps(record),flush=True)


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in ('scripts','src'):
        for path in sorted((root/directory).rglob('*.py')):
            digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def loaders(data_root, info, batch, workers, seed):
    from timm.data import create_transform
    from torchvision.transforms import InterpolationMode, v2
    train_transform=create_transform(input_size=(3,224,224),is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',interpolation='bicubic',re_prob=.25,re_mode='pixel',re_count=1)
    test_transform=v2.Compose([v2.Resize(256,interpolation=InterpolationMode.BICUBIC,antialias=True),
        v2.CenterCrop(224),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
        v2.Normalize((.485,.456,.406),(.229,.224,.225))])
    generator=torch.Generator().manual_seed(seed)
    kwargs=dict(num_workers=workers,pin_memory=True,persistent_workers=False)
    if workers: kwargs.update(multiprocessing_context='spawn',prefetch_factor=2)
    train=DataLoader(CUB(data_root,info['records'],True,train_transform),batch_size=batch,
        shuffle=True,drop_last=True,generator=generator,**kwargs)
    test=DataLoader(CUB(data_root,info['records'],False,test_transform),batch_size=batch,
        shuffle=False,drop_last=False,**kwargs)
    return train,test,generator


def capture_rng(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),loader=generator.get_state())


def restore_rng(value,generator):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch']); torch.cuda.set_rng_state_all(value['cuda'])
    generator.set_state(value['loader'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--seed',type=int,choices=(501,509,521),required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(0,2),default=2)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    root=args.output; root.mkdir(parents=True,exist_ok=True)
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 required; refusing silent precision fallback')
    torch.set_float32_matmul_precision('high')
    common._seed_everything(args.seed)
    # Only this isolated worker's shared helpers are adapted to the CUB class count.
    common.NUM_CLASSES=200
    common.registry.model_spec=lambda key: SimpleNamespace(precision='bfloat16')
    os.environ['LNET_BATCH_PROGRESS']='1'
    os.environ['LNET_BATCH_PROGRESS_INTERVAL']='5'
    info=manifest(args.data_root)
    train,test,generator=loaders(args.data_root,info,32,args.workers,args.seed)
    model=build(args.model,args.source_root).cuda().to(memory_format=torch.channels_last)
    optimizer=common._build_optimizer(model,.003,torch.device('cuda'))
    def check_gradients(opt, positional, keyword):
        # Infinity means no clipping; refuse nonfinite gradients before updating weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True,foreach=True)
    optimizer.register_step_pre_hook(check_gradients)
    contract=dict(schema='visionalphabet.cub200.scratch.v1',model=args.model,seed=args.seed,
        source_sha256=source_digest(),split_sha256=info['split_sha256'],epochs=100,classes=200,
        physical_batch=32,effective_batch=256,workers=args.workers,pretrained=False,
        optimizer='AdamW',learning_rate=.003,weight_decay=.05,warmup_epochs=5,
        schedule='cosine',mixup=.8,label_smoothing=.1,random_erasing=.25,
        randaugment='rand-m9-mstd0.5-inc1',bf16=True,test_policy='final_epoch_only',
        normalization='native_per_model',drop_last_train=True,last_accumulation_window='actual_microbatch_count',
        versions=dict(torch=torch.__version__,timm=__import__('timm').__version__),
        parameters=sum(p.numel() for p in model.parameters()))
    contract_hash=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    contract_path=root/'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()).get('sha256')!=contract_hash:
            raise RuntimeError('Existing contract changed; retaining original artifacts untouched')
    else:
        atomic(contract_path,dict(contract=contract,sha256=contract_hash))
    checkpoint=root/'checkpoint.pt'
    epoch_done=0; step=0
    if checkpoint.exists():
        if args.preflight: raise RuntimeError('Preflight must use a separate scratch output')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload.get('contract_sha256') != contract_hash: raise RuntimeError('Resume contract changed')
        model.load_state_dict(payload['model'],strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        restore_rng(payload['rng'],generator)
        epoch_done=payload['epoch']; step=payload['global_step']
        emit(root,'resume',epoch=epoch_done,global_step=step)
    task=SimpleNamespace(model_key=args.model,learning_rate=.003,epochs=100,
        gradient_accumulation_steps=8,phase='full',max_steps=1 if args.preflight else None,output_dir=root)
    mixup=common._make_mixup()
    def save(epoch):
        common._atomic_torch(checkpoint,dict(contract_sha256=contract_hash,epoch=epoch,
            global_step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(generator)))
    if args.preflight:
        before=time.monotonic()
        metrics,step,_=common._train_one_epoch(model,train,optimizer,mixup,torch.device('cuda'),task,epoch=1,global_step=0)
        finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
        report=dict(ready=math.isfinite(metrics['loss']) and finite,updates=step,
                    seconds=time.monotonic()-before,peak_bytes=torch.cuda.max_memory_allocated(),contract=contract)
        atomic(root/'preflight.json',report)
        if not report['ready']: raise RuntimeError('Nonfinite preflight')
        return
    for epoch in range(epoch_done+1,101):
        if stopped: break
        start=time.monotonic()
        metrics,step,_=common._train_one_epoch(model,train,optimizer,mixup,torch.device('cuda'),task,
                                               epoch=epoch,global_step=step)
        save(epoch)
        elapsed=time.monotonic()-start
        metrics['epoch_seconds_including_save']=elapsed
        metrics['images_per_second_including_save']=metrics['examples']/elapsed
        atomic(root/'progress.json',dict(state='training',epoch=epoch,epochs=100,global_step=step,
              saved_epoch=epoch,epoch_seconds=elapsed))
        emit(root,'train',epoch=epoch,global_step=step,metrics=metrics)
        epoch_done=epoch
        if stopped: break
    if stopped:
        atomic(root/'progress.json',dict(state='paused',epoch=epoch_done,global_step=step))
        emit(root,'paused',epoch=epoch_done,global_step=step)
        return
    if epoch_done != 100: raise RuntimeError('Refusing premature final test')
    emit(root,'testing',epoch=100,global_step=step)
    model.eval(); correct=torch.zeros((),device='cuda',dtype=torch.int64); count=0
    with torch.inference_mode():
        for images,labels in common._validation_batches(test,torch.device('cuda')):
            if stopped:
                emit(root,'paused',epoch=100,global_step=step); return
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(images)
            if logits.shape != (labels.numel(),200) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError('Invalid final test logits')
            correct += (logits.argmax(1)==labels).sum(); count+=labels.numel()
    if count != 5794: raise RuntimeError('Incomplete official test split')
    result=dict(status='completed',epoch=100,global_step=step,model=args.model,seed=args.seed,
        top1_percent=100*int(correct)/count,test_examples=count,contract_sha256=contract_hash)
    atomic(root/'result.json',result); emit(root,'completed',**result)


if __name__=='__main__': main()

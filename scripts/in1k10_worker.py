"""One matched scratch IN1K10 run; original recipe, fixed public subset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import dataclasses
import numpy as np
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset,DataLoader,Subset
from torchvision.datasets.folder import default_loader
import run_h200_baseline_worker as worker
from in1k10_subset import parse

MODELS={'va_k96':3253224,'va_k128':5083176,'convnextv2_atto':3708400,'tinyvim_s':5684084,'parc_net_s':5037088}


def numpy_collate(batch):
    # NumPy uses ordinary pipe serialization, not torch shared-memory tensor IPC.
    # All original per-image transformations have already run unchanged.
    images,labels=zip(*batch)
    return torch.stack(images).numpy(),np.asarray(labels,dtype=np.int64)


class ArrayLoader:
    def __init__(self,loader,pin):self.loader,self.pin=loader,pin
    @property
    def dataset(self):return self.loader.dataset
    @property
    def _iterator(self):return self.loader._iterator
    def __len__(self):return len(self.loader)
    def __iter__(self):
        for images,labels in self.loader:
            x,y=torch.from_numpy(images),torch.from_numpy(labels)
            yield (x.pin_memory(),y.pin_memory()) if self.pin else (x,y)


class FixedTrainingSubset(Dataset):
    def __init__(self,root,manifest_path,transform):
        root=Path(root);meta=json.loads(Path(manifest_path).read_text())
        names,_=parse(Path(manifest_path).with_name('simclr-10percent.txt').read_bytes())
        self.class_to_idx=meta['class_to_idx'];self.classes=sorted(self.class_to_idx,key=self.class_to_idx.get)
        self.samples=sorted((str(root/n.split('_')[0]/n),self.class_to_idx[n.split('_')[0]]) for n in names)
        self.targets=[y for _,y in self.samples];self.transform=transform
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        path,label=self.samples[index];image=default_loader(path)
        return self.transform(image),label


def build_model(key,source_root,num_classes):
    name=key.removeprefix('in10_')
    if num_classes!=1000 or name not in MODELS:raise ValueError('Unexpected model/class count')
    if name=='va_k96':
        import run_lnet_k96_p128_d2262_imagenet1k as k96
        model=k96._build_model(k96.MODEL_KEY,None,1000)
    elif name=='va_k128':
        from dense_transfer.backbones import _build_va_classifier
        from run_lnet_k96_p128_d2262_imagenet1k import PrimaryLogitsAdapter
        model=PrimaryLogitsAdapter(_build_va_classifier())
    elif name=='convnextv2_atto':
        import timm
        model=timm.create_model(name,pretrained=False,num_classes=1000)
    else:
        import h200_external_models as external
        directory='parc_net' if name=='parc_net_s' else 'tinyvim'
        model=external._build_model(name,Path(source_root)/directory,1000)
        model=external._LogitsAdapter(model)
    count=sum(p.numel() for p in model.parameters())
    if count!=MODELS[name]:raise RuntimeError(f'{name} parameter count changed: {count}')
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--seed',type=int,choices=(501,509,521),required=True)
    p.add_argument('--phase',choices=('preflight','full'),required=True);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    manifest=args.root/'dataset/subset-manifest.json';meta=json.loads(manifest.read_text())
    dataset_id=hashlib.sha256(manifest.read_bytes()).hexdigest()
    key='in10_'+args.model;job=f'{args.model}-{args.seed}'
    output=args.root/('runs' if args.phase=='full' else 'preflight')/job
    task=worker.BaselineTask(phase=args.phase,model_key=key,seed=args.seed,learning_rate=.003,
        epochs=100 if args.phase=='full' else 1,data_root=args.data_root,output_dir=output,
        result_path=output/'result.json',checkpoint_path=output/'checkpoint.pt',source_root=args.sources,
        batch_size=256,workers=args.workers,wandb_mode='disabled',resume=args.resume,
        max_steps=None if args.phase=='full' else 2)
    old_spec=worker.registry.model_spec
    worker.registry.model_spec=lambda k:SimpleNamespace(backend='internal',display_name=args.model,precision='bfloat16') if k==key else old_spec(k)
    worker._dataset_identity=lambda:{'identity_sha256':dataset_id,'manifest_sha256':dataset_id,
        'subset_sha256':meta['sha256'],'train_count':128116,'val_count':50000,'classes':1000}
    old_digests=worker._source_digests
    def digests(k):
        result=old_digests(k)
        import a2d_r2k3_runtime as runtime
        repo=Path(__file__).resolve().parents[1]
        result['in10_runtime']=runtime.source_fingerprint(repo,runtime.source_dependency_paths(repo,('in1k10_worker',)))
        result['subset_manifest']=dataset_id
        identity=args.sources/'source-identity.json'
        result['external_sources']=hashlib.sha256(identity.read_bytes()).hexdigest()
        return result
    worker._source_digests=digests
    original_contract=worker._contract
    def contract(t):
        value=original_contract(t)
        value['recipe']['loader_prefetch_factor']=1 if args.workers else None
        value['recipe']['loader_ipc']='numpy-pipe' if args.workers else 'in-process'
        return value
    worker._contract=contract
    def loaders(t,device):
        import torchvision.datasets
        original=torchvision.datasets.ImageFolder
        def folder(root,transform=None,**kwargs):
            if Path(root)==args.data_root/'train':return FixedTrainingSubset(root,manifest,transform)
            return original(root,transform=transform,**kwargs)
        torchvision.datasets.ImageFolder=folder
        try:bundle=worker._build_loaders(dataclasses.replace(t,workers=0),device)
        finally:torchvision.datasets.ImageFolder=original
        if args.workers:
            def wrap(dataset,generator,training):
                inner=DataLoader(dataset,batch_size=256,shuffle=training,drop_last=training,
                    num_workers=args.workers,prefetch_factor=1,persistent_workers=True,
                    multiprocessing_context='spawn',collate_fn=numpy_collate,generator=generator)
                return ArrayLoader(inner,device.type=='cuda')
            bundle.train=wrap(bundle.train.dataset,bundle.train_generator,True)
            bundle.validation=wrap(bundle.validation.dataset,bundle.validation_generator,False)
        if len(bundle.train.dataset)!=128116 or len(bundle.validation.dataset)!=50000:
            raise RuntimeError('Dataset sizes changed')
        if len(bundle.train)!=500:raise RuntimeError('Expected500updates/epoch at batch256')
        if args.phase=='preflight':
            bundle.validation=DataLoader(Subset(bundle.validation.dataset,range(512)),batch_size=256,
                                         num_workers=0,pin_memory=device.type=='cuda')
        return bundle
    save=worker._atomic_torch
    def checkpoint(path,payload):
        save(path,payload)
        if path==task.checkpoint_path:
            s=path.stat();last=payload['history'][-1]
            worker._atomic_json(output/'checkpoint-meta.json',{'job':job,'phase':args.phase,
                'epoch':payload['completed_epochs'],'global_step':payload['global_step'],
                'path':str(path),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,
                'workers':args.workers,
                'train':last['train'],'validation':last['validation'],'contract_sha256':payload['contract_sha256']})
    worker._atomic_torch=checkpoint
    stop=args.root/'STOP';os.environ['LNET_STOP_FILE']=str(stop)
    def terminate(signum,frame):stop.touch();print('IN10_STOP requested: save at epoch boundary',flush=True)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    if stop.exists():raise RuntimeError('STOP already requested')
    os.environ.update(LNET_GPU_MIXUP='1',LNET_YIELD_BEFORE_FETCH='1',LNET_VALIDATION_PERSISTENT='1',
        LNET_PROGRESS_INTERVAL='20',LNET_RETAIN_EXTERNAL_IMPORTS='1',
        LNET_LOADER_CONTEXT='spawn',LNET_BATCH_PROGRESS='1',LNET_COMPILE_RECOMPILE_LIMIT='32',
        H200_BASELINE_TORCH_COMPILE_MODE='default',LNET_DISABLE_LAUNCH_AUTOTUNE='1',
        H200_GPU_MEMORY_FRACTION='.90',WANDB_MODE='disabled')
    result=worker.run_task(task,model_builder=build_model,loader_builder=loaders)
    if args.phase=='full' and result.get('status')=='completed':
        if result['completed_epochs']!=100 or result['global_step']!=50000 or result['final_validation']['examples']!=50000:
            raise RuntimeError('Incomplete final endpoint')
    print('IN10_RESULT='+json.dumps({k:result.get(k) for k in ('status','model_key','seed','completed_epochs','global_step','final_validation')}),flush=True)


if __name__=='__main__':main()

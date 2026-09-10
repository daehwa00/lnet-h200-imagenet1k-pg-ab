"""Bounded original Vim-Tiny training benchmark on the explicitly selected GPU."""
import argparse
import json
import time
import torch
import h200_baseline_registry as registry
import run_h200_baseline_worker as worker


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',required=True)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--compile',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(501)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(0.95,device)
    model=registry.build_model('vision_mamba_tiny',a.source_root,1000).to(device)
    print('VIM_PARAMETERS='+str(sum(p.numel() for p in model.parameters())),flush=True)
    active=torch.compile(model,mode='default') if a.compile else model
    optimizer=worker._build_optimizer(model,0.003,device)
    x=torch.randn(a.batch_size,3,224,224,device=device)
    y=torch.randint(1000,(a.batch_size,),device=device)
    timings=[]
    for step in range(a.steps):
        started=time.monotonic()
        active.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=active(x)
            loss=torch.nn.functional.cross_entropy(logits.float(),y)
        loss.backward()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite Vim loss')
        if step==0 and not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('nonfinite Vim gradients')
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.monotonic()-started)
        print(json.dumps({'step':step+1,'loss':float(loss.detach()),'seconds':timings[-1]}),flush=True)
    optimizer.zero_grad(set_to_none=True)
    active.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        z=active(x)
    assert z.shape==(a.batch_size,1000) and bool(torch.isfinite(z).all())
    print('VIM_SMOKE_JSON='+json.dumps({'batch':a.batch_size,'compile':a.compile,
        'images_per_second':a.batch_size*(len(timings)-2)/sum(timings[2:]),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'finite':True}),flush=True)


if __name__=='__main__': main()

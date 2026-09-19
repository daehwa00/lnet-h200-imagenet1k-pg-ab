"""Single-request sequential15-run campaign, gated by W&B and private backup."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from in1k10_transport import atomic,read

MODELS=('va_k96','va_k128','convnextv2_atto','tinyvim_s','parc_net_s')


def wait_for(root,predicate,label,seconds=600):
    deadline=time.monotonic()+seconds
    while not predicate():
        if (root/'STOP').exists():raise InterruptedError('Stop requested while '+label)
        if time.monotonic()>deadline:raise TimeoutError(label)
        time.sleep(2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args();root=args.root
    def stop(signum,frame):(root/'STOP').touch()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    manifest=read(root/'dataset/subset-manifest.json')
    if manifest.get('images')!=128116 or manifest.get('validation_count')!=50000:raise RuntimeError('Subset manifest not ready')
    workers=2
    try:
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit.isdigit() and int(limit)<12*1024**3:workers=0
    except OSError:workers=0
    scripts=Path(__file__).resolve().parent;completed=[]
    for seed in (501,509,521):
        for model in MODELS:
            job=f'{model}-{seed}';output=root/'runs'/job
            def stage(name,**extra):
                atomic(root/'current.json',{'stage':name,'job':job,'model':model,'seed':seed,
                    'output':str(output),'subset_sha256':manifest['sha256'],'workers':workers,
                    'epochs':100,'updates':50000,'train_images':128116,'validation_images':50000,**extra})
                print(f'IN10_STAGE job={job} stage={name}',flush=True)
            if (root/'STOP').exists():return 130
            stage('restore')
            wait_for(root,lambda:(root/f'restored-{job}.json').exists(),'restore '+job)
            restored=read(root/f'restored-{job}.json').get('artifact')
            if restored:workers=int(restored.get('workers',workers))
            common=['--root',str(root),'--data-root',str(args.data_root),'--sources',str(args.sources),
                    '--model',model,'--seed',str(seed),'--workers',str(workers)]
            stage('preflight')
            cmd=[sys.executable,'-u',str(scripts/'in1k10_worker.py'),*common,'--phase','preflight']
            code=subprocess.run(cmd).returncode
            if code or (root/'STOP').exists():return code or 130
            gate=read(root/'preflight'/job/'result.json')
            if gate.get('global_step')!=2 or gate.get('final_validation',{}).get('examples')!=512:
                raise RuntimeError('Full-input gate did not complete its declared budget')
            if args.preflight_only:continue
            stage('waiting_wandb')
            wait_for(root,lambda:job in read(root/'control.json').get('ready_jobs',[]),'W&B readiness '+job)
            stage('training')
            cmd=[sys.executable,'-u',str(scripts/'in1k10_worker.py'),*common,'--phase','full']
            if (output/'checkpoint.pt').exists():cmd+=['--resume']
            process=subprocess.Popen(cmd);stage('training',child_pid=process.pid)
            code=process.wait()
            if code or (root/'STOP').exists():return code or 130
            result=read(output/'result.json')
            if result.get('completed_epochs')!=100 or result.get('global_step')!=50000:
                raise RuntimeError('Full100epoch result missing; refusing next run')
            stage('wait_backup')
            wait_for(root,lambda:(root/f'backup-ack-{job}.json').exists(),'external checkpoint verification '+job,2100)
            completed.append({'job':job,'top1':result['final_validation']['accuracy'],'checkpoint':read(root/f'backup-ack-{job}.json')})
            atomic(root/'completed.json',completed)
    atomic(root/'current.json',{'stage':'completed','completed_runs':len(completed),'preflight_only':args.preflight_only})
    print('IN10_CAMPAIGN_COMPLETE='+json.dumps({'runs':len(completed),'preflight_only':args.preflight_only}),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())

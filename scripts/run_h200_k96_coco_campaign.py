"""Sequential fixed-pretrain K96 COCO seeds, launched inside h200_http_guard."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path)
    args=p.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError('K96 fixed ImageNet checkpoint is not visible: '+str(args.checkpoint))
    with args.checkpoint.open('rb') as stream:
        actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=args.checkpoint_sha256:
        raise ValueError('K96 checkpoint SHA256 mismatch')
    stopped=False
    def stop(signum, frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    scripts=Path(__file__).resolve().parent
    # Conservative IPC profile; same batch, optimizer, head, resolution and schedule.
    for seed in (501,509,521):
        if stopped:return 130
        output=args.root/f'seed_{seed}'
        final=output/'status/final.json'
        common=['--task','coco','--model','va_k96','--seed',str(seed),
                '--checkpoint',str(args.checkpoint),'--data-root',str(args.data_root),
                '--output-root',str(output),'--physical-batch-size','2','--effective-batch-size','16',
                '--workers','0','--prefetch-factor','1','--sharing-strategy','file_system']
        print(f'K96_STAGE seed={seed} validating',flush=True)
        result=subprocess.run([sys.executable,str(scripts/'validate_dense_transfer_runtime.py'),*common,
                               '--mode','smoke','--max-probe-updates','2'])
        if stopped:return 130
        if result.returncode:return result.returncode
        resume=output/'checkpoints/last.pt'
        command=[sys.executable,str(scripts/'run_dense_transfer.py'),*common,'--mode','train',
                 '--confirm-training','DENSE_TRANSFER_TRAIN']
        if resume.exists():command+=['--resume',str(resume)]
        print(f'K96_STAGE seed={seed} training',flush=True)
        result=subprocess.run(command)
        if stopped:return 130
        if result.returncode:return result.returncode
        state=json.loads(final.read_text())
        if state.get('state')!='completed' or state.get('progress',{}).get('optimizer_updates')!=88716:
            raise RuntimeError('Seed did not complete; refusing next seed')
        print('K96_FINAL '+json.dumps({'seed':seed,'evaluation':state['evaluation']}),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())

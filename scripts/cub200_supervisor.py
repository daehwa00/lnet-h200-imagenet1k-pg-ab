"""CPU-side queue and W&B stop/log supervisor, separate from CUDA training."""
import argparse
from contextlib import suppress
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from cub200_run_ids import run_id


def read_json(path):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return {}


def atomic(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value)); tmp.replace(path)


def source_digest():
    root=Path(__file__).resolve().parents[1]; digest=hashlib.sha256()
    for directory in ('scripts','src'):
        for path in sorted((root/directory).rglob('*.py')):
            digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def completed_endpoint(output,model,seed,current_source):
    result=read_json(output/'result.json')
    if not result:return None
    document=read_json(output/'contract.json');contract=document.get('contract',{})
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if (result.get('status')!='completed' or result.get('epoch')!=100
        or result.get('model')!=model or result.get('seed')!=seed
        or result.get('test_examples')!=5794 or contract.get('source_sha256')!=current_source
        or result.get('contract_sha256')!=digest or document.get('sha256')!=digest
        or not (output/'checkpoint.pt').is_file()):
        raise RuntimeError('Existing result failed completion identity checks; retained untouched')
    return result


def terminate(child,grace):
    if child.poll() is not None: return False
    # Signal only the trainer first: killing its DataLoader workers would abort
    # the epoch before it can checkpoint. Force-kill the whole group on timeout.
    with suppress(ProcessLookupError): os.kill(child.pid,signal.SIGTERM)
    try: child.wait(timeout=grace); return False
    except subprocess.TimeoutExpired: pass
    with suppress(ProcessLookupError): os.killpg(child.pid,signal.SIGKILL)
    child.wait(timeout=30)
    return True


def supervise(command,root,run,stop_path,*,grace=600,stall=1800):
    root.mkdir(parents=True,exist_ok=True)
    if stop_path.exists():
        atomic(root/'supervisor.json',dict(state='stopped',reason='stop_marker_before_launch',forced=False))
        return False
    log=root/'console.log'; offset=0; telemetry_offset=0; last_progress=None
    last_change=time.monotonic(); child=None; stop_reason=None
    with log.open('ab',buffering=0) as stream:
        run.save(str(log),base_path=str(root),policy='live')
        environment={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONFAULTHANDLER':'1'}
        environment.pop('WANDB_API_KEY',None)
        environment.pop('WANDB_API_KEY_FILE',None)
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
            env=environment)
        try:
            while child.poll() is None:
                if stop_path.exists():
                    stop_reason='stop_marker'; break
                progress=max((read_json(root/'step-progress.json'),read_json(root/'progress.json')),
                             key=lambda row:row.get('global_step',0))
                signature=json.dumps(progress,sort_keys=True)
                if signature!=last_progress:
                    last_progress=signature; last_change=time.monotonic()
                if time.monotonic()-last_change>stall:
                    stop_reason='no_progress_timeout'; break
                with log.open('rb') as source:
                    source.seek(offset); chunk=source.read(65536); offset+=len(chunk)
                if chunk: print(chunk.decode(errors='replace'),end='',flush=True)
                event_path=root/'telemetry.jsonl'
                if event_path.exists():
                    with event_path.open() as source:
                        source.seek(telemetry_offset)
                        while True:
                            line_start=source.tell(); line=source.readline()
                            if not line: break
                            try: event=json.loads(line)
                            except ValueError:
                                source.seek(line_start); break
                            fields={'epoch':event.get('epoch',0),'global_step':event.get('global_step',0)}
                            if event['kind']=='train': fields.update({'train/'+k:v for k,v in event['metrics'].items()})
                            if event['kind']=='completed': fields['test/top1']=event['top1_percent']
                            run.log(fields)
                        telemetry_offset=source.tell()
                run.summary.update({'phase':'running','heartbeat_unix':time.time(),
                    'global_step':progress.get('global_step',0),'epoch':progress.get('epoch',0),
                    'saved_epoch':read_json(root/'progress.json').get('saved_epoch',0)})
                atomic(root/'supervisor.json',dict(state='running',pid=child.pid,heartbeat=time.time(),progress=progress))
                time.sleep(5)
        except KeyboardInterrupt:
            # W&B SDK StopRequested poll interrupts this CPU supervisor, not the CUDA worker.
            stop_reason='wandb_or_signal_stop'
        except BaseException:
            terminate(child,grace)
            raise
        if stop_reason:
            stop_path.touch(exist_ok=True)
            forced=terminate(child,grace)
            atomic(root/'supervisor.json',dict(state='stopped',reason=stop_reason,forced=forced,exit_code=child.returncode))
            run.summary.update({'phase':'stopped','stop_reason':stop_reason,'forced':forced})
            return False
        code=child.wait()
        with log.open('rb') as source:
            source.seek(offset); print(source.read().decode(errors='replace'),end='',flush=True)
        state='exited' if code==0 else 'failed'
        atomic(root/'supervisor.json',dict(state=state,exit_code=code,finished_unix=time.time()))
        run.summary.update({'phase':state,'exit_code':code})
        run.save(str(log),base_path=str(root),policy='now')
        if code: raise RuntimeError(f'Worker exited {code}; full stderr retained at {log}')
        return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(); config=json.loads(args.config.read_text())
    expected={'models':['va_k96','va_k128','convnextv2_atto','tinyvim_s','parc_net_s'],
        'seeds':[501,509,521],'epochs':100,'classes':200,'image_size':224,
        'physical_batch_size':32,'effective_batch_size':256,'learning_rate':.003,
        'weight_decay':.05,'warmup_epochs':5,'mixup_alpha':.8,'label_smoothing':.1,
        'random_erasing':.25,'pretrained':False,'bf16':True}
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('Campaign differs from the frozen CUB recipe')
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=args.root/'STOP'
    if stop.exists(): raise RuntimeError('Campaign stop is latched; explicit operator resume required')
    def stop_signal(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop_signal)
    import wandb
    if os.environ.get('WANDB_MODE','online')!='online': raise RuntimeError('Online W&B required')
    scripts=Path(__file__).resolve().parent
    shm=os.statvfs('/dev/shm'); workers=config['workers'] if shm.f_bavail*shm.f_frsize>=1024**3 else 0
    atomic(args.root/'loader-profile.json',{'workers':workers,'physical_batch':32,'effective_batch':256})
    def open_run(name,output):
        # Operator increments this only when explicitly resuming a stopped W&B run.
        attempt=os.environ.get('CUB_TELEMETRY_ATTEMPT','0')
        if not attempt.isdigit():raise ValueError('Invalid telemetry attempt')
        identifier=run_id(config['campaign_id'],name,attempt)
        run=wandb.init(entity=config['wandb_entity'],project=config['wandb_project'],
            group=config['campaign_id'],id=identifier,name=name,resume='allow',mode='online',
            config={**config,'actual_workers':workers},
            settings=wandb.Settings(init_timeout=45,console='off',disable_code=True,disable_git=True,
                x_disable_stats=True))
        if run.offline: raise RuntimeError('Refusing unobserved offline training')
        if run.summary.get('saved_epoch',0)>0 and not (output/'checkpoint.pt').exists():
            run.finish(exit_code=1)
            raise RuntimeError('W&B has prior progress but local checkpoint is missing; do not overwrite this run')
        print('CUB_WANDB_URL='+run.url,flush=True)
        return run
    # Test every model before scheduling any production run. No test-set evaluation.
    canary=open_run('preflight',args.root/'preflight')
    try:
        for model in config['models']:
            out=args.root/'preflight'/model
            command=[sys.executable,str(scripts/'train_cub200.py'),'--model',model,'--seed','501',
                '--data-root',str(args.data_root),'--source-root',str(args.source_root),'--output',str(out),
                '--workers',str(workers),'--preflight']
            if not supervise(command,out,canary,stop,grace=30,stall=600): return
            if not read_json(out/'preflight.json').get('ready'): raise RuntimeError('Preflight failed')
    finally: canary.finish()
    if args.preflight_only:return
    completed=[]
    current_source=source_digest()
    for seed in config['seeds']:
        for model in config['models']:
            if stop.exists():return
            out=args.root/'runs'/f'{model}-seed{seed}'
            existing=completed_endpoint(out,model,seed,current_source)
            if existing:
                completed.append(existing)
                print('CUB_SKIP_COMPLETED='+json.dumps(existing),flush=True)
                continue
            command=[sys.executable,str(scripts/'train_cub200.py'),'--model',model,'--seed',str(seed),
                '--data-root',str(args.data_root),'--source-root',str(args.source_root),'--output',str(out),
                '--workers',str(workers)]
            run=open_run(f'{model}-seed{seed}',out)
            try:
                if not supervise(command,out,run,stop,grace=config['stop_grace_seconds'],stall=config['max_no_progress_seconds']):return
                final=read_json(out/'result.json')
                if final.get('status')!='completed' or final.get('epoch')!=100:
                    raise RuntimeError('No valid final endpoint; refusing next run')
                run.summary.update({'phase':'completed','test/top1':final['top1_percent'],'saved_epoch':100})
                completed.append(final)
                atomic(args.root/'results.json',completed)
            except BaseException:
                run.finish(exit_code=1); raise
            else:run.finish()
    import statistics
    summary={model:{'mean_top1':statistics.mean(r['top1_percent'] for r in completed if r['model']==model),
                    'sample_sd':statistics.stdev(r['top1_percent'] for r in completed if r['model']==model)} for model in config['models']}
    atomic(args.root/'summary.json',summary)


if __name__=='__main__': main()

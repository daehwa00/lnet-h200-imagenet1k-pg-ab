"""Owner-side W&B observer and SHA-verified private checkpoint receiver."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import wandb
from in1k10_transport import API,atomic,download,read

MODELS={'va_k96':3253224,'va_k128':5083176,'convnextv2_atto':3708400,'tinyvim_s':5684084,'parc_net_s':5037088}
JOBS={f'{m}-{s}' for m in MODELS for s in (501,509,521)}
PROJECT='alphabet2d-imagenet1k-10pct'


def receive(api,job,meta,root):
    if shutil.disk_usage(root).free < meta['bytes']*3+1024**3:raise RuntimeError('Insufficient private backup space')
    target=root/'checkpoints'/job/'checkpoint.pt'
    download(api,job,meta,target)
    import torch
    checkpoint=torch.load(target,map_location='cpu',weights_only=True)
    model=job.rsplit('-',1)[0]
    if checkpoint.get('completed_epochs')!=meta['epoch'] or checkpoint.get('global_step')!=500*meta['epoch'] or checkpoint.get('parameters')!=MODELS[model]:
        raise RuntimeError('Downloaded checkpoint identity/update budget mismatch')
    atomic(target.with_suffix('.receipt.json'),meta)
    api.call(f'/artifact/{job}/{meta["sha"]}/ack',{})
    return {'job':job,'epoch':meta['epoch'],'sha':meta['sha'],'history':checkpoint['history']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--secrets',type=Path,required=True)
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    api=API(read(args.secrets)['IN10_OWNER_TOKEN']);state=read(root/'observer-state.json')
    cursor=state.get('cursor',0);steps=state.get('steps',{});epochs=state.get('epochs',{})
    run=None;active_job=None;backup=None;backup_job=None;last_api_check=0.;wb_api=wandb.Api(timeout=15)
    completed=set(state.get('completed',[]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            try:
                view=api.call('/snapshot?after='+str(cursor))
                for event in view['events']:
                    value=json.loads(event['value']);status=value.get('status',{});current=status.get('current',{})
                    job=current.get('job')
                    with (root/'events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
                    if job in JOBS and current.get('stage') in ('waiting_wandb','training','wait_backup') and job!=active_job:
                        if run:run.finish(exit_code=0 if active_job in completed else 1)
                        model,seed=job.rsplit('-',1);run_id=hashlib.sha256(('simclr-in1k10-v1:'+job).encode()).hexdigest()[:16]
                        run=wandb.init(entity='daehwa',project=PROJECT,group='simclr10-100ep-v1',id=run_id,resume='allow',
                            name=f'IN1K10-{model}-s{seed}',config={'model':model,'seed':int(seed),'scratch':True,
                                'train_images':128116,'validation_images':50000,'classes':1000,'subset_sha256':current['subset_sha256'],
                                'epochs':100,'optimizer_updates':50000,'effective_batch':256,'physical_batch':256,
                                'optimizer':'AdamW','learning_rate':.003,'weight_decay':.05,'warmup_epochs':5,'image_size':224,
                                'code_sha':view['session']['code'],'source':'official SimCLR10percent.txt'},
                            settings=wandb.Settings(console='off',disable_code=True,disable_git=True,x_disable_stats=True,init_timeout=45))
                        run.define_metric('optimizer_updates');run.define_metric('train/*',step_metric='optimizer_updates')
                        run.define_metric('validation/*',step_metric='optimizer_updates')
                        run.summary['logging_canary']='ready'
                        # Verify the named remote run exists before authorizing GPU training.
                        deadline=time.monotonic()+60
                        while True:
                            try:wb_api.run(f'daehwa/{PROJECT}/{run_id}');break
                            except Exception:
                                if time.monotonic()>deadline:raise
                                time.sleep(2)
                        active_job=job;api.call('/command',{'action':'ready','job':job})
                    if run and job==active_job:
                        checkpoint=status.get('checkpoint',{});epoch=checkpoint.get('epoch',0)
                        if checkpoint.get('phase')=='full' and epoch>epochs.get(job,0):
                            train=checkpoint['train'];validation=checkpoint['validation']
                            run.log({'optimizer_updates':checkpoint['global_step'],'epoch':epoch,
                                'train/loss_epoch':train['loss'],'validation/top1_percent':100*validation['accuracy'],
                                'validation/top5_percent':100*validation['top5_accuracy']})
                            epochs[job]=epoch
                        progress=status.get('progress',{});step=progress.get('global_step',0)
                        if step>steps.get(job,0):
                            payload={'optimizer_updates':step,'epoch':progress['epoch']}
                            if 'loss' in progress:payload['train/loss_step']=progress['loss']
                            run.log(payload);steps[job]=step
                        run.summary.update({'stage':current.get('stage'),'heartbeat_unix':status.get('time'),
                            'last_observed_step':step,'guard_error':status.get('error') or ''})
                        result=status.get('result',{})
                        if result.get('completed_epochs')==100 and result.get('global_step')==50000 and result.get('status')=='completed':
                            completed.add(job)
                            run.summary.update({'completed_epochs':100,'final/top1_percent':100*result['final_validation']['accuracy']})
                        if backup is None:
                            meta=api.call(f'/artifact/{job}/latest')
                            receipt=read(root/'checkpoints'/job/'checkpoint.receipt.json')
                            if meta and meta['sha']!=receipt.get('sha'):
                                backup=pool.submit(receive,api,job,meta,root);backup_job=job
                        if status.get('ended'):
                            run.summary['process_exit_code']=status.get('exit_code')
                    cursor=event['id']
                if backup and backup.done():
                    receipt=backup.result()
                    if run and active_job==receipt['job']:
                        run.summary.update({'backed_up_epoch':receipt['epoch'],'checkpoint_sha256':receipt['sha']})
                        for row in receipt['history']:
                            if row['epoch']>epochs.get(active_job,0):
                                run.log({'optimizer_updates':row['global_step'],'epoch':row['epoch'],
                                    'train/loss_epoch':row['train']['loss'],'validation/top1_percent':100*row['validation']['accuracy'],
                                    'validation/top5_percent':100*row['validation']['top5_accuracy']})
                                epochs[active_job]=row['epoch']
                    backup=None;backup_job=None
                if run and time.monotonic()-last_api_check>60:
                    wb_api.run(f'daehwa/{PROJECT}/{run.id}');last_api_check=time.monotonic()
                api.call('/command',{'action':'heartbeat'})
                atomic(root/'observer-state.json',{'cursor':cursor,'steps':steps,'epochs':epochs,'active_job':active_job,
                    'backup_job':backup_job,'completed':sorted(completed),'heartbeat':time.time()})
                if view['session'] and view['session']['ended'] and not view['events'] and backup is None:
                    if run:run.finish(exit_code=0 if active_job in completed else 1);run=None;active_job=None
            except Exception as error:
                atomic(root/'observer-error.json',{'time':time.time(),'error':type(error).__name__+': '+str(error)[:300]})
                print('IN10_OBSERVER_ERROR '+type(error).__name__,flush=True)
                if backup and backup.done():backup=None;backup_job=None
            time.sleep(5)


if __name__=='__main__':main()

"""Owner-side live logs and W&B metrics; local-only H200 checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import wandb
from in1k10_transport import API,atomic,read

CAMPAIGN='simclr10-finaleval-v3'
MODELS=('va_k96','va_k128','convnextv2_atto','tinyvim_s','parc_net_s')
JOBS={f'{m}-{s}' for m in MODELS for s in (501,509,521)}
PROJECT='alphabet2d-imagenet1k-10pct'


def run_id(job):
    return hashlib.sha256((CAMPAIGN+':'+job).encode()).hexdigest()[:16]


def checkpoint_metrics(checkpoint):
    train=checkpoint['train'];validation=checkpoint['validation']
    payload={'optimizer_updates':checkpoint['global_step'],'epoch':checkpoint['epoch'],
        'train/loss_epoch':train['loss'],'train/images_per_second':train['images_per_second']}
    if validation.get('accuracy') is not None:
        payload.update({'validation/top1_percent':100*validation['accuracy'],
            'validation/top5_percent':100*validation['top5_accuracy']})
    for key in ('loader_wait_seconds','pin_seconds','gpu_step_span_seconds','host_image_bytes'):
        if key in train:payload['timing/'+key]=train[key]
    return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--secrets',type=Path,required=True)
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    api=API(read(args.secrets)['IN10_OWNER_TOKEN'])
    state_path=root/'observer-state-finaleval-v3.json';state=read(state_path)
    cursor=state.get('cursor',read(root/'observer-state-local-v2.json').get('cursor',0))
    steps=state.get('steps',{});epochs=state.get('epochs',{});completed=set(state.get('completed',[]))
    run=None;active_job=None;last_api_check=0.;wb_api=wandb.Api(timeout=15)
    while True:
        try:
            view=api.call('/snapshot?after='+str(cursor))
            for event in view['events']:
                value=json.loads(event['value']);status=value.get('status',{});current=status.get('current',{})
                with (root/'events-finaleval-v3.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
                job=current.get('job')
                if current.get('campaign_id')!=CAMPAIGN:
                    cursor=event['id'];continue
                if job in JOBS and job not in completed and current.get('stage') in ('waiting_wandb','training','waiting_metrics') and (job!=active_job or run is None):
                    if run:run.finish(exit_code=0 if active_job in completed else 1)
                    model,seed=job.rsplit('-',1);identifier=run_id(job)
                    run=wandb.init(entity='daehwa',project=PROJECT,group=CAMPAIGN,id=identifier,resume='allow',
                        name=f'IN1K10-{model}-s{seed}-finaleval3',config={'model':model,'seed':int(seed),'scratch':True,
                            'train_images':128116,'validation_images':50000,'classes':1000,'subset_sha256':current['subset_sha256'],
                            'epochs':100,'optimizer_updates':current['updates'],'effective_batch':current['batch_size'],'physical_batch':current['batch_size'],
                            'optimizer':'AdamW','learning_rate':.003,'weight_decay':.05,'warmup_epochs':5,'image_size':224,
                            'code_sha':view['session']['code'],'source':'official SimCLR10percent.txt','external_backup':False,
                            'evaluation_policy':'final_epoch_only','loader_ipc':'memfd','workers':current['workers']},
                        settings=wandb.Settings(console='off',disable_code=True,disable_git=True,x_disable_stats=True,init_timeout=45))
                    run.define_metric('optimizer_updates');run.define_metric('train/*',step_metric='optimizer_updates')
                    run.define_metric('validation/*',step_metric='optimizer_updates')
                    run.define_metric('timing/*',step_metric='optimizer_updates')
                    run.summary['logging_canary']='ready'
                    wb_api.flush()
                    wb_api.run(f'daehwa/{PROJECT}/{identifier}')
                    active_job=job;api.call('/command',{'action':'ready','job':job})
                if run and job==active_job:
                    checkpoint=status.get('checkpoint',{});epoch=checkpoint.get('epoch',0)
                    if checkpoint.get('phase')=='full' and epoch>epochs.get(job,0):
                        run.log(checkpoint_metrics(checkpoint))
                        epochs[job]=epoch
                    progress=status.get('progress',{});step=progress.get('global_step',0)
                    if step>steps.get(job,0):
                        payload={'optimizer_updates':step,'epoch':progress['epoch']}
                        if 'loss' in progress:payload['train/loss_step']=progress['loss']
                        run.log(payload);steps[job]=step
                    run.summary.update({'stage':current.get('stage'),'heartbeat_unix':status.get('time'),
                        'last_observed_step':step,'guard_error':status.get('error') or ''})
                    result=status.get('result',{})
                    if result.get('completed_epochs')==100 and result.get('global_step')==current['updates'] and result.get('status')=='completed':
                        run.summary.update({'completed_epochs':100,'global_step':current['updates'],
                            'final/top1_percent':100*result['final_validation']['accuracy']})
                        run.finish();run=None
                        wb_api.flush()
                        saved=wb_api.run(f'daehwa/{PROJECT}/{run_id(job)}')
                        if saved.summary.get('completed_epochs')!=100 or saved.summary.get('global_step')!=current['updates']:
                            raise RuntimeError('Final W&B metrics not visible yet')
                        completed.add(job)
                if job in completed:api.call('/command',{'action':'finish','job':job})
                cursor=event['id']
            if run and time.monotonic()-last_api_check>60:
                wb_api.flush()
                wb_api.run(f'daehwa/{PROJECT}/{run.id}');last_api_check=time.monotonic()
            api.call('/command',{'action':'heartbeat'})
            atomic(state_path,{'cursor':cursor,'steps':steps,'epochs':epochs,'active_job':active_job,
                'completed':sorted(completed),'heartbeat':time.time(),'external_backup':False})
            if view['session'] and view['session']['ended'] and not view['events']:
                if run:run.finish(exit_code=0 if active_job in completed else 1);run=None;active_job=None
        except Exception as error:
            atomic(root/'observer-error-finaleval-v3.json',{'time':time.time(),'error':type(error).__name__+': '+str(error)[:300]})
            print('IN10_OBSERVER_ERROR '+type(error).__name__,flush=True)
        time.sleep(5)


if __name__=='__main__':main()

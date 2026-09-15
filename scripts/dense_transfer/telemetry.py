"""Long-lived, secret-free H200 relay logging and bounded local diagnostics."""
from __future__ import annotations
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict
from typing import Any

from .engine import atomic_json, contract_hash


def diagnostics(loaders=()) -> dict[str, Any]:
    out={'unix_time':time.time(),'pid':os.getpid(),'hostname':socket.gethostname(),'tmpdir':os.environ.get('TMPDIR'),
         'source_present':Path(__file__).is_file(),'open_file_limit':list(resource.getrlimit(resource.RLIMIT_NOFILE))}
    for name in ('memory.current','memory.max','memory.events','memory.peak'):
        p=Path('/sys/fs/cgroup')/name
        try:out['cgroup/'+name]=p.read_text().strip()[:1500]
        except OSError:pass
    for name in ('memory.usage_in_bytes','memory.limit_in_bytes','memory.failcnt','memory.oom_control'):
        p=Path('/sys/fs/cgroup/memory')/name
        try:out['cgroup_v1/'+name]=p.read_text().strip()[:1500]
        except OSError:pass
    for name,path in [('shm','/dev/shm'),('output',os.environ.get('H200_DENSE_TASK_ROOT','/app/output'))]:
        try:out[name+'_free_bytes']=shutil.disk_usage(path).free
        except OSError:pass
    out['tmpdir_present']=bool(out['tmpdir']) and Path(out['tmpdir']).is_dir()
    workers=[]
    for loader in loaders:
        for worker in getattr(getattr(loader,'_iterator',None),'_workers',[]):
            workers.append({'pid':worker.pid,'exitcode':worker.exitcode,'alive':worker.is_alive()})
    out['workers']=workers
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=5)
        if r.returncode==0:out['gpu']=r.stdout.strip()[:800]
    except (OSError,subprocess.SubprocessError):pass
    return out


def runtime_config() -> dict[str, Any]:
    return json.loads((Path(__file__).resolve().parents[2]/'h200/dense_transfer/wandb-v2.json').read_text())


def configure_relay(config):
    for key in ('WANDB_CONFIG_PATHS','WANDB_IDENTITY_TOKEN_FILE','WANDB_NAME','WANDB_RUN_ID','WANDB_RESUME','WANDB_TAGS'):
        os.environ.pop(key,None)
    os.environ.update(WANDB_API_KEY='0'*40,WANDB_BASE_URL=config['wandb_base_url'],
        WANDB_APP_URL=config['wandb_app_url'],WANDB_ENTITY=config['entity'],WANDB_PROJECT=config['project'],
        WANDB_GROUP=config['group'],WANDB_MODE='online',WANDB_CONSOLE='off',WANDB_INIT_TIMEOUT='30',WANDB_HTTP_TIMEOUT='20')


def relay_rejection_probe():
    """An intentionally invalid, non-writing request distinguishes IP/config failures."""
    import requests
    try:
        response=requests.post(runtime_config()['wandb_base_url']+'/graphql',json={},timeout=10)
        value=response.json()
        return {'http_status':response.status_code,'reason':value.get('reason','not-returned')}
    except Exception as error:return {'probe_error':type(error).__name__}


def open_run(role='train', *, has_checkpoint=False, config_values=None):
    config=runtime_config();configure_relay(config)
    import wandb
    if wandb.__version__!=config['wandb_sdk_version']:raise RuntimeError('Relay protocol requires pinned W&B SDK')
    spec=config['runs'][role]
    # Use the SDK's already traced native protocol. An unrelated Python API
    # helper emits a different query/envelope and is intentionally not allowed.
    resume='allow' if role=='canary' or has_checkpoint else 'never'
    run=wandb.init(entity=config['entity'],project=config['project'],group=config['group'],id=spec['id'],
        name=spec['display_name'],tags=spec['tags'],resume=resume,mode='online',force=True,anonymous='never',
        config=config_values or {},settings=wandb.Settings(console='off',disable_code=True,disable_git=True,
            disable_job_creation=True,save_code=False,x_disable_meta=True,x_disable_stats=True,x_disable_viewer=True,
            x_save_requirements=False,init_timeout=30,
            x_extra_http_headers={'User-Agent':'Mozilla/5.0 lnet-h200-dense-v2/1'}))
    if run is None or not run.url:raise RuntimeError('W&B online handshake did not return a run')
    return run


class Telemetry:
    def __init__(self,output: Path,*,has_checkpoint=False,loader_profile=None):
        self.output=output;output.mkdir(parents=True,exist_ok=True)
        self.loaders=();self.phase='initializing';self.latest={};self.last_update=None;self.last_time=time.monotonic()
        self.lock=threading.Lock();self.stop=threading.Event();self.last_error=None
        self.run=open_run(has_checkpoint=has_checkpoint,config_values={
            'task':'coco','model':'va_k128','downstream_seed':501,'pretrain_seed':501,'epochs':12,
            'physical_batch_size':2,'effective_batch_size':16,'execution_profile':'hybrid-20260915',
            'loader_profile':loader_profile,'code_commit':os.environ.get('H200_RUNTIME_COMMIT'),
            'resumed_from_visible_checkpoint':has_checkpoint})
        self.run.define_metric('train/*',step_metric='optimizer_updates')
        print('H200_DENSE_WANDB='+self.run.url,flush=True)
        self.thread=threading.Thread(target=self._heartbeat,name='dense-health',daemon=True);self.thread.start()

    def _emit(self,values):
        with self.lock:
            with (self.output/'telemetry.jsonl').open('a') as stream:stream.write(json.dumps(values,sort_keys=True)+'\n')
            try:self.run.log(values);self.last_error=None
            except Exception as error:self.last_error=type(error).__name__
            atomic_json(self.output/'logging-status.json',{'phase':self.phase,'updated_at_unix':time.time(),
                'last_queued_update':self.latest.get('optimizer_updates'),'sdk_queue_error':self.last_error,
                'note':'SDK queue success is not a per-packet remote-delivery receipt'})

    def _heartbeat(self):
        while not self.stop.wait(60):
            try:
                info=diagnostics(self.loaders);atomic_json(self.output/'health.json',info)
                values={'heartbeat/unix_time':time.time(),'lifecycle/phase':self.phase}
                for key in ('memory.current','memory.peak'):
                    value=info.get('cgroup/'+key)
                    if isinstance(value,str) and value.isdigit():values['system/'+key]=int(value)
                self._emit(values)
            except Exception as error:
                print('H200_DENSE_TELEMETRY_WARNING='+type(error).__name__,flush=True)

    def stage(self,name):
        self.phase=name;self._emit({'lifecycle/phase':name,'heartbeat/unix_time':time.time()})
        print('H200_DENSE_STAGE='+name,flush=True)

    def attach(self,train,val,contract):
        self.loaders=(train,val)
        try:
            self.run.config.update({'runtime_contract_sha256':contract_hash(contract),
                'runtime_versions':contract['runtime_versions'],'runtime_source_sha256':contract['runtime_source_sha256'],
                'backbone_provenance':contract.get('backbone')},allow_val_change=True)
        except Exception as error:self.last_error=type(error).__name__

    def summary(self,values):
        try:self.run.summary.update(values)
        except Exception as error:self.last_error=type(error).__name__

    def progress(self,progress,metrics=None,*,force=False):
        self.latest=asdict(progress);update=progress.optimizer_updates
        if not force and update%20:return
        now=time.monotonic();values={'optimizer_updates':update,'train/epoch':progress.epoch,'train/micro_steps':progress.micro_steps}
        if metrics:values.update({'train/'+k:v for k,v in metrics.items()})
        if self.last_update is not None and update>self.last_update:
            values['train/seconds_per_update']=(now-self.last_time)/(update-self.last_update)
        self.last_update=update;self.last_time=now;self._emit(values)
        if force or update%1000==0:
            print('H200_DENSE_PROGRESS='+json.dumps({'u':update,'ep':progress.epoch,'loss':None if not metrics else round(metrics['loss'],6)}),flush=True)

    def checkpoint(self,progress,path,state):
        self.summary({'last_saved_update':progress.optimizer_updates,'checkpoint_path':str(path),'checkpoint_state':state})

    def fail(self,error):
        info=diagnostics(self.loaders)
        path=self.output/'status/progress.json'
        try:saved=json.loads(path.read_text()).get('progress') if path.exists() else None
        except (OSError,ValueError):saved=None
        payload={'state':'failed','error_type':type(error).__name__,'message':str(error)[:2000],
                 'observed_progress':self.latest,'last_saved_progress':saved,'diagnostics':info,
                 'checkpoint_exists':(self.output/'checkpoints/last.pt').is_file(),
                 'last_checkpoint':str(self.output/'checkpoints/last.pt')}
        atomic_json(self.output/'failure.json',payload)
        print('H200_DENSE_FAILURE='+json.dumps(payload,sort_keys=True),flush=True)
        self.summary({'lifecycle_state':'failed','error_type':type(error).__name__,'failure_diagnostics':info})

    def close(self,success,final=None):
        self.stop.set();self.thread.join(timeout=7)
        if final:
            self.summary({'final/'+k:v for k,v in final.get('evaluation',{}).items()})
            self.summary({'completed_updates':final.get('progress',{}).get('optimizer_updates')})
        self.summary({'lifecycle_state':'completed' if success else 'failed_or_paused'})
        try:self.run.finish(exit_code=0 if success else 1)
        except Exception as error:print('H200_DENSE_WANDB_FINISH_WARNING='+type(error).__name__,flush=True)
        finally:
            for loader in self.loaders:
                iterator=getattr(loader,'_iterator',None)
                if iterator is not None:iterator._shutdown_workers()


def canary():
    run=open_run('canary');now=int(time.time())
    run.log({'relay_canary/ok':1,'relay_canary/unix_time':now},step=now)
    run.summary['relay_canary_status']='ok';url=run.url;run.finish()
    print('H200_DENSE_RELAY_CANARY_OK='+url,flush=True)

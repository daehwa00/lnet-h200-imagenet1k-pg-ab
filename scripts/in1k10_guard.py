"""Independent watchdog from bootstrap through final external checkpoint ack."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from in1k10_transport import API,atomic,read


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code-sha',required=True)
    p.add_argument('--grace-seconds',type=int,default=900);p.add_argument('--stall-seconds',type=int,default=1200)
    p.add_argument('--poll-seconds',type=float,default=5)
    p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    if args.command[:1]==['--']:args.command=args.command[1:]
    if not args.command:raise ValueError('Missing supervised command')
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    token=secrets.token_hex(32);api=API(token)
    response=api.call('/enroll',{'pod':os.uname().nodename,'token_hash':hashlib.sha256(token.encode()).hexdigest(),'code_sha':args.code_sha})
    api.session=response['session_id'];atomic(root/'session.json',{'id':api.session,'code_sha':args.code_sha})
    print('IN10_SESSION='+api.session,flush=True)
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    seq=offset=printed=0;last_contact=time.monotonic();deadline=None;forced=False;last_signature=None;last_progress=time.monotonic()
    transfer=None;uploaded={};restored=set();last_error=None;started=time.monotonic();drain_deadline=None;backup_failure_since=None
    def begin_transfer(kind,job,**values):
        result=root/f'transfer-{secrets.token_hex(6)}.json'
        process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('in1k10_transfer.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        payload={'token':token,'session':api.session,'kind':kind,'job':job,'result':str(result),**values}
        process.stdin.write(json.dumps(payload).encode());process.stdin.close()
        return {'process':process,'kind':kind,'job':job,'result':result,'started':time.monotonic()}
    with (root/'console.log').open('ab',buffering=0) as console:
        child=subprocess.Popen(args.command,stdout=console,stderr=subprocess.STDOUT,start_new_session=True,
                               env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1'))
        try:
            while True:
                now=time.monotonic();current=read(root/'current.json');job=current.get('job')
                control={}
                try:
                    control=api.call('/control');last_contact=now;atomic(root/'control.json',control)
                    stopped|=control.get('stop',False)
                    if control.get('force'):forced=True
                except Exception as exc:last_error='control:'+type(exc).__name__
                if now-last_contact>600:stopped=True;last_error='control_timeout'
                seen=control.get('observer_seen',0)/1000
                if now-started>600 and time.time()-seen>600:stopped=True;last_error='observer_timeout'
                output=Path(current['output']) if current.get('output') else None
                progress=read(output/'step-progress.json') if output else {}
                checkpoint=read(output/'checkpoint-meta.json') if output else {}
                result=read(output/'result.json') if output else {}
                signature=(job,current.get('stage'),progress.get('global_step'),checkpoint.get('epoch'))
                if signature!=last_signature:last_signature=signature;last_progress=now
                if now-last_progress>args.stall_seconds:stopped=True;last_error='progress_stalled'
                if transfer:
                    proc=transfer['process']
                    if now-transfer['started']>1200:
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    if proc.poll() is not None:
                        receipt=read(transfer['result'])
                        if receipt.get('ok'):
                            if transfer['kind']=='restore':
                                atomic(root/f'restored-{transfer["job"]}.json',{'done':True,'artifact':receipt['value']})
                                restored.add(transfer['job'])
                            elif receipt['value']:
                                uploaded[transfer['job']]=receipt['value'];atomic(root/'uploaded.json',uploaded)
                                backup_failure_since=None
                        else:
                            last_error='transfer:'+receipt.get('error','timeout_or_failure')
                            if transfer['kind']=='upload' and backup_failure_since is None:backup_failure_since=now
                        transfer=None
                if backup_failure_since is not None and now-backup_failure_since>600:
                    stopped=True;last_error='checkpoint_backup_unavailable'
                if job and current.get('stage')=='restore' and job not in restored and transfer is None and not stopped:
                    transfer=begin_transfer('restore',job,target=str(output/'checkpoint.pt'))
                if job and checkpoint.get('phase')=='full' and transfer is None and not forced:
                    prior=uploaded.get(job,{}).get('epoch',0);epoch=checkpoint.get('epoch',0)
                    if epoch>prior and (epoch%10==0 or stopped or bool(result)):
                        transfer=begin_transfer('upload',job,meta=checkpoint)
                if job and current.get('stage')=='wait_backup':
                    try:
                        meta=api.call(f'/artifact/{job}/latest')
                        if meta and meta['epoch']==100 and meta['verified']:
                            atomic(root/f'backup-ack-{job}.json',meta)
                    except Exception as exc:last_error='backup_ack:'+type(exc).__name__
                if stopped and deadline is None:
                    (root/'STOP').touch();deadline=now+args.grace_seconds
                    # Signal only the supervisor. It lets the current epoch finish;
                    # DataLoader workers must not receive an early SIGTERM.
                    try:os.kill(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                if forced or (deadline is not None and now>=deadline):
                    forced=True
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    if transfer:
                        try:os.killpg(transfer['process'].pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                status={'time':time.time(),'current':current,'progress':progress,'checkpoint':checkpoint,
                        'result':{k:result.get(k) for k in ('status','completed_epochs','global_step','final_validation')},
                        'exit_code':child.poll(),'stop_requested':stopped,'forced':forced,'error':last_error,
                        'ended':child.poll() is not None and transfer is None}
                for name in ('memory.current','memory.max','memory.events'):
                    try:status[name]=Path('/sys/fs/cgroup',name).read_text().strip()
                    except OSError:pass
                atomic(root/'guard-status.json',status)
                with (root/'console.log').open('rb') as stream:
                    stream.seek(printed);visible=stream.read(16000)
                if visible:
                    sys.stdout.write(visible.decode('utf-8',errors='replace'));sys.stdout.flush();printed+=len(visible)
                try:
                    with (root/'console.log').open('rb') as stream:stream.seek(offset);chunk=stream.read(16000)
                    api.call('/events',{'seq':seq,'logs':chunk.decode('utf-8',errors='replace'),'status':status})
                    offset+=len(chunk);seq+=1
                except Exception as exc:last_error='events:'+type(exc).__name__
                if child.poll() is not None:
                    if drain_deadline is None:drain_deadline=now+1260
                    if (transfer is None and min(offset,printed)>=(root/'console.log').stat().st_size) or now>=drain_deadline:break
                time.sleep(args.poll_seconds)
        finally:
            if transfer:
                try:os.killpg(transfer['process'].pid,signal.SIGKILL)
                except ProcessLookupError:pass
                transfer['process'].wait(timeout=10)
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=child.wait(timeout=30);atomic(root/'exit.json',{'code':code,'forced':forced,'stopped':stopped})
        return 130 if stopped or forced else code


if __name__=='__main__':sys.exit(main())

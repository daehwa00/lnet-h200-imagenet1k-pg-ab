"""Owner-only isolated live canary. Never enrolls a production H200 session."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import wandb
from in1k10_transport import API,CHUNK,atomic,upload,download,read

class TestAPI(API):
    def call(self,path,*args,**kwargs):
        return super().call(path+('&' if '?' in path else '?')+'test=1',*args,**kwargs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--secrets',type=Path,required=True)
    args=p.parse_args();key=read(args.secrets)['IN10_OWNER_TOKEN'];api=TestAPI(key)
    api.call('/command',{'action':'release'});api.call('/command',{'action':'arm'})
    session=api.call('/enroll',{'pod':'job-daehwa00-0-canary','code_sha':'0'*40,
                              'token_hash':hashlib.sha256(b'isolated-test-agent').hexdigest()})
    api.session=session['session_id']
    api.call('/events',{'seq':0,'logs':'IN10 isolated transport canary','status':{'stage':'test','training_started':False}})
    snapshot=api.call('/snapshot?after=0');assert snapshot['events']
    with tempfile.TemporaryDirectory(prefix='in10-network-canary-',dir=args.root) as folder:
        source=Path(folder)/'source.bin';source.write_bytes(os.urandom(768000))
        info=source.stat();meta={'path':str(source),'inode':info.st_ino,'bytes':info.st_size,'mtime_ns':info.st_mtime_ns,'epoch':3,'workers':0}
        sent=upload(api,'va_k96-501',meta)
        available=api.call('/artifact/va_k96-501/latest')
        digest=download(api,'va_k96-501',available,Path(folder)/'download.bin')
        assert digest==sent['sha'] and (Path(folder)/'download.bin').read_bytes()==source.read_bytes()
        api.call(f'/artifact/va_k96-501/{digest}/ack',{})
    api.call('/command',{'action':'stop'})
    assert api.call('/control')['stop']
    api.call('/command',{'action':'release'})
    run_id='in10canary20260920'
    run=wandb.init(entity='daehwa',project='alphabet2d-imagenet1k-10pct',group='integration-canary',
        id=run_id,resume='allow',name='IN10-control-canary-not-training',
        config={'training_started':False,'synthetic_payload':True},
        settings=wandb.Settings(console='off',disable_code=True,disable_git=True,x_disable_stats=True,init_timeout=45))
    marker=str(time.time_ns());run.log({'canary/transport_ok':1});run.summary['transport_canary']=marker;run.finish()
    remote=wandb.Api(timeout=15).run('daehwa/alphabet2d-imagenet1k-10pct/'+run_id)
    assert remote.summary.get('transport_canary')==marker
    report={'live_transport':True,'live_stop_latch':True,'synthetic_payload_bytes':768000,'checkpoint_chunk_bytes':CHUNK,
        'download_sha256_verified':True,'wandb_remote_summary_verified':True,'wandb_run':run_id,
        'production_training_started':False,'production_session_enrolled':False}
    atomic(args.root/'live-canary.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()

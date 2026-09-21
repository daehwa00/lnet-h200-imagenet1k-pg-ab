"""Readiness/final acknowledgment survives transient relay errors, without GPUs."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.error import HTTPError
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import in1k10_observer as observer


@pytest.mark.parametrize('failed_action',['ready','finish'])
def test_ack_retries_without_losing_completed_job(tmp_path,monkeypatch,failed_action):
    class Finished(BaseException):pass
    clock=[1000.];initializations=[];commands=[];summary={}
    class Run:
        id='fake-run'
        def __init__(self):self.summary=summary
        def define_metric(self,*a,**kw):pass
        def log(self,payload):pass
        def finish(self,**kw):pass
    def init(**kw):initializations.append(kw);return Run()
    class WB:
        def run(self,path):return SimpleNamespace(summary=summary)
        def flush(self):pass
    job='va_k128-501';current={'job':job,'campaign_id':observer.CAMPAIGN,'stage':'waiting_wandb',
        'subset_sha256':'fixed','updates':25000,'batch_size':512,'workers':8}
    events=[{'id':1,'value':json.dumps({'status':{'current':current}})},
        {'id':2,'value':json.dumps({'status':{'current':dict(current,stage='waiting_metrics'),
            'result':{'status':'completed','completed_epochs':100,'global_step':25000,'final_validation':{'accuracy':.5}}}})}]
    class API:
        def __init__(self,*a):self.calls=0;self.failed=False
        def call(self,path,value=None,**kw):
            assert kw['attempts']==1
            if path.startswith('/snapshot'):
                self.calls+=1
                if self.calls>2:raise Finished()
                cursor=int(path.split('=')[1])
                return {'session':{'id':'session','code':'a'*40,'ended':False},'control':{'ready_jobs':[]},
                    'events':[e for e in events if e['id']>cursor]}
            if value['action']==failed_action and not self.failed:
                self.failed=True
                raise HTTPError('https://example.invalid',429,'rate limited',{'Retry-After':'30'},io.BytesIO(b'limited'))
            commands.append(value['action'])
            return {}
    secret=tmp_path/'private.json';secret.write_text(json.dumps({'IN10_OWNER_TOKEN':'fixture'}))
    monkeypatch.setattr(observer,'API',API)
    monkeypatch.setattr(observer.wandb,'init',init)
    monkeypatch.setattr(observer.wandb,'Api',lambda **kw:WB())
    monkeypatch.setattr(observer.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(observer.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(sys,'argv',['observer','--root',str(tmp_path),'--secrets',str(secret)])
    with pytest.raises(Finished):observer.main()
    state=json.loads((tmp_path/'observer-state-networkfix-v4.json').read_text())
    assert set(state['completed'])=={'va_k96-501','va_k128-501'}
    assert state['cursor']==2 and commands.count('finish')==1
    assert len(initializations)==1

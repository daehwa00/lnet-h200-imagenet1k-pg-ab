import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import in1k10_subset as subset
import in1k10_worker as worker
import in1k10_guard as guard


def test_wrong_subset_rejected():
    with pytest.raises(ValueError,match='digest'):subset.parse(b'n01440764_1.JPEG\n')


def test_numpy_ipc_preserves_values_and_labels():
    batch=[(torch.arange(12,dtype=torch.float32).reshape(3,2,2),4),
           (torch.ones(3,2,2),9)]
    images,labels=worker.numpy_collate(batch)
    assert isinstance(images,np.ndarray) and isinstance(labels,np.ndarray)
    x,y=next(iter(worker.ArrayLoader([(images,labels)],False)))
    torch.testing.assert_close(x,torch.stack([b[0] for b in batch]),rtol=0,atol=0)
    assert y.tolist()==[4,9]


def test_subset_wrapper_never_resamples_by_seed(tmp_path,monkeypatch):
    names=['n00000001_2.JPEG','n00000001_1.JPEG','n00000002_1.JPEG']
    (tmp_path/'simclr-10percent.txt').write_text('fixture')
    manifest=tmp_path/'subset-manifest.json'
    manifest.write_text(json.dumps({'class_to_idx':{'n00000001':0,'n00000002':1}}))
    monkeypatch.setattr(worker,'parse',lambda _: (names,{}))
    a=worker.FixedTrainingSubset(tmp_path/'train',manifest,None)
    torch.manual_seed(521)
    b=worker.FixedTrainingSubset(tmp_path/'train',manifest,None)
    assert a.samples==b.samples and a.targets==[0,0,1]


def test_watchdog_forces_unresponsive_child_without_github(tmp_path,monkeypatch):
    class FakeAPI:
        def __init__(self,token):self.polls=0;self.session=''
        def call(self,path,value=None,**kwargs):
            if path=='/enroll':return {'session_id':'test-session'}
            if path=='/control':
                self.polls+=1
                return {'stop':self.polls>=5,'force':False,'observer_seen':time.time()*1000}
            return {'ok':True}
    monkeypatch.setattr(guard,'API',FakeAPI)
    monkeypatch.setattr(sys,'argv',['guard','--root',str(tmp_path),'--code-sha','a'*40,
        '--grace-seconds','1','--poll-seconds','.05','--',sys.executable,'-u','-c',
        'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("visible"); time.sleep(30)'])
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    try:assert guard.main()==130
    finally:
        for s,h in previous.items():signal.signal(s,h)
    result=json.loads((tmp_path/'exit.json').read_text())
    assert result['forced'] and result['code']==-signal.SIGKILL
    assert 'visible' in (tmp_path/'console.log').read_text()

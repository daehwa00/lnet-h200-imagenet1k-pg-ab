"""Watchdog decisions under simulated time/network; no GPU or live requests."""
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import in1k10_guard as guard


def run_guard(monkeypatch,tmp_path,*,duration=650,progress=True,stage='training',control=None,local_stop=False,event_outage=False):
    clock=[1000.];killed=[];calls=[]
    monkeypatch.setattr(guard.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(guard.time,'time',lambda:clock[0]+10000)
    monkeypatch.setattr(guard.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(guard.signal,'signal',lambda *args:None)
    monkeypatch.setattr(guard.os,'kill',lambda pid,sig:killed.append(sig))
    monkeypatch.setattr(guard.os,'killpg',lambda pid,sig:killed.append(sig))
    class Child:
        pid=123456
        returncode=None
        def poll(self):
            if self.returncode is None:
                self.returncode=-9 if signal.SIGKILL in killed else 0 if clock[0]>=1000+duration else None
            return self.returncode
        def wait(self,timeout):return self.poll()
    def launch(*args,**kwargs):
        kwargs['stdout'].write(b'child-visible-output\n')
        return Child()
    monkeypatch.setattr(guard.subprocess,'Popen',launch)
    monkeypatch.setattr(guard.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='',returncode=0))
    class FakeAPI:
        session=''
        def __init__(self,token):pass
        def call(self,path,value=None,**kwargs):
            calls.append((path,clock[0]))
            assert kwargs['attempts']==1 and kwargs['timeout']==8
            if path=='/enroll':return {'session_id':'test'}
            if path=='/events':
                if event_outage:raise ConnectionError('event outage')
                return {'ok':True}
            if control:return control(clock[0]-1000)
            raise ConnectionError('network outage')
    monkeypatch.setattr(guard,'API',FakeAPI)
    original_read=guard.read
    def read(path):
        if Path(path).name=='step-progress.json':return {'global_step':int(clock[0]) if progress else 1}
        return original_read(path)
    monkeypatch.setattr(guard,'read',read)
    guard.atomic(tmp_path/'current.json',{'job':'va_k128-501','stage':stage,'output':str(tmp_path/'run')})
    if local_stop:(tmp_path/'STOP').touch()
    monkeypatch.setattr(sys,'argv',['guard','--root',str(tmp_path),'--code-sha','a'*40,
        '--grace-seconds','1','--stall-seconds','10','--','fake-child'])
    result=guard.main()
    assert clock[0]<=1000+duration+60
    return result,json.loads((tmp_path/'exit.json').read_text()),calls


def test_long_control_outage_does_not_stop_progressing_training(monkeypatch,tmp_path,capsys):
    code,summary,calls=run_guard(monkeypatch,tmp_path)
    assert code==0 and not summary['stopped']
    assert not (tmp_path/'STOP').exists()
    assert 'control_unavailable' in capsys.readouterr().out
    assert len([p for p,t in calls if p=='/control'])<10


def test_single_failure_retains_last_valid_control(monkeypatch,tmp_path):
    def control(elapsed):
        if 15<=elapsed<30:raise ConnectionError('one failure')
        return {'observer_seen':11000*1000,'ready_jobs':['va_k128-501']}
    code,summary,_=run_guard(monkeypatch,tmp_path,duration=45,control=control)
    assert code==0 and not summary['stopped']
    assert guard.read(tmp_path/'control.json')['ready_jobs']==['va_k128-501']


@pytest.mark.parametrize('stage',['waiting_wandb','waiting_metrics'])
def test_external_wait_is_not_compute_stall(monkeypatch,tmp_path,stage):
    code,summary,_=run_guard(monkeypatch,tmp_path,duration=650,progress=False,stage=stage)
    assert code==0 and not summary['stopped']


def test_real_compute_stall_stops_and_reports_locally(monkeypatch,tmp_path,capsys):
    code,summary,_=run_guard(monkeypatch,tmp_path,duration=650,progress=False)
    assert code==130 and summary['forced']
    assert 'progress_stalled' in summary['stop_reasons']
    assert 'progress_stalled' in capsys.readouterr().out
    assert 'progress_stalled' in (tmp_path/'guard-events.jsonl').read_text()


@pytest.mark.parametrize('force',[False,True])
def test_explicit_remote_stop_and_force_work(monkeypatch,tmp_path,force):
    code,summary,_=run_guard(monkeypatch,tmp_path,control=lambda t:{'stop':not force,'force':force})
    assert code==130
    assert ('remote_force' if force else 'remote_stop') in summary['stop_reasons']


def test_local_stop_file_works_without_network(monkeypatch,tmp_path):
    code,summary,_=run_guard(monkeypatch,tmp_path,local_stop=True)
    assert code==130 and 'local_STOP' in summary['stop_reasons']


def test_events_outage_keeps_stdout_and_bounded_shutdown(monkeypatch,tmp_path,capsys):
    code,summary,_=run_guard(monkeypatch,tmp_path,duration=30,event_outage=True)
    assert code==0 and not summary['stopped']
    assert summary['relay_offset']==0 and summary['console_bytes']>0
    assert 'child-visible-output' in capsys.readouterr().out
    assert 'child-visible-output' in (tmp_path/'console.log').read_text()

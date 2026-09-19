import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('guard',Path(__file__).parents[1]/'scripts/h200_http_guard.py')
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def args(tmp_path, code):
    return SimpleNamespace(url='http://unused',root=tmp_path,command=[sys.executable,'-u','-c',code],
                           disconnect_seconds=.15,grace_seconds=.1,poll_seconds=.02)


@pytest.fixture(autouse=True)
def restore_signals(monkeypatch):
    old={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    monkeypatch.setenv('H200_AGENT_TOKEN','test-only')
    yield
    for s,h in old.items():signal.signal(s,h)


def test_exit_and_log_capture(tmp_path,monkeypatch):
    events=[]
    def request(base,token,route,value=None):
        if value:events.append(value)
        return {'stop':False}
    monkeypatch.setattr(guard,'request',request)
    assert guard.supervise(args(tmp_path,"print('visible error'); raise SystemExit(7)"))==7
    assert 'visible error' in (tmp_path/'console.log').read_text()
    assert any('visible error' in e['text'] for e in events)
    assert json.loads((tmp_path/'exit.json').read_text())['exit_code']==7


def test_refuse_stopped_before_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(guard,'request',lambda *a,**k:{'stop':True})
    with pytest.raises(RuntimeError,match='stopped'):
        guard.supervise(args(tmp_path,'raise Exception()'))
    assert not (tmp_path/'console.log').exists()


def test_force_stop_unresponsive_child(tmp_path,monkeypatch):
    polls=0
    def request(base,token,route,value=None):
        nonlocal polls
        if route=='/control':polls+=1
        return {'stop':polls>12}
    monkeypatch.setattr(guard,'request',request)
    code='import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready"); time.sleep(20)'
    assert guard.supervise(args(tmp_path,code))==130
    state=json.loads((tmp_path/'exit.json').read_text())
    assert state['forced'] and state['exit_code']==-signal.SIGKILL

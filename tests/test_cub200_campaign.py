import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import pytest

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from cub200_data import manifest
from prepare_cub200 import safe_members
from cub200_supervisor import supervise


def fixture(root):
    (root/'images'/'class_a').mkdir(parents=True)
    (root/'images'/'class_a'/'a.jpg').write_bytes(b'fixture')
    (root/'images'/'class_a'/'b.jpg').write_bytes(b'fixture')
    (root/'images.txt').write_text('1 class_a/a.jpg\n2 class_a/b.jpg\n')
    (root/'image_class_labels.txt').write_text('1 1\n2 1\n')
    (root/'train_test_split.txt').write_text('1 1\n2 0\n')


def test_split_identity_and_no_leakage(tmp_path):
    fixture(tmp_path)
    info=manifest(tmp_path,official_counts=False)
    assert (info['train'],info['test'])==(1,1)
    assert {r[0] for r in info['records'] if r[3]} .isdisjoint({r[0] for r in info['records'] if not r[3]})
    assert info['split_sha256']==manifest(tmp_path,official_counts=False)['split_sha256']
    with pytest.raises(ValueError,match='official'):manifest(tmp_path)


def test_bad_split_and_path(tmp_path):
    fixture(tmp_path)
    (tmp_path/'train_test_split.txt').write_text('1 1\n2 7\n')
    with pytest.raises(ValueError):manifest(tmp_path,official_counts=False)
    (tmp_path/'train_test_split.txt').write_text('1 1\n2 0\n')
    (tmp_path/'images.txt').write_text('1 ../secret\n2 class_a/b.jpg\n')
    with pytest.raises(ValueError):manifest(tmp_path,official_counts=False)


@pytest.mark.parametrize('name,kind',[('../escape',tarfile.REGTYPE),('CUB_200_2011/link',tarfile.SYMTYPE)])
def test_unsafe_tar(name,kind):
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        entry=tarfile.TarInfo(name);entry.type=kind;archive.addfile(entry)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer,mode='r') as archive:
        with pytest.raises(ValueError):safe_members(archive)


class Run:
    def __init__(self): self.summary={};self.events=[]
    def save(self,*args,**kwargs): pass
    def log(self,row): self.events.append(row)


def test_worker_error_is_not_success(tmp_path):
    run=Run()
    with pytest.raises(RuntimeError,match='exited 7'):
        supervise([sys.executable,'-c','print("visible failure"); raise SystemExit(7)'],tmp_path,run,tmp_path/'STOP',grace=.1,stall=10)
    assert 'visible failure' in (tmp_path/'console.log').read_text()
    assert json.loads((tmp_path/'supervisor.json').read_text())['exit_code']==7


def test_stop_marker_stops_child_and_queue(tmp_path):
    (tmp_path/'STOP').touch()
    assert not supervise([sys.executable,'-c','import time; time.sleep(60)'],tmp_path,Run(),tmp_path/'STOP',grace=.1,stall=10)
    assert json.loads((tmp_path/'supervisor.json').read_text())['state']=='stopped'


def test_stall_forces_unresponsive_child(tmp_path,monkeypatch):
    import cub200_supervisor as module
    original_sleep=module.time.sleep
    monkeypatch.setattr(module.time,'sleep',lambda n:original_sleep(.02))
    code='import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'
    assert not supervise([sys.executable,'-u','-c',code],tmp_path,Run(),tmp_path/'STOP',grace=.1,stall=.4)
    state=json.loads((tmp_path/'supervisor.json').read_text())
    assert state['forced'] and state['exit_code']==-9


def test_completed_endpoint_rejects_wrong_identity(tmp_path):
    from cub200_supervisor import completed_endpoint
    (tmp_path/'result.json').write_text(json.dumps({'status':'completed','epoch':100,'model':'wrong'}))
    with pytest.raises(RuntimeError):completed_endpoint(tmp_path,'va_k96',501,'source')


def test_telemetry_incremental_read(tmp_path,monkeypatch):
    import cub200_supervisor as module
    original_sleep=module.time.sleep
    monkeypatch.setattr(module.time,'sleep',lambda n:original_sleep(.02))
    code='import time,pathlib,json; pathlib.Path('+repr(str(tmp_path/'telemetry.jsonl'))+').write_text(json.dumps({"kind":"train","epoch":1,"global_step":5,"metrics":{"loss":1.2}})+"\\n"); time.sleep(.15)'
    run=Run()
    assert supervise([sys.executable,'-c',code],tmp_path,run,tmp_path/'STOP',grace=.1,stall=10)
    assert len(run.events)==1 and run.events[0]['train/loss']==1.2


def test_recipe_schedule_and_counts():
    import run_h200_baseline_worker as shared
    assert shared._learning_rate(.003,119,24,100,5)==pytest.approx(.003)
    assert shared._learning_rate(.003,2399,24,100,5)==pytest.approx(0)
    config=json.loads((SCRIPTS.parent/'h200/cub200/campaign.json').read_text())
    assert len(config['models'])*len(config['seeds'])==15
    assert config['pretrained'] is False and config['test_policy']=='final_epoch_only_no_selection'


def test_epoch_resume_preserves_cpu_trajectory(tmp_path,monkeypatch):
    import copy
    from types import SimpleNamespace
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    import run_h200_baseline_worker as common
    from train_cub200 import capture_rng,restore_rng
    torch.set_num_threads(2)
    monkeypatch.setattr(common,'NUM_CLASSES',200)
    monkeypatch.setattr(common.registry,'model_spec',lambda k:SimpleNamespace(precision='bfloat16'))
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda x:None)
    common._seed_everything(501)
    data=TensorDataset(torch.randn(32,3,4,4),torch.arange(32)%200)
    gen=torch.Generator().manual_seed(501)
    loader=DataLoader(data,batch_size=8,shuffle=True,generator=gen)
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(48,200))
    opt=common._build_optimizer(model,.003,torch.device('cpu'))
    task=SimpleNamespace(model_key='toy',gradient_accumulation_steps=2,learning_rate=.003,
                         epochs=100,phase='full',max_steps=None,output_dir=tmp_path)
    mix=common._make_mixup()
    common._train_one_epoch(model,loader,opt,mix,torch.device('cpu'),task,epoch=1,global_step=0)
    saved=copy.deepcopy({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':capture_rng(gen)})
    common._train_one_epoch(model,loader,opt,mix,torch.device('cpu'),task,epoch=2,global_step=2)
    expected=copy.deepcopy(model.state_dict())
    model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],gen)
    common._train_one_epoch(model,loader,opt,mix,torch.device('cpu'),task,epoch=2,global_step=2)
    assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())

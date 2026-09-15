import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset
from dense_transfer import data,engine
import run_dense_transfer as runner
import run_h200_dense_v2 as h200
from dense_transfer.telemetry import diagnostics,Telemetry


def test_loader_profile_respects_memory_and_shm():
    assert h200.loader_profile(32*1024**3,2*1024**3)['workers']==2
    assert h200.loader_profile(8*1024**3,2*1024**3)['workers']==0
    assert h200.loader_profile(None,64*1024**2)['workers']==0


def test_seed_and_ipc_profile_are_bound_to_contract(tmp_path):
    args=runner.parser().parse_args(['--task','coco','--model','va_k128','--checkpoint',str(tmp_path/'seed501_ep100.pt'),
        '--data-root',str(tmp_path),'--output-root',str(tmp_path),'--seed','501','--prefetch-factor','1',
        '--sharing-strategy','file_system'])
    settings=runner.settings_from_args(args)
    assert settings.seed==501 and settings.pin_memory and settings.prefetch_factor==1
    assert settings.sharing_strategy=='file_system'
    assert settings.physical_batch_size==2 and settings.effective_batch_size==16
    assert runner.plan_from_args(args).epochs==12


def test_multiprocess_named_storage_loader(tmp_path,monkeypatch):
    old=torch.multiprocessing.get_sharing_strategy()
    monkeypatch.setenv('LNET_MP_SHARING_STRATEGY',old)
    fixture=TensorDataset(torch.arange(8.).view(8,1,1,1).expand(8,3,4,4).contiguous(),torch.arange(8))
    monkeypatch.setattr(data,'build_dataset',lambda *a,**k:fixture)
    bundle=data.build_loaders('coco',tmp_path,workers=2,pin_memory=False,prefetch_factor=1,sharing_strategy='file_system')
    try:
        seen=[]
        for images,targets in bundle.train:
            seen.extend(int(x) for x in targets)
        assert sorted(seen)==list(range(8))
        assert bundle.train.prefetch_factor==1
        assert torch.multiprocessing.get_sharing_strategy()=='file_system'
    finally:
        if bundle.train._iterator:bundle.train._iterator._shutdown_workers()
        torch.multiprocessing.set_sharing_strategy(old)


def test_metrics_callback_only_reports_completed_updates():
    model=torch.nn.Linear(1,1);opt=torch.optim.SGD(model.parameters(),lr=.01)
    plan=engine.OptimizerPlan(epochs=1,learning_rate=.01,weight_decay=0,warmup_updates=0)
    scheduler=engine.UpdateScheduler(opt,plan,total_updates=2);seen=[]
    engine.train_batches(model,[(torch.ones(1,1),torch.zeros(1,1))]*3,opt,scheduler,torch.device('cpu'),2,
        bf16=False,loss_fn=lambda y,t:((y-t)**2).mean(),on_metrics=lambda p,m:seen.append((p.optimizer_updates,m['loss'])))
    assert [n for n,_ in seen]==[1,2]
    assert all(torch.isfinite(torch.tensor(v)) for _,v in seen)


def contracts():
    old={'schema':'dense-transfer.engine.v1','settings':{'task':'coco','model_key':'va_k128','seed':501,
        'workers':4,'output_root':'old','physical_batch_size':2,'effective_batch_size':16},
        'optimizer':{'epochs':12,'learning_rate':.0001},'dataset':{'sha':'same'},
        'runtime_source_sha256':{'engine':'old','data':'old','backbones':'old','models':'m','metrics':'x'},
        'backbone':{'pretrained_provenance':{'seed':501,'dense_kernel_sha256':'old'}}}
    new=copy.deepcopy(old);new['settings'].update(workers=2,output_root='new',pin_memory=True,prefetch_factor=1,sharing_strategy='file_system')
    new['backbone']['pretrained_provenance'].update(dense_kernel_sha256='new',dense_performance={'optimized':True})
    return old,new


def test_recovery_preserves_all_tensor_state(tmp_path):
    old,new=contracts();source=tmp_path/'old.pt'
    payload={'schema':'dense-transfer.engine.v1','contract':old,'contract_hash':engine.contract_hash(old),
        'model':{'w':torch.randn(3)},'optimizer':{'moment':torch.randn(3)},'scheduler':{'count':200},
        'rng':{'test':torch.arange(4)},'progress':{'epoch':0,'batch_in_epoch':1600,'micro_steps':1600,'optimizer_updates':200}}
    torch.save(payload,source)
    target,progress=h200.recover(source,{'contract':new,'contract_hash':engine.contract_hash(new)},tmp_path/'new')
    saved=torch.load(target,weights_only=False)
    for key in ('model','optimizer','scheduler','rng','progress'):assert h200.validator._same(payload[key],saved[key])
    assert source.is_file() and progress['optimizer_updates']==200


@pytest.mark.parametrize('field',['seed','physical_batch_size','effective_batch_size'])
def test_recovery_refuses_science_change(field):
    old,new=contracts();new['settings'][field]=999
    with pytest.raises(ValueError):h200.verify_science(old,new)


def test_diagnostics_excludes_credentials(monkeypatch):
    monkeypatch.setenv('WANDB_API_KEY','not-to-be-printed-test')
    value=json.dumps(diagnostics())
    assert 'not-to-be-printed-test' not in value and 'authkey' not in value


def test_telemetry_update_sampling_without_network(tmp_path):
    subject=Telemetry.__new__(Telemetry)
    subject.latest={};subject.last_update=None;subject.last_time=0
    seen=[];subject._emit=seen.append
    for update in (1,19,20,21,40):subject.progress(engine.TrainProgress(optimizer_updates=update),{'loss':.5})
    assert [r['optimizer_updates'] for r in seen]==[20,40]


@pytest.mark.parametrize('failure',['auth','gate'])
def test_startup_failure_never_launches_training(tmp_path,monkeypatch,failure):
    import sys
    events=[]
    class FakeTelemetry:
        def __init__(self,*a,**k):
            events.append('auth')
            if failure=='auth':raise RuntimeError('fixture auth failure')
        def stage(self,name):events.append(name)
        def fail(self,error):events.append('failed')
        def close(self,*a):events.append('closed')
    monkeypatch.setattr(h200,'Telemetry',FakeTelemetry)
    monkeypatch.setattr(h200,'diagnostics',lambda:{'fixture':True})
    monkeypatch.setattr(h200,'read_limits',lambda:h200.loader_profile(0,0))
    monkeypatch.setattr(h200.validator,'run_validation',lambda *_:{'status':'needs_attention','evidence':{}})
    monkeypatch.setattr(h200.training,'run',lambda *_:pytest.fail('training must not start'))
    monkeypatch.setattr(sys,'argv',['run','--root',str(tmp_path),'--data-root',str(tmp_path)])
    with pytest.raises(RuntimeError):h200.main()
    assert events[0]=='auth'


def test_canary_failure_has_actionable_diagnostics(tmp_path,monkeypatch):
    import sys
    monkeypatch.setattr(h200,'canary',lambda:(_ for _ in ()).throw(RuntimeError('fixture403')))
    monkeypatch.setattr(h200,'relay_rejection_probe',lambda:{'reason':'source_not_allowed'})
    monkeypatch.setattr(h200,'diagnostics',lambda:{'fixture':True})
    monkeypatch.setattr(sys,'argv',['run','--root',str(tmp_path),'--canary-only'])
    with pytest.raises(RuntimeError):h200.main()
    assert json.loads((tmp_path/'logging-preflight-failure.json').read_text())['relay_probe']['reason']=='source_not_allowed'


@pytest.mark.parametrize('role,checkpoint,expected',[('train',False,'never'),('train',True,'allow'),('canary',False,'allow')])
def test_native_sdk_resume_guard(monkeypatch,role,checkpoint,expected):
    import wandb
    from dense_transfer import telemetry
    captured={}
    monkeypatch.setattr(telemetry,'configure_relay',lambda *_:None)
    def init(**kwargs):
        captured.update(kwargs);return SimpleNamespace(url='https://wandb.ai/test')
    monkeypatch.setattr(wandb,'init',init)
    telemetry.open_run(role,has_checkpoint=checkpoint)
    assert captured['resume']==expected and captured['mode']=='online'


def test_training_starts_only_after_online_and_ready(tmp_path,monkeypatch):
    import sys
    events=[]
    class FakeTelemetry:
        def __init__(self,*a,**k):events.append('online')
        def stage(self,name):events.append(name)
        def fail(self,error):pytest.fail(str(error))
        def close(self,success,final):assert success;events.append('closed')
    def gate(args):
        path=tmp_path/'runs/coco-va_k128-seed501-v2/queue/readiness.json';path.parent.mkdir(parents=True)
        path.write_text('{}');events.append('gate')
        return {'status':'ready','evidence':{'continuation_parity_verified':True}}
    def train(args):
        assert events==['online','validating','gate','training']
        assert args.seed==501 and args.sharing_strategy=='file_system'
        path=tmp_path/'runs/coco-va_k128-seed501-v2/status/final.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'state':'completed','progress':{'optimizer_updates':88716}}))
        return {}
    monkeypatch.setattr(h200,'Telemetry',FakeTelemetry)
    monkeypatch.setattr(h200,'diagnostics',lambda:{})
    monkeypatch.setattr(h200,'read_limits',lambda:h200.loader_profile(0,0))
    monkeypatch.setattr(h200.validator,'run_validation',gate)
    monkeypatch.setattr(h200.training,'run',train)
    monkeypatch.setattr(h200.torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(sys,'argv',['run','--root',str(tmp_path),'--data-root',str(tmp_path)])
    h200.main();assert events[-1]=='closed'

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import resume_h200_k96_input as resume
import run_h200_baseline_worker as worker


def contract():
    return {'task':{'model_key':resume.MODEL,'phase':'full','seed':521,'epochs':100,
                     'batch_size':256,'workers':8,'learning_rate':.003},
            'source_sha256':{'worker':resume.LEGACY_WORKER_SHA,'imagenet1k_runner':resume.LEGACY_RUNNER_SHA},
            'model':{'key':resume.MODEL},'dataset':{'identity_sha256':'known'},
            'recipe':{'precision':'bfloat16','batch_size':256,'device_prefetch_scope':'copy_only'},
            'runtime':{'torch':'2.9.1+cu128','cuda_runtime':'12.8','torch_compile_mode':'default'}}


def test_missing_checkpoint_refuses_fresh_start(tmp_path,monkeypatch):
    monkeypatch.setattr(resume,'LEGACY_RUN',tmp_path/'missing')
    with pytest.raises(RuntimeError,match='refusing a fresh run'): resume.locate()


def test_legacy_checkpoint_bound_to_exact_source():
    c=contract()
    ck={'completed_epochs':18,'contract_sha256':worker._sha256_payload(c)}
    resume.validate_legacy(c,ck)
    c['source_sha256']['worker']='wrong'
    with pytest.raises(RuntimeError,match='source fingerprint'): resume.validate_legacy(c,ck)


@pytest.mark.parametrize('field,value',[('seed',509),('batch_size',128),('epochs',300),('learning_rate',.001)])
def test_scientific_changes_rejected(field,value):
    old=contract(); new=copy.deepcopy(old)
    new['task'][field]=value
    with pytest.raises(RuntimeError): worker.validate_input_migration(old,new)


def test_only_input_options_may_change():
    old=contract(); new=copy.deepcopy(old)
    new['recipe'].update(gpu_mixup=True,yield_before_fetch=True,device_prefetch_scope='copy_and_mixup')
    worker.validate_input_migration(old,new)
    new['recipe']['precision']='float32'
    with pytest.raises(RuntimeError,match='scientific recipe'): worker.validate_input_migration(old,new)


def test_finished_or_empty_checkpoint_is_not_restarted():
    c=contract()
    for epoch in (0,100):
        with pytest.raises(RuntimeError,match='unfinished checkpoint'):
            resume.validate_legacy(c,{'contract_sha256':worker._sha256_payload(c),'completed_epochs':epoch})

import importlib.util
import json
from pathlib import Path
import sys
import pytest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1]/'scripts/probe_k96_checkpoint.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_missing(tmp_path):
    result=probe.inspect(tmp_path)
    assert result['checkpoint_found'] is False and result['result_found'] is False
    assert list(tmp_path.iterdir()) == []


def test_saved_result_not_weights(tmp_path):
    (tmp_path/'result.json').write_text(json.dumps({'completed_epochs':100,'final_validation':{'accuracy':.72374}}))
    result=probe.inspect(tmp_path)
    assert result['saved_result']['top1_percent']==pytest.approx(72.374)
    assert not result['checkpoint_found']


def test_exact_checkpoint_metadata(monkeypatch,tmp_path):
    payload={'completed_epochs':100,'parameters':3253224,'contract_sha256':probe.CONTRACT,
        'history':[{'epoch':100,'validation':{'accuracy':.72374}}],
        'model':{'model.classifier.affine.linear.weight':SimpleNamespace(shape=(1000,512))}}
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(load=lambda *a,**k:payload))
    result=probe.checkpoint_metadata(tmp_path/'checkpoint.pt')
    assert result['expected_identity_match']
    assert result['saved_top1_percent']==pytest.approx(72.374)
    payload['model']['model.classifier.affine.linear.weight'].shape=(100,512)
    assert not probe.checkpoint_metadata(tmp_path/'checkpoint.pt')['expected_identity_match']

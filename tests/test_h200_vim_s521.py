import json
import hashlib
import zipfile
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_h200_vim_native as native
import run_h200_vim_tiny_s521 as runner


@pytest.mark.parametrize('preflight',[False,True])
def test_matched_recipe_and_preflight_isolation(tmp_path,monkeypatch,preflight):
    argv=['run','--data-root',str(tmp_path/'data'),'--source-root',str(tmp_path/'source'),'--output-root',str(tmp_path)]
    if preflight: argv.append('--preflight')
    monkeypatch.setattr(sys,'argv',argv)
    seen=[]
    monkeypatch.setattr(runner.worker,'run_task',lambda t:seen.append(t))
    monkeypatch.setattr(runner.os,'environ',dict(runner.os.environ))
    runner.main()
    task=seen[0]
    assert task.seed==521 and task.batch_size==256 and task.learning_rate==.003
    assert task.epochs==(1 if preflight else 100)
    assert task.wandb_mode==('disabled' if preflight else 'online')
    assert task.output_dir.name==('preflight' if preflight else 'seed_521')
    assert task.max_steps==(2 if preflight else None)
    assert 'H200_BASELINE_TORCH_COMPILE_MODE' not in runner.os.environ


def test_relay_identity_matches_runner():
    root=Path(__file__).resolve().parents[1]
    runtime=json.loads((root/'h200/baselines/wandb.runtime.json').read_text())
    run=runtime['runs'][runner.MODEL_KEY]['seeds']['521']
    assert run['id']==runner.RUN_ID
    assert run['tags']==runner.TAGS
    assert run['display_name']=='H200-Vim-Tiny-s521'


def test_native_checksum_rejects_wrong_download(tmp_path):
    source=tmp_path/'source.whl'
    source.write_bytes(b'wheel')
    dest=tmp_path/'cache'/'target.whl'
    with pytest.raises(RuntimeError,match='digest'):
        native.fetch_verified(source.as_uri(),'0'*64,dest)
    assert not dest.exists()
    native.fetch_verified(source.as_uri(),hashlib.sha256(b'wheel').hexdigest(),dest)
    assert dest.read_bytes()==b'wheel'


def test_extracts_only_single_root_scan_library(tmp_path):
    wheel=tmp_path/'mamba.whl'
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('selective_scan_cuda.cpython-313-x86_64-linux-gnu.so',b'native')
        archive.writestr('../unrelated',b'bad')
    out=native.extract_scan(wheel,tmp_path/'native')
    assert out.read_bytes()==b'native'
    assert not (tmp_path/'unrelated').exists()


def test_entrypoint_is_dedicated_and_preflight_first():
    root=Path(__file__).resolve().parents[1]
    source=(root/'h200/run_baselines.sh').read_text()
    assert 'refs/heads/control/imagenet1k-vim-tiny-s521' in source
    assert 'h200/vim/requirements.lock' in source
    assert source.index('--preflight')<source.index('H200_VIM_TINY_COMPLETE')
    lock=(root/'h200/vim/requirements.lock').read_text()
    assert 'https://download.pytorch.org/whl/cu128' in lock
    assert 'torch==2.9.1+cu128' in lock and 'transformers==5.14.1' in lock


def test_cuda_link_is_local_and_never_replaces_regular_files(tmp_path,monkeypatch):
    driver=tmp_path/'libcuda.so.1'
    driver.write_bytes(b'driver')
    monkeypatch.setattr(native.subprocess,'check_output',lambda *a,**k:f'libcuda.so.1 (libc6,x86-64) => {driver}\n')
    out=tmp_path/'native'
    out.mkdir()
    native.prepare_cuda_link(out)
    assert (out/'libcuda.so').resolve()==driver
    native.prepare_cuda_link(out)
    (out/'libcuda.so').unlink()
    (out/'libcuda.so').write_bytes(b'keep')
    with pytest.raises(RuntimeError,match='regular'): native.prepare_cuda_link(out)


def test_cuda_link_supports_container_driver_mount(tmp_path,monkeypatch):
    (tmp_path/'libcuda.so.1').write_bytes(b'driver')
    monkeypatch.setattr(native.subprocess,'check_output',lambda *a,**k:'')
    monkeypatch.setenv('LD_LIBRARY_PATH',str(tmp_path))
    out=tmp_path/'native'
    out.mkdir()
    native.prepare_cuda_link(out)
    assert (out/'libcuda.so').resolve()==tmp_path/'libcuda.so.1'

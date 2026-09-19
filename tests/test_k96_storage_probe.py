import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('storage',Path(__file__).parents[1]/'scripts/probe_k96_storage.py')
storage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def test_finds_weights_without_following_symlinks(tmp_path,capsys):
    model=tmp_path/'lnet-k96'/'seed_501'
    model.mkdir(parents=True)
    (model/'checkpoint.pt').write_bytes(b'not loaded')
    (model/'loop').symlink_to(tmp_path, target_is_directory=True)
    result=storage.scan(tmp_path)
    assert result['matching_files']==1
    assert 'K96_SYMLINK_NOT_FOLLOWED' in capsys.readouterr().out


def test_limit_and_pruning(tmp_path):
    (tmp_path/'datasets').mkdir()
    (tmp_path/'datasets'/'k96.pt').write_bytes(b'ignored')
    assert storage.scan(tmp_path)['matching_files']==0
    assert storage.scan(tmp_path,max_entries=0)['reason']=='search_limit_reached'


def test_mounts_are_filtered():
    text='1 0 0:1 / / rw - overlay overlay secret-option\n2 1 8:2 / /app/output rw - ext4 /dev/sdb rw\n3 1 8:3 / /unrelated rw - ext4 /dev/sdc rw'
    rows=storage.mount_summary(text)
    assert len(rows)==2
    assert 'secret' not in str(rows)
    assert not storage.owned_name('another-user')
    assert storage.owned_name('daehwa00')

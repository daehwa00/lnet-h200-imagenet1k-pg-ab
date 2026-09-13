import importlib.util
from pathlib import Path
import hashlib

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1] / 'scripts/probe_k64_recovery.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_missing_snapshot_does_not_create_files(tmp_path):
    root = tmp_path / 'missing'
    assert not probe.inspect(root)['checkpoint_found']
    assert not root.exists()


def test_existing_snapshot_is_only_hashed(tmp_path):
    p = tmp_path / 'checkpoint.pt'
    p.write_bytes(b'fixture-not-a-training-checkpoint')
    before = p.read_bytes()
    result = probe.inspect(tmp_path)
    assert result['checkpoint_found']
    assert result['sha256_observed'] == hashlib.sha256(before).hexdigest()
    assert result['checkpoint_epoch_verified'] is False
    assert p.read_bytes() == before

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_h200_dense_data as prep


def test_download_only_and_reuse(tmp_path):
    calls = []
    def extract(archive, stage):
        calls.append(archive.name)
        if archive.name.startswith('ADE'):
            (stage / 'ADEChallengeData2016').mkdir()
    with patch.object(prep.assets, 'verify_coco', return_value={'ready': True}), \
         patch.object(prep.assets, 'verify_ade', return_value={'ready': True}), \
         patch.object(prep.assets, 'download_archive', return_value={'ok': True}), \
         patch.object(prep.assets, 'safe_extract_zip', side_effect=extract):
        report = prep.prepare(tmp_path)
        assert report['ready'] and not report['training_started']
        assert len(calls) == 4
        prep.prepare(tmp_path)
        assert len(calls) == 4


def test_incomplete_preserved(tmp_path):
    target = tmp_path / 'datasets/coco'
    target.mkdir(parents=True)
    marker = target / 'user-file'
    marker.touch()
    with patch.object(prep.assets, 'verify_coco', return_value={'ready': False}):
        import pytest
        with pytest.raises(RuntimeError, match='untouched'):
            prep.prepare(tmp_path)
    assert marker.exists()


def test_coco_only(tmp_path):
    calls = []
    with patch.object(prep.assets, 'verify_coco', return_value={'ready': True}), \
         patch.object(prep.assets, 'verify_ade', side_effect=AssertionError('ADE must not be checked')), \
         patch.object(prep.assets, 'download_archive', return_value={'ok': True}), \
         patch.object(prep.assets, 'safe_extract_zip', side_effect=lambda archive, stage: calls.append(archive.name)):
        report = prep.prepare(tmp_path, only_coco=True)
    assert report['ready']
    assert set(report['datasets']) == {'coco'}
    assert len(calls) == 3

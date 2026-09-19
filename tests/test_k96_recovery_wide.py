import sys
from pathlib import Path
import time

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import probe_k96_recovery_wide as probe


def test_owner_scope(tmp_path,capsys):
    base=tmp_path/'storage'
    target=base/'outputs'/'daehwa00_650'
    target.mkdir(parents=True)
    other=base/'outputs'/'someone_else'
    other.mkdir()
    (other/'daehwa00_rogue').mkdir()
    roots=probe.discover([str(base)],time.monotonic()+10)
    assert roots==[target]
    assert 'someone_else' not in capsys.readouterr().out


def test_symlinks_not_followed(tmp_path):
    actual=tmp_path/'hidden'
    actual.mkdir()
    (tmp_path/'daehwa00_link').symlink_to(actual,target_is_directory=True)
    assert probe.discover([str(tmp_path)],time.monotonic()+10)==[]


def test_expired_discovery(tmp_path):
    assert probe.discover([str(tmp_path)],time.monotonic()-1)==[]

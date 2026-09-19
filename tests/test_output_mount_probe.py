import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('output_probe',Path(__file__).parents[1]/'scripts/probe_output_mount.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_mount_root_identifies_subdirectory():
    text='10 1 259:4 /jobs/800 /app/output rw - ext4 /dev/nvme1n1 rw\n11 1 0:2 / /secrets rw - tmpfs tmpfs rw'
    rows=probe.mounts(text)
    assert len(rows)==1
    assert rows[0]['mount_root']=='/jobs/800'
    assert rows[0]['source']=='/dev/nvme1n1'


def test_top_level_only_and_symlink(tmp_path,capsys):
    (tmp_path/'folder').mkdir()
    (tmp_path/'folder'/'hidden-content').write_text('never read')
    (tmp_path/'link').symlink_to('folder',target_is_directory=True)
    result=probe.listing(tmp_path)
    output=capsys.readouterr().out
    assert result=={'entries_reported':2,'truncated':False}
    assert 'hidden-content' not in output and 'never read' not in output
    assert '"link_target": "folder"' in output


def test_listing_limit(tmp_path):
    (tmp_path/'a').touch()
    (tmp_path/'b').touch()
    assert probe.listing(tmp_path,limit=1)=={'entries_reported':1,'truncated':True}

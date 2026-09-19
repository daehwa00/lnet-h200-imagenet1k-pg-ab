"""Pinned official CUB download and safe isolated extraction; no GPU work."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

URL='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
MD5='97eceeb196236b17998738112f37df78'


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe_members(archive):
    members=archive.getmembers()
    for entry in members:
        name=PurePosixPath(entry.name)
        if name.is_absolute() or '..' in name.parts or not (entry.isfile() or entry.isdir()):
            raise ValueError('Unsafe tar entry: '+entry.name)
        if name.parts[0]!='CUB_200_2011':raise ValueError('Unexpected archive root')
    return members


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True,type=Path); args=p.parse_args()
    root=args.root; root.mkdir(parents=True,exist_ok=True)
    from cub200_data import manifest
    target=root/'CUB_200_2011'
    archive=root/'CUB_200_2011.tgz'
    if target.exists():
        info=manifest(target)
        print('CUB_DATA_READY='+json.dumps({k:v for k,v in info.items() if k!='records'}),flush=True)
        return
    if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('Need at least 5GiB free for data preparation')
    if not archive.exists():
        partial=root/'CUB_200_2011.tgz.part'
        subprocess.run(['curl','--fail','--location','--retry','3','--connect-timeout','30',
            '--max-time','1800','--continue-at','-','--output',str(partial),URL],check=True)
        if digest(partial,'md5')!=MD5:raise RuntimeError('Official CUB archive MD5 mismatch')
        partial.replace(archive)
    if digest(archive,'md5')!=MD5:raise RuntimeError('Existing archive MD5 mismatch; retained untouched')
    stage=Path(tempfile.mkdtemp(prefix='.cub-stage-',dir=root))
    with tarfile.open(archive,'r:gz') as tar:
        members=safe_members(tar)
        tar.extractall(stage,members=members,filter='data')
    info=manifest(stage/'CUB_200_2011')
    os.rename(stage/'CUB_200_2011',target)
    stage.rmdir()
    (root/'data-ready.json').write_text(json.dumps({k:v for k,v in info.items() if k!='records'}))
    print('CUB_DATA_READY='+json.dumps({'root':str(target),'train':info['train'],'test':info['test'],
        'split_sha256':info['split_sha256'],'archive_sha256':digest(archive)}),flush=True)


if __name__=='__main__':main()

"""Two pinned source trees and one pinned native wheel; never pretrained weights."""
import argparse
import hashlib
from pathlib import Path
import subprocess

SOURCES={
 'parc_net':('https://github.com/hkzhang91/ParC-Net.git','f6780d0ad795835df2181b8fd4737276deb6e211'),
 'tinyvim':('https://github.com/xwmaxwma/TinyViM.git','cff64524117b1439becf89827a3428a288677b95')}
WHEEL='mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl'
WHEEL_URL='https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab/releases/download/h200-selective-scan-torch291-cu128-v1/'+WHEEL
WHEEL_SHA='7201849146fb3b517e1a89741c4042596652dea24f44a94e4a83e6246353f49e'


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True,type=Path); args=p.parse_args()
    root=args.root; root.mkdir(parents=True,exist_ok=True)
    for name,(url,commit) in SOURCES.items():
        path=root/name
        if not path.exists():
            subprocess.run(['git','clone','--no-checkout','--filter=blob:none','--sparse',url,str(path)],check=True)
            directories=['model'] if name=='tinyvim' else ['cvnets','utils','optim','options','loss_fn','metrics','data','engine','common']
            subprocess.run(['git','-C',str(path),'sparse-checkout','set',*directories],check=True)
            subprocess.run(['git','-C',str(path),'checkout','--detach',commit],check=True)
        if git(path,'rev-parse','HEAD')!=commit or git(path,'status','--porcelain','--untracked-files=no'):
            raise RuntimeError('Wrong/modified external source; not overwriting '+str(path))
    wheel=root/WHEEL
    if not wheel.exists():
        part=root/(WHEEL+'.part')
        subprocess.run(['curl','--fail','--location','--retry','3','--connect-timeout','30',
            '--max-time','600','--continue-at','-','--output',str(part),WHEEL_URL],check=True)
        with part.open('rb') as stream: sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha!=WHEEL_SHA:raise RuntimeError('Native wheel checksum mismatch')
        part.replace(wheel)
    with wheel.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if sha!=WHEEL_SHA:raise RuntimeError('Existing wheel checksum mismatch')
    print('CUB_SOURCES_READY',flush=True)


if __name__=='__main__':main()

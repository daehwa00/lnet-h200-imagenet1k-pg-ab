"""Fetch verified official cp313/torch2.9 CUDA wheels; use only needed native code."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import urllib.request
import urllib.parse
import zipfile

WHEELS={
 'mamba':('https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.9cxx11abiTRUE-cp313-cp313-linux_x86_64.whl',
          '7201849146fb3b517e1a89741c4042596652dea24f44a94e4a83e6246353f49e'),
 'causal':('https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.9cxx11abiTRUE-cp313-cp313-linux_x86_64.whl',
           'a147b0d9eabede544c9b9a79ae1257d3759d82c62c887540ddef2f406837fa39'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def fetch_verified(url,sha,path):
    if path.exists() and digest(path)==sha:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.download')
    with urllib.request.urlopen(url,timeout=120) as response, tmp.open('wb') as stream:
        while data:=response.read(1024*1024): stream.write(data)
    if digest(tmp)!=sha:
        raise RuntimeError('official Vim dependency wheel digest mismatch')
    tmp.replace(path)


def extract_scan(wheel,native):
    with zipfile.ZipFile(wheel) as archive:
        names=[n for n in archive.namelist() if n.startswith('selective_scan_cuda.') and n.endswith('.so') and '/' not in n]
        if len(names)!=1:
            raise RuntimeError('wheel does not contain exactly one root selective_scan extension')
        native.mkdir(parents=True,exist_ok=True)
        target=native/names[0]
        temporary=target.with_suffix('.tmp')
        with archive.open(names[0]) as source, temporary.open('wb') as out:
            while data:=source.read(1024*1024): out.write(data)
        temporary.replace(target)
        return target


def prepare_cuda_link(native):
    # Some managed images expose libcuda.so.1 but omit the linker-name symlink.
    # Keep this fix inside this run's native directory, never in system folders.
    listing=subprocess.check_output([shutil.which('ldconfig') or '/sbin/ldconfig','-p'],text=True)
    candidates=[Path(line.split('=>',1)[1].strip()).resolve()
                for line in listing.splitlines() if 'libcuda.so.1 ' in line and '=>' in line]
    candidates.extend((Path(d)/'libcuda.so.1').resolve()
                      for d in os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep) if d)
    target=next((p for p in candidates if p.is_file()),None)
    if target is None:
        raise RuntimeError('CUDA driver library libcuda.so.1 is unavailable')
    link=native/'libcuda.so'
    if link.is_symlink():
        if link.resolve()==target: return
        link.unlink()
    elif link.exists():
        raise RuntimeError('refusing to replace a regular libcuda.so file')
    link.symlink_to(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--native-root',type=Path,required=True)
    uv=parser.add_mutually_exclusive_group(required=True)
    uv.add_argument('--uv',type=Path)
    uv.add_argument('--uv-bootstrap',type=Path)
    parser.add_argument('--python',type=Path,required=True)
    a=parser.parse_args()
    wheels={}
    for key,(url,sha) in WHEELS.items():
        path=a.cache_root/urllib.parse.unquote(url.rsplit('/',1)[-1])
        fetch_verified(url,sha,path)
        wheels[key]=path
    # Official Vim supplies its own Python Mamba. Installing modern Mamba's
    # metadata would spuriously require unrelated tilelang/tvm/quack packages.
    extension=extract_scan(wheels['mamba'],a.native_root)
    prepare_cuda_link(a.native_root)
    command=[str(a.uv)] if a.uv else [str(a.python),'-m','uv']
    env=None if a.uv else dict(os.environ,PYTHONPATH=str(a.uv_bootstrap))
    subprocess.run(command+['pip','install','--python',str(a.python),'--no-deps',str(wheels['causal'])],env=env,check=True)
    print('VIM_NATIVE_SCAN_SHA256='+digest(extension),flush=True)


if __name__=='__main__': main()

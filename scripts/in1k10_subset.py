"""Fixed official SimCLR10% membership, never resampled per training seed."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request

REVISION='be133b38fb21abb77403391dc6db3412edfe92f2'
URL=f'https://raw.githubusercontent.com/google-research/simclr/{REVISION}/imagenet_subsets/10percent.txt'
GIT_BLOB='1ca337e2a25273c022a7a955893ae6c2a6cb856c'
COUNT=128116
SHA256='6d09de11e7bdaf5b1f3b1f249b6183695f97310cdd20f0c03e7235b6b9392091'


def parse(payload):
    blob=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    if blob!=GIT_BLOB or hashlib.sha256(payload).hexdigest()!=SHA256:raise ValueError('Official subset digest mismatch')
    names=payload.decode('ascii').splitlines()
    if len(names)!=COUNT or len(set(names))!=COUNT:raise ValueError('Subset count/uniqueness mismatch')
    if any(not re.fullmatch(r'n\d{8}_\d+\.JPEG',name) for name in names):raise ValueError('Unexpected ImageNet filename')
    counts=Counter(n.split('_')[0] for n in names)
    if len(counts)!=1000:raise ValueError('Subset must contain all1000 classes')
    return names,counts


def download(target):
    if target.exists():payload=target.read_bytes()
    else:
        with urllib.request.urlopen(URL,timeout=30) as response:payload=response.read(4*1024**2)
    names,counts=parse(payload)
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        part=target.with_suffix('.part');part.write_bytes(payload);part.replace(target)
    return {'url':URL,'revision':REVISION,'git_blob':GIT_BLOB,
            'sha256':hashlib.sha256(payload).hexdigest(),'images':len(names),'classes':len(counts),
            'min_per_class':min(counts.values()),'max_per_class':max(counts.values())}


def prepare(data_root,root):
    path=root/'simclr-10percent.txt';info=download(path)
    names,counts=parse(path.read_bytes())
    classes=sorted(p.name for p in (data_root/'train').iterdir() if p.is_dir())
    val_classes=sorted(p.name for p in (data_root/'val').iterdir() if p.is_dir())
    if classes!=val_classes or len(classes)!=1000 or set(classes)!=set(counts):
        raise ValueError('Main ImageNet class mapping mismatch')
    missing=[n for n in names if not (data_root/'train'/n.split('_')[0]/n).is_file()]
    if missing:raise FileNotFoundError(f'{len(missing)} official subset images missing; first={missing[:3]}')
    val_files=sorted(str(p.relative_to(data_root/'val')) for p in (data_root/'val').glob('*/*') if p.is_file())
    if len(val_files)!=50000:raise ValueError(f'Validation must remain50000 images, found{len(val_files)}')
    info.update(class_to_idx={name:i for i,name in enumerate(classes)},
        validation_count=len(val_files),validation_names_sha256=hashlib.sha256('\n'.join(val_files).encode()).hexdigest(),
        class_counts=dict(sorted(counts.items())),training_seeds=[501,509,521],subset_resampled=False)
    target=root/'subset-manifest.json'
    text=json.dumps(info,sort_keys=True,indent=2)+'\n'
    if target.exists() and target.read_text()!=text:raise RuntimeError('Refusing to change existing fixed subset manifest')
    if not target.exists():target.write_text(text)
    print(json.dumps({k:info[k] for k in ('images','classes','sha256','validation_count','min_per_class','max_per_class')}),flush=True)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data-root',type=Path)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    if args.data_root:prepare(args.data_root,args.root)
    else:print(json.dumps(download(args.root/'simclr-10percent.txt')))

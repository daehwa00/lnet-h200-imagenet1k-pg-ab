"""Clone only the two pinned original baseline repositories, without republishing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    specs=json.loads((Path(__file__).parents[1]/'h200/in1k10_sources.json').read_text())
    args.root.mkdir(parents=True,exist_ok=True);report={}
    for name,s in specs.items():
        target=args.root/name
        if not target.exists():
            subprocess.run(['git','clone','--no-checkout',s['repository'],str(target)],check=True)
            subprocess.run(['git','-C',str(target),'checkout','--detach',s['commit']],check=True)
        head=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(target),'status','--porcelain','--untracked-files=no'],text=True).strip()
        if head!=s['commit'] or dirty:raise RuntimeError('Pinned external source identity mismatch')
        report[name]={'repository':s['repository'],'commit':head}
    (args.root/'source-identity.json').write_text(json.dumps(report,sort_keys=True,indent=2))

if __name__=='__main__':main()

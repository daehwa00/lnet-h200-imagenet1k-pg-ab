"""Reuse the measured common profile; do not retune a partially completed panel."""
import argparse
from pathlib import Path
from in1k10_completed import FIXED_PROFILE,load_completed
from in1k10_probe import resource_budget
from in1k10_transport import atomic

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    load_completed()
    budget=resource_budget(512)
    if budget['max_workers']<8:raise RuntimeError('Current container cannot safely retain the verified8-worker profile')
    atomic(args.root/'input-profile.json',dict(FIXED_PROFILE,source='measured-v3-profile',resource_check=budget))
    print('IN10_PROFILE_FIXED batch=512 workers=8 ipc=memfd final_eval_only=true',flush=True)

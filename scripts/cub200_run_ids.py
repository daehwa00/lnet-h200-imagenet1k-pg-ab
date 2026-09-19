"""Deterministic relay allowlist; contains identifiers only, never API keys."""
import argparse
import hashlib
import json
from pathlib import Path


def run_id(campaign,name,attempt='0'):
    if not str(attempt).isdigit():raise ValueError('Invalid telemetry attempt')
    return hashlib.sha256((campaign+':'+name+':'+str(attempt)).encode()).hexdigest()[:16]


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='0');args=p.parse_args()
    config=json.loads((Path(__file__).parents[1]/'h200/cub200/campaign.json').read_text())
    names=['connectivity','preflight']+[f'{m}-seed{s}' for s in config['seeds'] for m in config['models']]
    print(json.dumps({'entity':config['wandb_entity'],'project':config['wandb_project'],
        'group':config['campaign_id'],'attempt':args.attempt,
        'runs':{name:run_id(config['campaign_id'],name,args.attempt) for name in names}},indent=2))


if __name__=='__main__':main()

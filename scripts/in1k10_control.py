"""Private owner control: no GitHub edits, credentials never on command line."""
import argparse
import json
from pathlib import Path
from in1k10_transport import API,read

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--secrets',type=Path,required=True)
    p.add_argument('action',choices=('status','stop','force','release','arm'))
    args=p.parse_args();api=API(read(args.secrets)['IN10_OWNER_TOKEN'])
    value=api.call('/snapshot') if args.action=='status' else api.call('/command',{'action':args.action})
    print(json.dumps(value,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

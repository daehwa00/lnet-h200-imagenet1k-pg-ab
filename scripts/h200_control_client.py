"""Owner-only control/log client. Secrets never belong in submitted GitHub commands."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',required=True)
    p.add_argument('--secrets',required=True,type=Path)
    p.add_argument('--action',choices=('status','logs','follow','stop'),required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    token=json.loads(args.secrets.read_text())['OWNER_TOKEN']
    def request(route,post=False):
        req=urllib.request.Request(args.url.rstrip('/')+route,data=b'{}' if post else None,
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','User-Agent':'lnet-h200-control/1.0'})
        with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)
    if args.action in ('status','stop'):
        print(json.dumps(request('/stop' if args.action=='stop' else '/control',args.action=='stop')))
        return
    cursor=0
    cursorfile=args.output.with_suffix('.cursor') if args.output else None
    if cursorfile and cursorfile.exists():cursor=int(cursorfile.read_text())
    while True:
        try:
            rows=request('/logs?after='+str(cursor))
            for row in rows:
                text=json.dumps(row,ensure_ascii=False)
                if args.output:
                    with args.output.open('a') as stream:stream.write(text+'\n')
                else:print(text,flush=True)
                cursor=row['id']
                if cursorfile:cursorfile.write_text(str(cursor))
        except Exception as error:
            print('control_receiver_error='+type(error).__name__,flush=True)
            if args.action!='follow':raise
        if args.action!='follow':return
        time.sleep(5)


if __name__=='__main__':main()

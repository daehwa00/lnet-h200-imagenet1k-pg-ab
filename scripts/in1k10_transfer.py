"""Killable transfer subprocess; credentials enter only through stdin."""
import json
from pathlib import Path
import sys
from in1k10_transport import API,atomic,upload,download

def main():
    request=json.load(sys.stdin);api=API(request['token'],request['session'])
    try:
        if request['kind']=='upload':result=upload(api,request['job'],request['meta'])
        else:
            result=api.call(f'/artifact/{request["job"]}/latest')
            if result:download(api,request['job'],result,Path(request['target']))
        atomic(Path(request['result']),{'ok':True,'value':result})
    except Exception as error:
        atomic(Path(request['result']),{'ok':False,'error':type(error).__name__+': '+str(error)[:300]})
        raise SystemExit(1)

if __name__=='__main__':main()

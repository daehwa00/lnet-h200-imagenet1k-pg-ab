"""Stdlib-only authenticated telemetry/checkpoint transport, without public keys."""
import concurrent.futures
import base64
import hashlib
import http.client
import json
import math
import re
from email.utils import parsedate_to_datetime
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

BASE='https://lnet-h200-baseline-relay-v1.gpupulse-monitor.workers.dev/in10'
CHUNK=32*1024


def _redact(value, secrets=()):
    text=str(value)
    for secret in secrets:
        if secret:text=text.replace(secret,'[REDACTED]')
    text=re.sub(r'(?i)\bBearer\s+[^\s,;"\'<>]+','Bearer [REDACTED]',text)
    text=re.sub(r'(?i)(["\']?(?:authorization|token|access_token|api[_-]?key|password|secret|x-session-id)["\']?\s*[:=]\s*)(?:"[^"]*"|\'[^\']*\'|[^\s,;<>]+)',r'\1[REDACTED]',text)
    text=re.sub(r'\b[A-Za-z0-9_+/=-]{32,}\b','[REDACTED]',text)
    # Error URLs can carry credentials in their query string.
    text=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[REDACTED]',text)
    return text[:400]


def _retry_after(error):
    headers=getattr(error,'headers',None)
    value=headers.get('Retry-After') if headers else None
    if not value:return None
    try:
        seconds=float(value)
    except (ValueError,TypeError):
        try:seconds=parsedate_to_datetime(value).timestamp()-time.time()
        except (ValueError,TypeError,OverflowError):return None
    return max(0.,seconds) if math.isfinite(seconds) else None


def _cache_error(error,secrets=()):
    """Read a bounded HTTP error excerpt once; never retain unredacted payloads."""
    if not hasattr(error,'_in10_body_excerpt'):
        body=''
        if isinstance(error,urllib.error.HTTPError):
            try:body=error.read(4096).decode('utf-8',errors='replace')
            except Exception:body='[error body unavailable]'
        error._in10_body_excerpt=_redact(body,secrets)
    else:
        error._in10_body_excerpt=_redact(error._in10_body_excerpt,secrets)
    error._in10_message=_redact(getattr(error,'_in10_message',str(error)),secrets)
    return error


def error_details(error):
    _cache_error(error)
    return {'type':type(error).__name__,'http_status':getattr(error,'code',None),
            'retry_after_seconds':_retry_after(error),'message':error._in10_message,
            'body_excerpt':error._in10_body_excerpt}


def retry_delay(error,failure_count,base=15,cap=300):
    """Bound exponential backoff, but never shorten a server Retry-After."""
    delay=min(cap,base*2**min(30,max(0,failure_count-1)))
    return max(delay,_retry_after(error) or 0.)


class API:
    def __init__(self,token,session='',base=BASE):
        self.token,self.session,self.base=token,session,base
        self._pace_lock=threading.Lock();self._next_chunk=0.
    def call(self,path,value=None,method=None,raw=False,timeout=15,attempts=4):
        if not isinstance(attempts,int) or attempts<1:raise ValueError('attempts must be a positive integer')
        binary=isinstance(value,bytes)
        data=json.dumps({'base64':base64.b64encode(value).decode()}).encode() if binary else None if value is None else json.dumps(value).encode()
        headers={'Authorization':'Bearer '+self.token,'X-Session-ID':self.session,
                 'User-Agent':'lnet-in1k10/1.0','Content-Type':'application/json','Accept':'application/json'}
        request=urllib.request.Request(self.base+path,data=data,headers=headers,method=method)
        for attempt in range(attempts):
            if binary or raw:
                with self._pace_lock:
                    time.sleep(max(0,self._next_chunk-time.monotonic()))
                    self._next_chunk=time.monotonic()+.125
            try:
                with urllib.request.urlopen(request,timeout=timeout) as response:
                    result=json.load(response)
                    return base64.b64decode(result['base64'],validate=True) if raw else result
            except urllib.error.HTTPError as error:
                _cache_error(error,(self.token,self.session))
                if not (error.code==429 or 500<=error.code<=599) or attempt==attempts-1:raise
                time.sleep(retry_delay(error,attempt+1,base=15 if error.code==429 else 1))
            except (urllib.error.URLError,TimeoutError,ConnectionError,http.client.HTTPException) as error:
                _cache_error(error,(self.token,self.session))
                if attempt==attempts-1:raise
                time.sleep(retry_delay(error,attempt+1,base=1))


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,sort_keys=True));temp.replace(path)


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def upload(api,job,meta):
    path=Path(meta['path'])
    # Open once: atomic checkpoint replacement cannot change this inode midway.
    with path.open('rb') as stream:
        import os
        info=os.fstat(stream.fileno())
        if (info.st_ino,info.st_size,info.st_mtime_ns)!=(meta['inode'],meta['bytes'],meta['mtime_ns']):return None
        digest=hashlib.sha256()
        for data in iter(lambda:stream.read(CHUNK),b''):digest.update(data)
        sha=digest.hexdigest();stream.seek(0)
        chunks=(info.st_size+CHUNK-1)//CHUNK
        record={'sha':sha,'bytes':info.st_size,'chunks':chunks,'epoch':meta['epoch'],'workers':meta.get('workers',0)}
        existing=api.call(f'/artifact/{job}/begin',record)
        if not existing['complete']:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pending=[]
                for part in range(chunks):
                    data=stream.read(CHUNK)
                    pending.append(pool.submit(api.call,f'/artifact/{job}/{sha}/{part}',data,'PUT',False,60))
                    if len(pending)==4:
                        for f in pending:f.result()
                        pending=[]
                for f in pending:f.result()
            api.call(f'/artifact/{job}/{sha}/finish',{})
        return record


def download(api,job,meta,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_suffix('.part');digest=hashlib.sha256();size=0
    with part.open('wb') as stream,concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for start in range(0,meta['chunks'],4):
            futures=[pool.submit(api.call,f'/artifact/{job}/{meta["sha"]}/{i}',None,None,True,60)
                     for i in range(start,min(start+4,meta['chunks']))]
            for f in futures:
                data=f.result();stream.write(data);digest.update(data);size+=len(data)
    if size!=meta['bytes'] or digest.hexdigest()!=meta['sha']:raise ValueError('Private checkpoint SHA256/size mismatch')
    part.replace(target)
    return meta['sha']

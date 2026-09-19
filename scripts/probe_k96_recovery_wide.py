"""Search accessible owner storage only; never mount devices or traverse other users."""
import collections
import os
from pathlib import Path
import subprocess
import sys
import time

from probe_k96_storage import emit, scan
from probe_output_mount import mounts

ANCHORS = ('/outputs', '/output', '/app/outputs', '/app/output', '/app/daehwa00',
           '/mnt', '/data', '/storage', '/workspace', '/volumes', '/home')
CONTAINERS = {'outputs', 'output', 'storage', 'persistent', 'volumes', 'data',
              'mnt', 'workspace', 'users', 'home'}


def owner(name):
    n=name.lower()
    return n=='daehwa00' or n.startswith(('daehwa00_', 'daehwa00-'))


def container(name):
    n=name.lower()
    return n in CONTAINERS or n.startswith(('nvme', 'disk', 'ssd', 'hdd', 'pvc-'))


def discover(anchors, deadline):
    found=[]
    queue=collections.deque((Path(p),0) for p in anchors)
    seen=set()
    examined=0
    while queue and time.monotonic()<deadline and examined<10000:
        path,depth=queue.popleft()
        if str(path) in seen:continue
        seen.add(str(path))
        try:
            if path.is_symlink():
                emit('RECOVERY_LINK_SKIPPED',{'path':str(path)})
                continue
            info=path.stat()
            if not path.is_dir():continue
            emit('RECOVERY_VISIBLE_CONTAINER',{'path':str(path),'device':info.st_dev})
            if owner(path.name):
                found.append(path)
                continue
            # This is the assigned job output, regardless of its visible basename.
            if str(path)=='/app/output':found.append(path)
            with os.scandir(path) as entries:
                for entry in entries:
                    examined+=1
                    if examined>=10000 or time.monotonic()>=deadline:break
                    if entry.is_symlink():
                        if owner(entry.name):emit('RECOVERY_LINK_SKIPPED',{'path':entry.path})
                        continue
                    if not entry.is_dir(follow_symlinks=False):continue
                    if owner(entry.name):
                        found.append(Path(entry.path))
                        emit('RECOVERY_OWNER_ROOT',{'path':entry.path})
                    elif depth<3 and container(entry.name):
                        queue.append((Path(entry.path),depth+1))
        except OSError as exc:
            emit('RECOVERY_CONTAINER_UNAVAILABLE',{'path':str(path),'error':type(exc).__name__})
    unique=[]
    for path in sorted(set(found), key=lambda p:len(p.parts)):
        if not any(path==old or old in path.parents for old in unique):unique.append(path)
    emit('RECOVERY_DISCOVERY',{'owner_roots':[str(p) for p in unique],
        'entries_examined':examined,'limit_reached':bool(queue) or examined>=10000})
    return unique


def probe():
    emit('RECOVERY_START',{'read_only':True,'training_started':False,
        'scope':'Accessible daehwa00 storage only; no privilege/mount operations',
        'metadata_only':True,'maximum_seconds':150})
    started=time.monotonic()
    try:emit('RECOVERY_OUTPUT_MOUNTS',mounts(Path('/proc/self/mountinfo').read_text()))
    except OSError as exc:emit('RECOVERY_MOUNT_ERROR',{'error':type(exc).__name__})
    roots=discover(ANCHORS,started+20)
    results=[]
    for root in roots:
        remaining=started+140-time.monotonic()
        if remaining<=0:break
        results.append(scan(root,seconds=min(20,remaining),max_entries=50000,max_depth=18))
    emit('RECOVERY_DONE',{'roots_discovered':len(roots),'roots_searched':len(results),
        'matching_files':sum(r['matching_files'] for r in results),
        'search_limits_hit':len(results)<len(roots) or any(r['reason']=='search_limit_reached' for r in results),
        'note':'Only paths visible inside this container were searched. No hits cannot establish deletion on the host. Matching files are not yet validated ImageNet-1K checkpoints.'})


if __name__=='__main__':
    if sys.argv[1:]==['--child']:probe()
    elif len(sys.argv)==1:
        try:
            result=subprocess.run([sys.executable,'-B','-u',__file__,'--child'],timeout=150)
            sys.exit(result.returncode)
        except subprocess.TimeoutExpired:
            emit('RECOVERY_TIMEOUT',{'seconds':150,'search_complete':False})
            sys.exit(2)
    else:raise SystemExit('No arguments required')

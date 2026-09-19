"""Read-only /app/output top-level metadata and its covering mount; no recursion."""
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


def emit(kind, value):
    print(kind + '=' + json.dumps(value, ensure_ascii=False), flush=True)


def unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), value)


def mounts(text, target='/app/output'):
    rows=[]
    for line in text.splitlines():
        try:
            left,right=line.split(' - ',1)
            a,b=left.split(),right.split()
            point=unescape(a[4])
            if point=='/' or target==point or target.startswith(point.rstrip('/')+'/') or point.startswith(target+'/'):
                rows.append({'mount_id':a[0],'parent_mount_id':a[1],'device':a[2],
                    'mount_root':unescape(a[3]),'mount_point':point,
                    'filesystem':b[0],'source':unescape(b[1]),
                    'read_only':'ro' in a[5].split(',')})
        except (ValueError,IndexError):
            continue
    return rows


def listing(root, limit=100):
    count=0
    with os.scandir(root) as entries:
        for entry in entries:
            if count>=limit:
                return {'entries_reported':count,'truncated':True}
            count+=1
            try:
                info=entry.stat(follow_symlinks=False)
                kind=('symlink' if stat.S_ISLNK(info.st_mode) else
                      'directory' if stat.S_ISDIR(info.st_mode) else
                      'file' if stat.S_ISREG(info.st_mode) else 'other')
                row={'name':entry.name,'type':kind,'bytes':info.st_size,
                     'uid':info.st_uid,'gid':info.st_gid,'mode':stat.filemode(info.st_mode),
                     'device':info.st_dev,'inode':info.st_ino,'mtime_unix':info.st_mtime}
                if kind=='symlink':row['link_target']=os.readlink(entry.path)
                emit('OUTPUT_ENTRY',row)
            except OSError as error:
                emit('OUTPUT_ENTRY_ERROR',{'name':entry.name,'error':type(error).__name__})
    return {'entries_reported':count,'truncated':False}


def probe():
    root=Path('/app/output')
    emit('OUTPUT_PROBE_START',{'read_only':True,'recursive':False,'file_contents_read':False,
        'training_started':False,'hostname':os.uname().nodename,'uid':os.getuid()})
    try:
        info=root.lstat()
        emit('OUTPUT_ROOT',{'path':str(root),'resolved_path':str(root.resolve()),
            'symlink':root.is_symlink(),'device':info.st_dev,'inode':info.st_ino,
            'uid':info.st_uid,'gid':info.st_gid,'mode':stat.filemode(info.st_mode)})
        v=os.statvfs(root)
        emit('OUTPUT_FILESYSTEM',{'total_bytes':v.f_blocks*v.f_frsize,
             'available_bytes':v.f_bavail*v.f_frsize,
             'note':'Filesystem-wide space, not the size of this visible directory.'})
        emit('OUTPUT_MOUNTS',mounts(Path('/proc/self/mountinfo').read_text()))
        emit('OUTPUT_LIST_SUMMARY',listing(root))
    except OSError as error:
        emit('OUTPUT_PROBE_ERROR',{'error':type(error).__name__,'message':str(error)[:500]})
        return 1
    emit('OUTPUT_PROBE_DONE',{'read_only':True,'files_changed':False})
    return 0


if __name__=='__main__':
    if sys.argv[1:]==['--child']:
        sys.exit(probe())
    elif len(sys.argv)==1:
        try:
            result=subprocess.run([sys.executable,'-B','-u',__file__,'--child'],timeout=30)
            sys.exit(result.returncode)
        except subprocess.TimeoutExpired:
            emit('OUTPUT_PROBE_TIMEOUT',{'seconds':30,'inspection_complete':False})
            sys.exit(2)
    else:
        raise SystemExit('No arguments required')

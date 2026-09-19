"""Bounded, read-only storage visibility search. No torch, downloads or training."""
import collections
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time

ROOTS = ('/app/output', '/app/daehwa00')
SKIP = {'datasets', 'data', 'train2017', 'val2017', 'annotations', 'archives',
        'code', 'src', 'scripts', 'tests', 'node_modules', '.git', '__pycache__',
        'site-packages', 'wandb', 'cache', '.cache', 'uv-python', 'uv-cache'}


def emit(kind, value):
    print(kind + '=' + json.dumps(value, ensure_ascii=False), flush=True)


def owned_name(name):
    n = name.lower()
    return n == 'daehwa00' or n.startswith(('lnet-', 'lnet_', 'run-data', 'dense-transfer')) or 'k96' in n


def mount_summary(text):
    rows = []
    for line in text.splitlines():
        try:
            left, right = line.split(' - ', 1)
            fields, fs = left.split(), right.split()
            mount = fields[4].replace('\\040', ' ')
            if mount == '/' or mount == '/app' or mount.startswith(('/app/output', '/app/daehwa00')):
                # No mount options, source addresses, credentials or unrelated mounts.
                rows.append({'mount_id': fields[0], 'device': fields[2],
                             'mount_point': mount, 'filesystem': fs[0]})
        except (ValueError, IndexError):
            continue
    return rows


def scan(root, *, seconds=25, max_entries=30000, max_depth=12):
    start = time.monotonic()
    queue = collections.deque([(Path(root), 0)])
    count = found = errors = 0
    reason = 'exhausted_accessible_scope'
    while queue:
        directory, depth = queue.popleft()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    count += 1
                    if count > max_entries or time.monotonic()-start >= seconds or found >= 100:
                        reason = 'search_limit_reached'
                        queue.clear()
                        break
                    # Avoid traversing other users' top-level trees on shared output.
                    if depth == 0 and str(root) == '/app/output' and not owned_name(entry.name):
                        continue
                    path = Path(entry.path)
                    matched = 'k96' in str(path).lower()
                    if entry.is_symlink():
                        if matched or owned_name(entry.name):
                            emit('K96_SYMLINK_NOT_FOLLOWED', {'path': str(path)})
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if 'k96' in entry.name.lower():
                            emit('K96_DIRECTORY', {'path': str(path)})
                        n = entry.name.lower()
                        if depth < max_depth and n not in SKIP and not n.startswith(('environment', 'venv', 'uv-bootstrap')):
                            queue.append((path, depth+1))
                    elif matched and entry.is_file(follow_symlinks=False) and (
                        path.suffix.lower() in {'.pt', '.pth', '.ckpt'} or entry.name == 'result.json'
                    ):
                        info = entry.stat(follow_symlinks=False)
                        emit('K96_FILE', {'path': str(path), 'bytes': info.st_size,
                                         'mtime_unix': info.st_mtime, 'content_loaded': False})
                        found += 1
        except OSError as exc:
            errors += 1
            if errors <= 8:
                emit('K96_ACCESS_ERROR', {'path': str(directory), 'error': type(exc).__name__})
    result = {'root': str(root), 'entries_examined': count, 'matching_files': found,
              'access_errors': errors, 'reason': reason, 'depth_limit': max_depth,
              'elapsed_seconds': round(time.monotonic()-start, 2),
              'note': 'Pruned caches/data/code and unrelated user roots; symlinks not followed. Not an exhaustive server-wide search.'}
    emit('K96_SEARCH_SUMMARY', result)
    return result


def probe():
    emit('K96_STORAGE_START', {'read_only': True, 'training_started': False,
        'hostname': socket.gethostname(), 'uid': os.getuid(),
        'accuracy_re_evaluated': False})
    try: emit('K96_MOUNTS', mount_summary(Path('/proc/self/mountinfo').read_text()))
    except OSError as exc: emit('K96_MOUNT_ERROR', {'error': type(exc).__name__})
    for name in ('/app/output', '/app/output/daehwa00', '/app/daehwa00',
                 '/app/output/daehwa00/dense-transfer/datasets-ready.json'):
        try:
            s = os.stat(name)
            emit('K96_PATH', {'path': name, 'exists': True, 'device': s.st_dev,
                             'directory': stat.S_ISDIR(s.st_mode), 'readable': os.access(name, os.R_OK)})
        except OSError as exc:
            emit('K96_PATH', {'path': name, 'exists': None if isinstance(exc, PermissionError) else False,
                             'error': type(exc).__name__})
    for root in ROOTS:
        if Path(root).is_dir() and not Path(root).is_symlink(): scan(root)
    emit('K96_STORAGE_DONE', {'read_only': True, 'training_started': False,
        'note': 'Missing means not visible in this mount/search scope, not proof of deletion. Found weights still require epoch/classifier/accuracy verification.'})


if __name__ == '__main__':
    if sys.argv[1:] == ['--child']:
        probe()
    elif len(sys.argv) == 1:
        # Includes slow filesystem operations; preserve incremental stdout on timeout.
        try:
            p = subprocess.run([sys.executable, '-B', '-u', __file__, '--child'], timeout=70)
            sys.exit(p.returncode)
        except subprocess.TimeoutExpired:
            emit('K96_STORAGE_TIMEOUT', {'seconds': 70, 'search_complete': False})
            sys.exit(2)
    else:
        raise SystemExit('No arguments required')

"""Independent stdlib-only HTTP stop/log guard; no GitHub or torch dependency."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def request(base, token, route, value=None):
    data = None if value is None else json.dumps(value).encode()
    req = urllib.request.Request(base.rstrip('/') + route, data=data,
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json', 'User-Agent': 'lnet-h200-control/1.0'})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def supervise(args):
    token = os.environ['H200_AGENT_TOKEN']
    args.root.mkdir(parents=True, exist_ok=True)
    # Refuse to spend GPU time before the authenticated control path works.
    control = request(args.url, token, '/control')
    if control['stop']:
        raise RuntimeError('Campaign is stopped; use a new explicitly armed campaign')
    logpath = args.root / 'console.log'
    stopped = False
    def stop_signal(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_signal)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
    # The training subprocess has no need for transport credentials.
    env.pop('H200_AGENT_TOKEN', None)
    with logpath.open('ab', buffering=0) as output:
        child = subprocess.Popen(args.command, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        offset = 0
        last_contact = time.monotonic()
        deadline = None
        forced = False
        error = None
        try:
            while True:
                now = time.monotonic()
                try:
                    stopped |= bool(request(args.url, token, '/control')['stop'])
                    last_contact = now
                except Exception as exc:
                    error = type(exc).__name__
                if now - last_contact > args.disconnect_seconds:
                    stopped = True
                    error = 'control_connection_timeout'
                if stopped and deadline is None:
                    deadline = now + args.grace_seconds
                    try: os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                if deadline is not None and now >= deadline and child.poll() is None:
                    forced = True
                    os.killpg(child.pid, signal.SIGKILL)
                state = {'pid': child.pid, 'unix_time': time.time(), 'exit_code': child.poll(),
                         'stop_requested': stopped, 'forced': forced, 'transport_error': error}
                state['training'] = {}
                for path in args.root.glob('seed_*/status/progress.json'):
                    try: state['training'][path.parent.parent.name] = json.loads(path.read_text())
                    except (OSError, ValueError): pass
                for name in ('memory.current', 'memory.events', 'memory.max'):
                    try: state[name] = Path('/sys/fs/cgroup', name).read_text().strip()
                    except OSError: pass
                atomic(args.root / 'guard-status.json', state)
                try:
                    with logpath.open('rb') as stream:
                        stream.seek(offset)
                        chunk = stream.read(24000)
                    request(args.url, token, '/events', {'offset': offset, 'text': chunk.decode('utf-8', errors='replace'), 'status': state})
                    offset += len(chunk)
                    error = None
                    atomic(args.root / 'uploaded-offset.json', {'offset': offset})
                except Exception as exc:
                    error = type(exc).__name__
                if child.poll() is not None:
                    # Full local output is authoritative if the network is down.
                    if offset >= logpath.stat().st_size or error:
                        break
                time.sleep(args.poll_seconds)
        finally:
            # Reap descendants too if a bootstrap shell exited before its workers.
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            code = child.wait()
            atomic(args.root / 'exit.json', {'exit_code': code, 'forced': forced, 'stop_requested': stopped, 'unix_time': time.time()})
        return 130 if stopped else code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--poll-seconds', type=float, default=5)
    p.add_argument('--grace-seconds', type=float, default=120)
    p.add_argument('--disconnect-seconds', type=float, default=600)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.command[:1] == ['--']: args.command = args.command[1:]
    if not args.command or min(args.poll_seconds, args.grace_seconds, args.disconnect_seconds) <= 0:
        p.error('command and positive timeout values required')
    return supervise(args)


if __name__ == '__main__':
    sys.exit(main())

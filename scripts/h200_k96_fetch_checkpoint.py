"""Fetch the private K96 ImageNet checkpoint from the authenticated campaign relay."""
import argparse
import hashlib
import os
from pathlib import Path

from in1k10_transport import API, download


BASE = 'https://lnet-h200-baseline-relay-v1.gpupulse-monitor.workers.dev/k96coco'
JOB = 'k96coco-521-8k'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    token = os.environ['H200_AGENT_TOKEN']
    session = os.environ['H200_SESSION_ID']
    api = API(token, session=session, base=BASE)
    meta = api.call(f'/artifact/{JOB}/latest')
    if not meta.get('complete') or not meta.get('verified') or meta.get('sha') != args.sha256 or meta.get('epoch') != 100:
        raise RuntimeError('Expected verified K96 epoch100 checkpoint is unavailable')
    if args.output.exists():
        if sha256(args.output) != args.sha256:
            raise RuntimeError('Existing checkpoint has unexpected SHA256; refusing overwrite')
        print('K96_CHECKPOINT=reused_verified', flush=True)
        return
    download(api, JOB, meta, args.output)
    if sha256(args.output) != args.sha256:
        raise RuntimeError('Downloaded checkpoint has unexpected SHA256')
    print('K96_CHECKPOINT=downloaded_verified', flush=True)


if __name__ == '__main__':
    main()

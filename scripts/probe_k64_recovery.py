"""Read-only visibility probe for the exact K64 seed501 run; never train.

No root scan, checkpoint upload, deletion, PyTorch import, or GPU allocation.
Only the previously identified experiment's candidate mount layouts are read.
"""
import hashlib
import json
from pathlib import Path

STEM = 'lnet-h200-imagenet1k-baselines-v1-f4f734e8512d-53b7663aa027'
RUN = 'run-data992f76a6fb09/lnet-k64-p80x4-d2262-mig1-lee/lnet_k64_p80x4_d2262_mig1_lee_v1/seed_501'
CANDIDATES = [Path('/app/output/Lee-Wonwoo1') / STEM / RUN,
              Path('/app/output') / STEM / RUN,
              Path('/app/output') / RUN]


def inspect(root):
    checkpoint = root / 'checkpoint.pt'
    result = {'run_root': str(root), 'checkpoint_found': False}
    try:
        if not checkpoint.is_file():
            return result
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result.update(checkpoint_found=True, bytes=checkpoint.stat().st_size,
                      sha256_observed=digest.hexdigest(), checkpoint=str(checkpoint))
        result['contracts'] = [str(p) for p in (root / 'contracts').glob('*.json')]
        result['result_json_exists'] = (root / 'result.json').is_file()
        result['checkpoint_epoch_verified'] = False
    except OSError as error:
        result['error_type'] = type(error).__name__
    return result


if __name__ == '__main__':
    results = [inspect(root) for root in CANDIDATES]
    print(json.dumps({'read_only': True, 'training_started': False,
        'original_issue': 403, 'original_build': 713, 'seed': 501,
        'checkpoint_found': any(r['checkpoint_found'] for r in results),
        'note': 'Not found means not visible here; it does not prove deletion on the server.',
        'candidates': results}), flush=True)

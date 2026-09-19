"""Read-only exact-run K96 recovery probe. No training/downloads/GPU required."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

MODEL = 'lnet_k96_p128x4_d2262_optimized_v2'
CONTRACT = '94c9c944743ab6f05205d8bec740718df637bdeede6090bf1cbf1fe12e0d810d'
RELATIVE = ('lnet-h200-imagenet1k-baselines-v1-1b08ae8ae53d-f361b248aeb1/'
            'run-data992f76a6fb09/lnet-k96-p128x4-d2262-3seed/' + MODEL + '/seed_501')
ROOTS = [Path(prefix) / RELATIVE for prefix in
         ('/app/output/daehwa00', '/app/output', '/app/daehwa00')]


def accuracy(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
        return 100 * value
    return None


def checkpoint_metadata(path):
    import torch
    # Exact user-owned training file, not an arbitrary downloaded pickle.
    p = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(p, dict):
        raise ValueError('Checkpoint is not a dictionary')
    epoch = p.get('completed_epochs')
    history = p.get('history', [])
    last = next((x for x in reversed(history) if x.get('epoch') == epoch), {})
    state = p.get('model', {})
    heads = {k: list(v.shape) for k, v in state.items()
             if k.endswith('classifier.affine.linear.weight') and hasattr(v, 'shape')}
    return {'completed_epochs': epoch, 'global_step': p.get('global_step'),
            'contract_sha256': p.get('contract_sha256'),
            'parameters': p.get('parameters'), 'classifier_weight_shapes': heads,
            'saved_top1_percent': accuracy(last.get('validation', {}).get('accuracy')),
            'saved_top5_percent': accuracy(last.get('validation', {}).get('top5_accuracy')),
            'expected_identity_match': (epoch == 100 and p.get('contract_sha256') == CONTRACT
                and p.get('parameters') == 3253224 and bool(heads)
                and all(shape == [1000, 512] for shape in heads.values()))}


def inspect(root, timeout=90):
    checkpoint, result = root / 'checkpoint.pt', root / 'result.json'
    out = {'run_root': str(root), 'checkpoint_found': checkpoint.is_file(),
           'result_found': result.is_file()}
    if result.is_file():
        try:
            r = json.loads(result.read_text())
            out['saved_result'] = {'status': r.get('status'), 'model_key': r.get('model_key'),
                'seed': r.get('seed'), 'completed_epochs': r.get('completed_epochs'),
                'top1_percent': accuracy(r.get('final_validation', {}).get('accuracy')),
                'top5_percent': accuracy(r.get('final_validation', {}).get('top5_accuracy')),
                'contract_sha256': r.get('contract_sha256')}
        except (OSError, ValueError) as exc:
            out['result_error'] = type(exc).__name__ + ': ' + str(exc)[:300]
    # Print presence before trying imports/deserialization, so a timeout still has evidence.
    print('K96_PRESENCE=' + json.dumps(out), flush=True)
    if checkpoint.is_file():
        out['checkpoint_bytes'] = checkpoint.stat().st_size
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        out['checkpoint_sha256'] = digest.hexdigest()
        try:
            child = subprocess.run([sys.executable, '-u', __file__, '--metadata-file', str(checkpoint)],
                                   capture_output=True, text=True, timeout=timeout)
            if child.returncode:
                out['checkpoint_error'] = child.stderr[-1500:]
            else:
                out['checkpoint_metadata'] = json.loads(child.stdout)
                a = out.get('saved_result', {}).get('top1_percent')
                b = out['checkpoint_metadata']['saved_top1_percent']
                out['saved_accuracies_agree'] = None if a is None or b is None else abs(a-b) < 1e-8
        except subprocess.TimeoutExpired:
            out['checkpoint_error'] = 'CPU metadata read timed out; file existence was confirmed'
        except ValueError:
            out['checkpoint_error'] = 'Could not decode checkpoint metadata output'
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-file', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.metadata_file:
        if args.metadata_file not in [root / 'checkpoint.pt' for root in ROOTS]:
            parser.error('Only exact original-run checkpoint candidates are allowed')
        print(json.dumps(checkpoint_metadata(args.metadata_file)))
        return
    print('K96_PROBE_START read_only=true training_started=false', flush=True)
    candidates = []
    for root in ROOTS:
        try: candidates.append(inspect(root))
        except OSError as exc:
            candidates.append({'run_root': str(root), 'access_error': type(exc).__name__+': '+str(exc)[:300]})
    print('K96_PROBE_RESULT=' + json.dumps({'read_only': True, 'training_started': False,
        'original_issue': 379, 'seed': 501, 'expected_epochs': 100,
        'historical_top1_percent': 72.374, 'accuracy_is_re_evaluated': False,
        'note': 'Accuracy comes from saved records, not fresh inference. Missing files mean not visible here, not proof of deletion.',
        'candidates': candidates}), flush=True)


if __name__ == '__main__': main()

"""One fixed-pretrain COCO seed521 on a full H200, with bounded proof first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    with args.checkpoint.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == args.checkpoint_sha256
    output = args.root / 'run'
    scripts = Path(__file__).resolve().parent
    python = sys.executable
    current = args.root / 'current.json'
    stopped = False
    child = None

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    shared = ['--task', 'coco', '--model', 'va_k96', '--seed', '521',
              '--checkpoint', str(args.checkpoint), '--data-root', str(args.data_root),
              '--output-root', str(output), '--physical-batch-size', '2',
              '--effective-batch-size', '16', '--workers', '0',
              '--prefetch-factor', '1', '--sharing-strategy', 'file_system']
    atomic_json(current, {'job': 'k96coco-521', 'seed': 521, 'stage': 'validating',
                          'output': str(output), 'training_started': False})
    print('K96_COCO521 validating full input and checkpoint restore', flush=True)
    child = subprocess.Popen([python, '-u', str(scripts / 'validate_dense_transfer_runtime.py'),
                              *shared, '--mode', 'smoke', '--max-probe-updates', '2'])
    code = child.wait()
    if stopped:
        return 130
    if code:
        raise RuntimeError(f'K96 COCO readiness failed: {code}')
    receipt = read_json(output / 'queue/readiness.json')
    if receipt.get('status') != 'ready' or receipt.get('evidence', {}).get('production_train_started'):
        raise RuntimeError('Readiness proof is incomplete')
    atomic_json(current, {'job': 'k96coco-521', 'seed': 521, 'stage': 'training',
                          'output': str(output), 'training_started': True})
    print('K96_COCO521 training fixed downstream seed521', flush=True)
    command = [python, '-u', str(scripts / 'run_dense_transfer.py'), *shared,
               '--mode', 'train', '--confirm-training', 'DENSE_TRANSFER_TRAIN']
    checkpoint = output / 'checkpoints/last.pt'
    if checkpoint.is_file():
        command.extend(['--resume', str(checkpoint)])
    finish = threading.Event()
    progress_path = output / 'status/progress.json'
    step_path = output / 'step-progress.json'

    def bridge():
        previous = -1
        while not finish.wait(15):
            progress = read_json(progress_path).get('progress', {})
            step = int(progress.get('optimizer_updates', 0))
            if step != previous:
                atomic_json(step_path, {'global_step': step, 'epoch': progress.get('epoch', 0)})
                print('K96_COCO521_PROGRESS=' + json.dumps({'updates': step, 'target': 88716}), flush=True)
                previous = step

    watcher = threading.Thread(target=bridge, daemon=True)
    watcher.start()
    try:
        child = subprocess.Popen(command)
        code = child.wait()
    finally:
        finish.set()
        watcher.join(timeout=20)
    if stopped:
        return 130
    if code:
        raise RuntimeError(f'K96 COCO training exited {code}')
    final = read_json(output / 'status/final.json')
    if final.get('state') != 'completed' or final.get('progress', {}).get('optimizer_updates') != 88716:
        raise RuntimeError('Full 12-epoch completion was not verified')
    metrics = final.get('evaluation', {})
    if not all(key in metrics for key in ('bbox_AP', 'segm_AP')):
        raise RuntimeError('Full COCO box/mask evaluation is missing')
    result = {'status': 'completed', 'completed_epochs': 12, 'global_step': 88716,
              'final_validation': metrics, 'checkpoint': str(checkpoint), 'seed': 521}
    atomic_json(output / 'result.json', result)
    atomic_json(current, {'job': 'k96coco-521', 'seed': 521, 'stage': 'completed',
                          'output': str(output), 'training_started': True})
    print('K96_COCO521_RESULT=' + json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

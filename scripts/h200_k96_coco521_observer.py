"""Owner-side private H200 COCO log and W&B observer; no GPU or job control."""
import argparse
import json
from pathlib import Path
import sys
import time

from in1k10_transport import API, atomic, error_details, read

BASE = 'https://lnet-h200-baseline-relay-v1.gpupulse-monitor.workers.dev/k96coco'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--secrets', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--code-sha', required=True)
    args = parser.parse_args()
    if len(args.code_sha) != 40 or any(character not in '0123456789abcdef' for character in args.code_sha):
        raise ValueError('Expected an exact deployed Git commit SHA')
    args.root.mkdir(parents=True, exist_ok=True)
    api = API(read(args.secrets)['IN10_OWNER_TOKEN'], base=BASE)
    state_path = args.root / 'observer-state.json'
    state = read(state_path)
    cursor = int(state.get('cursor', 0))
    session_id = state.get('session')
    run = None
    last_update = -1
    while True:
        try:
            view = api.call(f'/snapshot?after={cursor}', attempts=1, timeout=8)
            session = view.get('session')
            if session and session['code'] == args.code_sha:
                if session['id'] != session_id:
                    session_id = session['id']
                    cursor = 0
                if run is None:
                    import wandb
                    run = wandb.init(entity='daehwa', project='alphabet2d-dense-transfer',
                                     group='coco-k96-3seed-20260922',
                                     id='k96coco521h200fastsep24', resume='allow',
                                     name='COCO-VA-K96-seed521-H200-fast', mode='online',
                                     config={'task': 'coco', 'model': 'va_k96', 'seed': 521,
                                             'pretrain_seed': 501, 'pretrain_epochs': 100,
                                             'epochs': 12, 'optimizer_updates': 88716,
                                             'code_sha': args.code_sha},
                                     settings=wandb.Settings(console='off', disable_code=True,
                                                             disable_git=True, x_disable_stats=True,
                                                             init_timeout=45))
                for event in view.get('events', []):
                    event_id = int(event['id'])
                    if event_id <= cursor:
                        continue
                    payload = json.loads(event['value'])
                    with (args.root / 'events.jsonl').open('a') as stream:
                        stream.write(json.dumps({'id': event_id, 'value': payload}) + '\n')
                    cursor = event_id
                    status = payload.get('status', {})
                    update = int(status.get('progress', {}).get('global_step') or 0)
                    if update > last_update:
                        run.log({'optimizer_updates': update,
                                 'epoch': status.get('progress', {}).get('epoch') or 0})
                        last_update = update
                    final = status.get('result', {})
                    if final.get('status') == 'completed':
                        metrics = final.get('final_validation') or {}
                        run.summary.update({'final/' + key: value for key, value in metrics.items()
                                            if isinstance(value, (float, int))})
                        run.summary['completed_updates'] = 88716
                    elif status.get('ended') and status.get('exit_code') not in (None, 0):
                        run.summary['failure'] = status.get('error') or status.get('exit_code')
                atomic(state_path, {'cursor': cursor, 'session': session_id,
                                    'ended': session.get('ended', False), 'last_seen': time.time()})
                if session.get('ended') and not view.get('events'):
                    run.finish()
                    return
            else:
                atomic(state_path, {'cursor': cursor, 'session': session_id,
                                    'waiting_for_code_sha': args.code_sha,
                                    'last_checked': time.time()})
        except Exception as error:
            # A telemetry outage never interrupts the H200 training process.
            atomic(args.root / 'observer-error.json', error_details(error))
            print('observer warning: ' + type(error).__name__, file=sys.stderr, flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()

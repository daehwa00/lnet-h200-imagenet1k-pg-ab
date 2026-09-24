"""Owner-only H200 stop/status control; token stays in a private local file."""
import argparse
from pathlib import Path
from in1k10_transport import API, read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--secrets', type=Path, required=True)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    api = API(read(args.secrets)['IN10_OWNER_TOKEN'],
              base='https://lnet-h200-baseline-relay-v1.gpupulse-monitor.workers.dev/k96coco')
    if args.stop:
        session = api.call('/snapshot?after=9007199254740991', attempts=1, timeout=8).get('session')
        if not session or session.get('ended'):
            print({'stop_requested': False, 'reason': 'no active H200 session'})
            return
        result = api.call('/command', {'action': 'stop'}, 'POST', attempts=1, timeout=8)
        print({'stop_requested': result.get('stop', False)})
    else:
        view = api.call('/snapshot?after=9007199254740991', attempts=1, timeout=8)
        print({'session': view.get('session'), 'stop_requested': view['control']['stop'],
               'events': len(view.get('events', []))})


if __name__ == '__main__':
    main()

"""Download-only persistent H200 preparation; never launch training."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

import prepare_dense_transfer_assets as assets

ROOT = Path('/app/output/daehwa00/dense-transfer')


def prepare(root: Path = ROOT, *, only_coco: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {'ready': False, 'imagenet_root': '/app/data/ImageNet-2012',
                  'imagenet_exists': Path('/app/data/ImageNet-2012').is_dir(),
                  'datasets': {}, 'archives': [], 'training_started': False}
        try:
            for kind, dirname, verify, candidates in (
                ('coco', 'coco', assets.verify_coco, [Path('/app/data/coco'), Path('/app/data/COCO')]),
                ('ade20k', 'ADEChallengeData2016', assets.verify_ade, [Path('/app/data/ADEChallengeData2016'), Path('/app/data/ade20k')]),
            ):
                if only_coco and kind != 'coco':
                    continue
                target = root / 'datasets' / dirname
                selected = next((p for p in [target, *candidates] if p.is_dir() and verify(p)['ready']), None)
                if selected is None:
                    if target.exists():
                        raise RuntimeError(f'Existing incomplete dataset left untouched: {target}')
                    stage = Path(tempfile.mkdtemp(prefix=f'.{kind}-', dir=root))
                    archive_root = root / 'archives'
                    archive_root.mkdir(exist_ok=True)
                    for spec in assets.ARCHIVE_SPECS:
                        if not (spec.kind == kind or (kind == 'coco' and spec.kind.startswith('coco_'))):
                            continue
                        print(f'Download/verify {spec.name}', flush=True)
                        for attempt in range(3):
                            try:
                                observed = assets.download_archive(spec, archive_root / spec.name)
                                break
                            except (assets.DownloadError, OSError):
                                if attempt == 2:
                                    raise
                                time.sleep(5)
                        report['archives'].append(observed)
                        assets.safe_extract_zip(archive_root / spec.name, stage)
                    prepared = stage / dirname if kind == 'ade20k' else stage
                    verification = verify(prepared)
                    if not verification['ready']:
                        raise RuntimeError(f'Dataset validation failed: {verification}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise RuntimeError(f'Refusing replacement: {target}')
                    os.rename(prepared, target)
                    selected = target
                report['datasets'][kind] = {'path': str(selected), 'verification': verify(selected)}
                assets._atomic_json(root / 'datasets-ready.json', report)
                print(f'{kind}: verified at {selected}', flush=True)
            report['ready'] = True
        except Exception as error:
            report['error'] = str(error)
            raise
        finally:
            assets._atomic_json(root / 'datasets-ready.json', report)
        return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-coco', action='store_true')
    args = parser.parse_args()
    result = prepare(only_coco=args.only_coco)
    print(json.dumps({'ready': result['ready'], 'manifest': str(ROOT / 'datasets-ready.json'),
                      'datasets': {k: v['path'] for k, v in result['datasets'].items()},
                      'training_started': False}), flush=True)

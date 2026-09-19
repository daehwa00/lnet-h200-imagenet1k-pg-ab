"""Official split, image labels only. No bounding boxes or part supervision."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from PIL import Image
from torch.utils.data import Dataset


def pairs(path):
    rows = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1)
        key = int(key)
        if key in rows:
            raise ValueError(f'Duplicate ID in {path.name}: {key}')
        rows[key] = value
    return rows


def manifest(root, *, official_counts=True):
    root = Path(root)
    images = pairs(root/'images.txt')
    labels = pairs(root/'image_class_labels.txt')
    splits = pairs(root/'train_test_split.txt')
    if set(images) != set(labels) or set(images) != set(splits):
        raise ValueError('Split, labels, images must have identical ID sets')
    records = []
    names = set()
    for key in sorted(images):
        name = images[key]
        posix = PurePosixPath(name)
        if posix.is_absolute() or '..' in posix.parts or name in names:
            raise ValueError('Unsafe or duplicate image path')
        names.add(name)
        label, split = int(labels[key])-1, int(splits[key])
        if not 0 <= label < 200 or split not in (0, 1):
            raise ValueError('Invalid class or split')
        path = root/'images'/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to((root/'images').resolve()):
            raise ValueError(f'Unsafe or missing image {name}')
        records.append((key, name, label, split))
    train = sum(r[3] for r in records)
    test = len(records)-train
    if official_counts and (train, test, len({r[2] for r in records})) != (5994, 5794, 200):
        raise ValueError(f'Not the official CUB2011 split: {train}/{test}')
    digest = hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()
    return {'records': records, 'train': train, 'test': test, 'split_sha256': digest}


class CUB(Dataset):
    def __init__(self, root, records, train, transform):
        self.root = Path(root)
        self.rows = [r for r in records if bool(r[3]) == train]
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        _, name, label, _ = self.rows[index]
        with Image.open(self.root/'images'/name) as image:
            image = image.convert('RGB')
        return self.transform(image), label

import gc
import os
from pathlib import Path
import pickle
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from in1k10_ipc import BatchLoader, SharedBatch, collate


class Fixture(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        # Include negative zero: equality alone does not establish byte fidelity.
        image = torch.arange(48, dtype=torch.float32).reshape(3,4,4) + index
        image[0,0,0] = -0.0
        return image, index


def fd_count():
    return len(os.listdir('/proc/self/fd'))


def test_metadata_only_and_byte_exact():
    fixture = Fixture()
    batch = collate([fixture[i] for i in range(4)])
    assert len(pickle.dumps(batch)) < 1024
    result = batch.materialize()
    expected = torch.stack([fixture[i][0] for i in range(4)])
    assert torch.equal(result.images.view(torch.int32), expected.view(torch.int32))
    assert result.labels.tolist() == list(range(4))


def test_spawn_persistent_epochs_fd_stability():
    loader = DataLoader(Fixture(), batch_size=4, num_workers=2, prefetch_factor=1,
                        persistent_workers=True, multiprocessing_context='spawn', collate_fn=collate)
    wrapped = BatchLoader(loader)
    counts = []
    worker_counts = []
    try:
        for _ in range(5):
            for index, (images, labels) in enumerate(wrapped):
                expected = torch.stack([Fixture()[i][0] for i in range(index*4,index*4+4)])
                assert torch.equal(images.view(torch.int32), expected.view(torch.int32))
                assert labels.tolist() == list(range(index*4,index*4+4))
            gc.collect()
            counts.append(fd_count())
            worker_counts.append([len(os.listdir(f'/proc/{w.pid}/fd')) for w in loader._iterator._workers])
        assert max(counts[1:])-min(counts[1:]) <= 2
        for column in zip(*worker_counts[1:]):
            assert max(column)-min(column) <= 2
        assert wrapped.timing['host_image_bytes'] == 12*48*4
        assert wrapped.timing['loader_wait_seconds'] > 0
        assert wrapped.timing['pin_seconds'] > 0
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_invalid_metadata_closes_received_fd():
    batch = collate([Fixture()[0]])
    batch.shape = (1, 3, -1, 4)
    before = fd_count()
    with pytest.raises(ValueError, match='shape'):
        batch.materialize()
    assert fd_count() <= before


def test_repeated_inprocess_no_fd_growth():
    # Warm up multiprocessing resource-sharer socket/thread first.
    collate([Fixture()[0]]).materialize()
    before = fd_count()
    for _ in range(30):
        collate([Fixture()[0]]).materialize()
    assert fd_count() <= before + 1


def test_allocation_failure_closes_fd_and_mapping(monkeypatch):
    batch = collate([Fixture()[0]])
    before = fd_count()
    def fail(*args, **kwargs):
        raise RuntimeError('allocation failed')
    monkeypatch.setattr(torch, 'empty', fail)
    with pytest.raises(RuntimeError, match='allocation failed'):
        batch.materialize()
    assert fd_count() <= before


def test_dataloader_pin_dispatch_without_gpu(monkeypatch):
    from torch.utils.data._utils.pin_memory import pin_memory
    calls = []
    empty = torch.empty
    def host_empty(*args, **kwargs):
        calls.append(kwargs.pop('pin_memory', False))
        return empty(*args, **kwargs)
    batch = collate([Fixture()[0]])
    monkeypatch.setattr(torch, 'empty', host_empty)
    ready = pin_memory(batch)
    assert calls == [True, True]
    assert ready.labels.tolist() == [0]

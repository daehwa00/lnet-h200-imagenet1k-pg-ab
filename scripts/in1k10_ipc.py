"""Lossless Linux memfd batches; only descriptors cross the worker pipe.

memfd pages use RAM, not the container's small /dev/shm mount. A received
mapping is copied into parent-owned (optionally pinned) tensors before closing.
DataLoader invokes SharedBatch.pin_memory in its normal pin-memory thread.
"""
from dataclasses import dataclass
import ctypes
import math
import mmap
import os
import time
from multiprocessing.reduction import DupFd

import numpy as np
import torch

MAX_BATCH_BYTES = 1024 * 1024**2


def _memfd_create():
    if hasattr(os, 'memfd_create'):
        return os.memfd_create('in1k10-batch', os.MFD_CLOEXEC)
    # Portable Python builds can omit the os wrapper even on a capable Linux.
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(b'in1k10-batch', 1)  # Linux MFD_CLOEXEC
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return fd


@dataclass
class ReadyBatch:
    images: torch.Tensor
    labels: torch.Tensor
    pin_seconds: float
    host_image_bytes: int


@dataclass
class SharedBatch:
    descriptor: object
    shape: tuple
    size: int

    def materialize(self, pin=False):
        """Consume the descriptor once; close it even on allocation/copy failure."""
        fd = self.descriptor.detach()
        mapping = None
        source = None
        started = time.perf_counter()
        try:
            if len(self.shape) != 4 or any(type(n) is not int or n <= 0 for n in self.shape):
                raise ValueError('Invalid batch shape')
            image_bytes = math.prod(self.shape) * 4
            expected = image_bytes + self.shape[0] * 8
            if expected != self.size or not 0 < expected <= MAX_BATCH_BYTES:
                raise ValueError('Invalid batch size')
            if os.fstat(fd).st_size != expected:
                raise ValueError('Descriptor size mismatch')
            mapping = mmap.mmap(fd, expected, access=mmap.ACCESS_WRITE)
            source = torch.frombuffer(mapping, dtype=torch.float32, count=math.prod(self.shape))
            images = torch.empty(self.shape, dtype=torch.float32, pin_memory=pin)
            images.view(-1).copy_(source)
            source = torch.frombuffer(mapping, dtype=torch.int64, count=self.shape[0], offset=image_bytes)
            labels = torch.empty((self.shape[0],), dtype=torch.int64, pin_memory=pin)
            labels.copy_(source)
            return ReadyBatch(images, labels, time.perf_counter()-started, image_bytes)
        finally:
            source = None
            if mapping is not None:
                mapping.close()
            os.close(fd)

    def pin_memory(self):
        return self.materialize(pin=True)


def collate(batch):
    """Preserve float32 image bits and int64 labels without tensor IPC."""
    if not batch:
        raise ValueError('Empty batch')
    images, labels = zip(*batch)
    shape = (len(images), *images[0].shape)
    if len(shape) != 4 or any(x.dtype != torch.float32 or tuple(x.shape) != shape[1:] or x.device.type != 'cpu' for x in images):
        raise ValueError('Expected equally shaped CPU float32 CHW images')
    image_bytes = math.prod(shape) * 4
    size = image_bytes + len(labels) * 8
    if not 0 < size <= MAX_BATCH_BYTES:
        raise ValueError('Batch exceeds memfd limit')
    fd = _memfd_create()
    mapping = None
    target = None
    try:
        os.ftruncate(fd, size)
        mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        target = np.ndarray(shape, dtype=np.float32, buffer=mapping)
        for index, image in enumerate(images):
            np.copyto(target[index], image.detach().numpy(), casting='no')
        target = np.ndarray((len(labels),), dtype=np.int64, buffer=mapping, offset=image_bytes)
        target[:] = labels
        # DupFd owns a duplicate until the receiver detaches it. Original closes.
        return SharedBatch(DupFd(fd), shape, size)
    finally:
        target = None
        if mapping is not None:
            mapping.close()
        os.close(fd)


memfd_collate = collate


class BatchLoader:
    """DataLoader adapter with cumulative non-synchronizing host timings.

    pin_seconds includes allocation/materialization and may overlap loader wait
    when DataLoader's pin thread is enabled; do not add these as GPU idle time.
    """
    def __init__(self, loader, pin=False):
        self.loader = loader
        self.pin = pin
        self.timing = dict(loader_wait_seconds=0.0, pin_seconds=0.0, host_image_bytes=0)

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def _iterator(self):
        return self.loader._iterator

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.timing = dict(loader_wait_seconds=0.0, pin_seconds=0.0, host_image_bytes=0)
        iterator = iter(self.loader)
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.timing['loader_wait_seconds'] += time.perf_counter()-started
            if isinstance(batch, SharedBatch):
                batch = batch.materialize(pin=self.pin)
            if not isinstance(batch, ReadyBatch):
                raise TypeError('Expected memfd batch')
            self.timing['pin_seconds'] += batch.pin_seconds
            self.timing['host_image_bytes'] += batch.host_image_bytes
            yield batch.images, batch.labels

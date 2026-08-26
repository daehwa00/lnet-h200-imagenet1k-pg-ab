"""Immutable fixed-length token blocks for ALPHABET-LM viability runs."""

from __future__ import annotations

# pyright: reportExplicitAny=false
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TokenBlockManifest:
    schema: str
    split: str
    token_file: str
    token_file_sha256: str
    tokenizer_sha256: str
    source_sha256: str
    sample_count: int
    real_token_count: int
    block_size: int
    context_length: int
    vocab_size: int
    pad_id: int
    eos_id: int
    dtype: str
    cross_document_packing: bool

    @classmethod
    def load(cls, path: Path) -> TokenBlockManifest:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(**payload)
        if (
            manifest.schema != "lnet.alphabet_lm.token_blocks.v1"
            or manifest.dtype != "uint16"
            or manifest.block_size != manifest.context_length + 1
            or manifest.sample_count <= 0
            or manifest.real_token_count <= 0
            or manifest.cross_document_packing
        ):
            raise ValueError("invalid ALPHABET-LM token-block manifest")
        return manifest


class TokenBlockDataset(Dataset[Tensor]):
    def __init__(self, manifest_path: Path, *, verify_sha256: bool = False) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = TokenBlockManifest.load(self.manifest_path)
        self.token_path = self.manifest_path.parent / self.manifest.token_file
        if verify_sha256 and sha256_file(self.token_path) != self.manifest.token_file_sha256:
            raise RuntimeError("token-block shard SHA256 mismatch")
        expected_bytes = self.manifest.sample_count * self.manifest.block_size * 2
        if self.token_path.stat().st_size != expected_bytes:
            raise RuntimeError("token-block shard size disagrees with manifest")
        self._tokens = np.memmap(
            self.token_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.manifest.sample_count, self.manifest.block_size),
        )

    def __len__(self) -> int:
        return self.manifest.sample_count

    def __getitem__(self, index: int) -> Tensor:
        return torch.tensor(self._tokens[index], dtype=torch.long)


__all__ = ["TokenBlockDataset", "TokenBlockManifest", "sha256_file"]

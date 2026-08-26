#!/usr/bin/env python3
"""Prepare one immutable FineWeb-Edu token stream for H200 LM viability runs."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportMissingImports=false
import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np

from lnet.alphabet_lm_data import TokenBlockManifest, sha256_file

SPECIAL_TOKENS = ("<pad>", "<eos>", "<unk>")
RUNTIME_SCHEMA = "lnet.h200.alphabet_lm.viability_10m.runtime.v1"
KAU_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.pole_init_10m.runtime.v1"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _documents(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            identifier = payload.get("id")
            if not isinstance(identifier, str) or not identifier:
                identifier = hashlib.sha256(text.encode()).hexdigest()
            yield identifier, text


def _parquet_to_jsonl(parquet_path: Path, output: Path) -> None:
    import pyarrow.parquet as pq

    if output.is_file():
        return
    parquet = pq.ParquetFile(parquet_path)
    names = set(parquet.schema.names)
    if "text" not in names:
        raise RuntimeError("FineWeb-Edu parquet has no text column")
    columns = ["text", *( ["id"] if "id" in names else [])]
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            values = batch.to_pydict()
            identifiers = values.get("id")
            for index, text in enumerate(values["text"]):
                if not isinstance(text, str) or not text.strip():
                    continue
                identifier = None if identifiers is None else identifiers[index]
                if not isinstance(identifier, str) or not identifier:
                    identifier = hashlib.sha256(text.encode()).hexdigest()
                handle.write(json.dumps({"id": identifier, "text": text}, ensure_ascii=False))
                handle.write("\n")
    temporary.replace(output)


def _split_documents(
    source: Path,
    train: Path,
    validation: Path,
    *,
    validation_fraction: float,
    salt: str,
) -> None:
    if train.is_file() and validation.is_file():
        return
    train_temporary = train.with_suffix(train.suffix + f".tmp-{os.getpid()}")
    validation_temporary = validation.with_suffix(validation.suffix + f".tmp-{os.getpid()}")
    with train_temporary.open("w", encoding="utf-8") as train_handle, validation_temporary.open(
        "w", encoding="utf-8"
    ) as validation_handle:
        for identifier, text in _documents(source):
            value = int.from_bytes(hashlib.sha256(f"{salt}\0{identifier}".encode()).digest()[:8])
            active = validation_handle if value / 2**64 < validation_fraction else train_handle
            active.write(json.dumps({"id": identifier, "text": text}, ensure_ascii=False) + "\n")
    train_temporary.replace(train)
    validation_temporary.replace(validation)


def _train_tokenizer(train_jsonl: Path, output: Path, *, vocab_size: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    if output.is_file():
        return
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))  # noqa: S106 - token, not credential
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=list(SPECIAL_TOKENS))
    tokenizer.train_from_iterator((text for _identifier, text in _documents(train_jsonl)), trainer)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    tokenizer.save(str(temporary))
    temporary.replace(output)


def _fixed_blocks(
    documents: Iterable[tuple[str, str]],
    tokenizer: Any,
    *,
    block_size: int,
    pad_id: int,
    eos_id: int,
    token_limit: int,
) -> Iterator[tuple[np.ndarray, int]]:
    emitted_tokens = 0
    for _identifier, text in documents:
        ids = [*tokenizer.encode(text).ids, eos_id]
        for offset in range(0, len(ids), block_size):
            active = ids[offset : offset + block_size]
            if len(active) < 2 or emitted_tokens >= token_limit:
                break
            real = min(len(active), token_limit - emitted_tokens)
            if real < 2:
                break
            block = np.full(block_size, pad_id, dtype=np.uint16)
            block[:real] = np.asarray(active[:real], dtype=np.uint16)
            emitted_tokens += real
            yield block, real
        if emitted_tokens >= token_limit:
            break


def _tokenize_split(
    source: Path,
    tokenizer_path: Path,
    output: Path,
    *,
    split: str,
    context_length: int,
    token_limit: int,
) -> Path:
    from tokenizers import Tokenizer

    manifest_path = output / f"{split}.manifest.json"
    if manifest_path.is_file():
        manifest = TokenBlockManifest.load(manifest_path)
        token_path = output / manifest.token_file
        if (
            manifest.real_token_count == token_limit
            and manifest.token_file_sha256 == sha256_file(token_path)
        ):
            return manifest_path
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id, eos_id = (tokenizer.token_to_id(token) for token in SPECIAL_TOKENS[:2])
    if pad_id is None or eos_id is None or tokenizer.get_vocab_size() != 32_768:
        raise RuntimeError("tokenizer violates the frozen 32K token contract")
    output.mkdir(parents=True, exist_ok=True)
    token_path = output / f"{split}.uint16.bin"
    temporary = token_path.with_suffix(token_path.suffix + f".tmp-{os.getpid()}")
    samples = 0
    real_tokens = 0
    with temporary.open("wb") as handle:
        for block, real in _fixed_blocks(
            _documents(source),
            tokenizer,
            block_size=context_length + 1,
            pad_id=pad_id,
            eos_id=eos_id,
            token_limit=token_limit,
        ):
            block.tofile(handle)
            samples += 1
            real_tokens += real
    if real_tokens != token_limit:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{split} source yielded {real_tokens}, expected {token_limit} tokens")
    temporary.replace(token_path)
    _atomic_json(
        manifest_path,
        {
            "schema": "lnet.alphabet_lm.token_blocks.v1",
            "split": split,
            "token_file": token_path.name,
            "token_file_sha256": sha256_file(token_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "source_sha256": sha256_file(source),
            "sample_count": samples,
            "real_token_count": real_tokens,
            "block_size": context_length + 1,
            "context_length": context_length,
            "vocab_size": tokenizer.get_vocab_size(),
            "pad_id": pad_id,
            "eos_id": eos_id,
            "dtype": "uint16",
            "cross_document_packing": False,
        },
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    if runtime.get("schema") not in {RUNTIME_SCHEMA, KAU_RUNTIME_SCHEMA}:
        raise RuntimeError("invalid H200 ALPHABET-LM data runtime")
    contract = cast("dict[str, Any]", runtime["dataset"])
    args.root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    parquet_path = Path(
        hf_hub_download(
            repo_id=contract["repo_id"],
            filename=contract["filename"],
            repo_type=contract["repo_type"],
            revision=contract["revision"],
            cache_dir=args.root / "huggingface-cache",
        )
    )
    if parquet_path.stat().st_size != contract["size_bytes"]:
        raise RuntimeError("FineWeb-Edu parquet size mismatch")
    if sha256_file(parquet_path) != contract["sha256"]:
        raise RuntimeError("FineWeb-Edu parquet SHA256 mismatch")
    documents = args.root / "documents.jsonl"
    train_jsonl = args.root / "train.documents.jsonl"
    validation_jsonl = args.root / "validation.documents.jsonl"
    tokenizer_path = args.root / "tokenizer.json"
    _parquet_to_jsonl(parquet_path, documents)
    _split_documents(
        documents,
        train_jsonl,
        validation_jsonl,
        validation_fraction=contract["validation_fraction"],
        salt=contract["document_split_salt"],
    )
    _train_tokenizer(train_jsonl, tokenizer_path, vocab_size=contract["vocab_size"])
    train_manifest = _tokenize_split(
        train_jsonl,
        tokenizer_path,
        args.root / "tokens",
        split="train",
        context_length=contract["context_length"],
        token_limit=contract["train_token_limit"],
    )
    validation_manifest = _tokenize_split(
        validation_jsonl,
        tokenizer_path,
        args.root / "tokens",
        split="validation",
        context_length=contract["context_length"],
        token_limit=contract["validation_token_limit"],
    )
    receipt = {
        "schema": "lnet.h200.alphabet_lm.data_receipt.v1",
        "dataset_contract": contract,
        "parquet_sha256": sha256_file(parquet_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest": str(validation_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
    }
    _atomic_json(args.root / "receipt.json", receipt)
    print("ALPHABET_LM_DATA=" + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

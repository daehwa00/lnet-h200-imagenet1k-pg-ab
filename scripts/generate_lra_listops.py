#!/usr/bin/env python3
# Derived from Google Research Long Range Arena ListOps generator (Apache-2.0).
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import statistics
from pathlib import Path

type Tree = int | str | tuple[Tree, Tree]

OPERATORS = ("[MIN", "[MAX", "[MED", "[SM")
END = "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--train", type=int, default=96_000)
    parser.add_argument("--validation", type=int, default=2_000)
    parser.add_argument("--test", type=int, default=2_000)
    arguments = parser.parse_args()
    generate(
        arguments.output_root,
        arguments.seed,
        arguments.train,
        arguments.validation,
        arguments.test,
    )


def generate(output_root: Path, seed: int, train: int, validation: int, test: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    writers: list[tuple[int, csv.writer, object]] = []
    for name, count in (("train", train), ("val", validation), ("test", test)):
        handle = (output_root / f"basic_{name}.tsv").open("w", encoding="utf-8", newline="")
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("Source", "Target"))
        writers.append((count, writer, handle))
    random_generator = random.Random(seed)  # noqa: S311 - benchmark generator, not security
    seen: set[bytes] = set()
    accepted = 0
    writer_index = 0
    written_to_split = 0
    total = train + validation + test
    try:
        while accepted < total:
            tree, length = _generate_tree(random_generator, 1, 10, 10)
            if not 500 < length < 2000:
                continue
            source = _to_string(tree)
            digest = hashlib.sha256(source.encode()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            split_count, writer, _ = writers[writer_index]
            writer.writerow((source, _to_value(tree)))
            accepted += 1
            written_to_split += 1
            if written_to_split == split_count and writer_index + 1 < len(writers):
                writer_index += 1
                written_to_split = 0
            if accepted % 1000 == 0:
                print(f"generated {accepted}/{total}", flush=True)  # noqa: T201
    finally:
        for _, _, handle in writers:
            handle.close()  # type: ignore[attr-defined]


def _generate_tree(
    generator: random.Random,
    depth: int,
    max_depth: int,
    max_args: int,
) -> tuple[Tree, int]:
    if depth >= max_depth or generator.random() > 0.25:
        return generator.randrange(10), 1
    length = 2
    values: list[Tree] = []
    for _ in range(generator.randint(2, max_args)):
        subtree, subtree_length = _generate_tree(generator, depth + 1, max_depth, max_args)
        values.append(subtree)
        length += subtree_length
    tree: Tree = (generator.choice(OPERATORS), values[0])
    for value in values[1:]:
        tree = (tree, value)
    return (tree, END), length


def _to_string(tree: Tree) -> str:
    if isinstance(tree, (str, int)):
        return str(tree)
    return f"( {_to_string(tree[0])} {_to_string(tree[1])} )"


def _to_value(tree: Tree) -> int | tuple[str, list[int]] | str:  # noqa: PLR0911
    if not isinstance(tree, tuple):
        return tree
    left = _to_value(tree[0])
    right = _to_value(tree[1])
    if left in OPERATORS:
        return str(left), [int(right)]
    if right == END:
        operator, values = left  # type: ignore[misc]
        if operator == "[MIN":
            return min(values)
        if operator == "[MAX":
            return max(values)
        if operator == "[MED":
            return int(statistics.median(values))
        return sum(values) % 10
    operator, values = left  # type: ignore[misc]
    return operator, [*values, int(right)]


if __name__ == "__main__":
    main()

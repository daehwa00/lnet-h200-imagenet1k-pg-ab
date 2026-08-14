from __future__ import annotations

import argparse
from pathlib import Path

from lnet.pac_external_preparation import (
    prepare_audioset_task,
    prepare_imdb_task,
    prepare_listops_task,
    prepare_retrieval_task,
    prepare_vision_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    vision = subparsers.add_parser("vision")
    vision.add_argument("--source-root", type=Path, required=True)
    vision.add_argument("--output-root", type=Path, required=True)

    imdb = subparsers.add_parser("imdb")
    imdb.add_argument("--archive", type=Path, required=True)
    imdb.add_argument("--output", type=Path, required=True)

    listops = subparsers.add_parser("listops")
    _add_split_paths(listops)
    listops.add_argument("--output", type=Path, required=True)

    retrieval = subparsers.add_parser("retrieval")
    _add_split_paths(retrieval)
    retrieval.add_argument("--output", type=Path, required=True)

    audioset = subparsers.add_parser("audioset")
    audioset.add_argument("--feature-archive", type=Path, required=True)
    audioset.add_argument("--balanced-csv", type=Path, required=True)
    audioset.add_argument("--evaluation-csv", type=Path, required=True)
    audioset.add_argument("--class-csv", type=Path, required=True)
    audioset.add_argument("--output", type=Path, required=True)

    arguments = parser.parse_args()
    if arguments.command == "vision":
        prepare_vision_tasks(arguments.source_root, arguments.output_root)
    elif arguments.command == "imdb":
        prepare_imdb_task(arguments.archive, arguments.output)
    elif arguments.command == "listops":
        prepare_listops_task(
            arguments.train, arguments.validation, arguments.test, arguments.output
        )
    elif arguments.command == "retrieval":
        prepare_retrieval_task(
            arguments.train, arguments.validation, arguments.test, arguments.output
        )
    elif arguments.command == "audioset":
        prepare_audioset_task(
            arguments.feature_archive,
            arguments.balanced_csv,
            arguments.evaluation_csv,
            arguments.class_csv,
            arguments.output,
        )


def _add_split_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)


if __name__ == "__main__":
    main()

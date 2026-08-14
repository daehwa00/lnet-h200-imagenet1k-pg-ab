from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the LRA Pathfinder32 hard task with the original generator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-shard")
    generate.add_argument("--generator-root", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--batch-id", type=int, required=True)
    generate.add_argument("--images", type=int, required=True)
    generate.add_argument("--seed", type=int, default=1729)

    manifest = subparsers.add_parser("write-manifest")
    manifest.add_argument("--output-root", type=Path, required=True)
    manifest.add_argument("--expected-images", type=int, required=True)
    manifest.add_argument("--seed", type=int, default=1729)
    return parser


def _generate_shard(args: argparse.Namespace) -> None:
    generator_root = args.generator_root.resolve()
    sys.path.insert(0, str(generator_root))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import snakes2  # pyright: ignore[reportMissingImports]  # noqa: PLC0415

    def save_float_png(path: str, array: np.ndarray) -> None:
        # Match the byte scaling used by the generator's former scipy.misc.imsave path.
        values = np.asarray(array, dtype=np.float32)
        lower = float(values.min())
        upper = float(values.max())
        if upper > lower:
            values = (values - lower) * (255.0 / (upper - lower))
        else:
            values = np.zeros_like(values)
        Image.fromarray(np.clip(values + 0.5, 0.0, 255.0).astype(np.uint8)).save(path)

    snakes2._imsave = save_float_png  # noqa: SLF001

    shard_seed = args.seed + args.batch_id
    random.seed(shard_seed)
    np.random.seed(shard_seed)  # noqa: NPY002 - upstream generator uses global NumPy RNG.
    contour_root = args.output_root.resolve() / "source" / "hard"
    generator_args = SimpleNamespace(
        contour_path=str(contour_root),
        batch_id=args.batch_id,
        n_images=args.images,
        window_size=[32, 32],
        padding=1,
        antialias_scale=2,
        LABEL=1,
        seed_distance=7,
        marker_radius=1.5,
        contour_length=14,
        distractor_length=14 // 3,
        num_distractor_snakes=20 // (14 // 3),
        snake_contrast_list=[2.0],
        use_single_paddles=True,
        max_target_contour_retrial=4,
        max_distractor_contour_retrial=4,
        max_paddle_retrial=2,
        continuity=1.0,
        paddle_length=2,
        paddle_thickness=0.5,
        paddle_margin_list=[1],
        paddle_contrast_list=[0.75],
        pause_display=False,
        save_images=True,
        save_metadata=True,
        segmentation_task=False,
        segmentation_task_double_circle=False,
    )
    snakes2.from_wrapper(generator_args)


def _write_manifest(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    source_root = output_root / "source" / "hard"
    rows: list[tuple[str, int, str]] = []
    for metadata_path in sorted(
        (source_root / "metadata").glob("*.npy"), key=lambda path: int(path.stem)
    ):
        metadata = np.load(metadata_path, allow_pickle=False)
        for entry in metadata:
            relative = Path("source") / "hard" / str(entry[0]) / str(entry[1])
            rows.append((relative.as_posix(), int(entry[3]), metadata_path.stem))
    if len(rows) != args.expected_images:
        message = f"generated {len(rows)} Pathfinder images; expected {args.expected_images}"
        raise RuntimeError(message)

    random.Random(args.seed).shuffle(rows)  # noqa: S311 - deterministic dataset split.
    train_end = int(0.8 * len(rows))
    validation_end = int(0.9 * len(rows))
    manifest_path = output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "split", "group"))
        writer.writeheader()
        for index, (path, label, shard) in enumerate(rows):
            if index < train_end:
                split = "train"
            elif index < validation_end:
                split = "validation"
            else:
                split = "test"
            writer.writerow(
                {"path": path, "label": label, "split": split, "group": f"{shard}:{path}"}
            )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate-shard":
        _generate_shard(args)
    else:
        _write_manifest(args)


if __name__ == "__main__":
    main()

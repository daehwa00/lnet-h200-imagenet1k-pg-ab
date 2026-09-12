# Copyright (c) 2026

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

import numpy as np
import pytest
import torch
from dense_transfer.data import (
    COCODataset,
    ade_collate,
    build_dataset,
    build_loaders,
    coco_collate,
    evaluate,
    map_ade_labels,
)
from dense_transfer.metrics import SegmentationMetric, evaluate_coco
from PIL import Image


def _write_ade_fixture(root: Path) -> None:
    for split in ("training", "validation"):
        (root / "ADEChallengeData2016" / "images" / split).mkdir(parents=True)
        (root / "ADEChallengeData2016" / "annotations" / split).mkdir(parents=True)
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = np.arange(4, dtype=np.uint8)
    mask = np.array([[0, 1, 150, 255], [2, 3, 4, 5], [6, 7, 8, 9]], dtype=np.uint8)
    for split in ("training", "validation"):
        Image.fromarray(image, mode="RGB").save(
            root / "ADEChallengeData2016" / "images" / split / "sample.jpg"
        )
        Image.fromarray(mask, mode="L").save(
            root / "ADEChallengeData2016" / "annotations" / split / "sample.png"
        )


def _write_coco_fixture(root: Path) -> None:
    (root / "train2017").mkdir(parents=True)
    (root / "val2017").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(root / "train2017" / "000000000001.jpg")
    Image.fromarray(image, mode="RGB").save(root / "val2017" / "000000000001.jpg")
    payload = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 5, "height": 4}],
        "categories": [{"id": 17, "name": "fixture"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 17,
                "bbox": [1, 1, 2, 2],
                "area": 4,
                "iscrowd": 0,
                "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]],
            }
        ],
    }
    empty_payload = {**payload, "annotations": []}
    (root / "annotations" / "instances_train2017.json").write_text(json.dumps(payload))
    (root / "annotations" / "instances_val2017.json").write_text(json.dumps(empty_payload))


def test_ade_label_mapping_and_native_outputs(tmp_path: Path) -> None:
    assert torch.equal(
        map_ade_labels(torch.tensor([[0, 1, 150, 255]], dtype=torch.int64)),
        torch.tensor([[255, 0, 149, 255]], dtype=torch.int64),
    )
    _write_ade_fixture(tmp_path)
    dataset = build_dataset("ade20k", tmp_path, "val", train=False)
    image, mask = dataset[0]
    assert image.dtype == torch.float32
    assert image.shape == (3, 512, 683)
    assert mask.dtype == torch.int64
    assert mask.shape == (512, 683)
    assert set(mask.unique().tolist()) >= {0, 1, 149, 255}
    images, masks = ade_collate(
        [
            (torch.ones(3, 3, 4), torch.zeros(3, 4, dtype=torch.int64)),
            (torch.ones(3, 5, 2), torch.zeros(5, 2, dtype=torch.int64)),
        ]
    )
    assert images.shape == (2, 3, 32, 32)
    assert masks.shape == (2, 32, 32)
    assert torch.all(images[0, :, 3:, :] == 0)
    assert torch.all(masks[0, 3:, :] == 255)


def test_ade_collate_aligns_non_square_aspect_ratios_and_masks_padding() -> None:
    wide_image = torch.ones(3, 512, 683)
    wide_mask = torch.zeros(512, 683, dtype=torch.int64)
    tall_image = torch.full((3, 683, 512), 2.0)
    tall_mask = torch.full((683, 512), 7, dtype=torch.int64)

    images, masks = ade_collate([(wide_image, wide_mask), (tall_image, tall_mask)])

    assert images.shape == (2, 3, 704, 704)
    assert masks.shape == (2, 704, 704)
    assert images.shape[-2] % 32 == images.shape[-1] % 32 == 0
    assert torch.equal(images[0, :, :512, :683], wide_image)
    assert torch.equal(images[1, :, :683, :512], tall_image)
    assert torch.all(images[0, :, :, 683:] == 0)
    assert torch.all(images[1, :, 683:, :] == 0)
    assert torch.all(masks[0, :, 683:] == 255)
    assert torch.all(masks[1, 683:, :] == 255)
    assert torch.equal(masks[0, :512, :683], wide_mask)
    assert torch.equal(masks[1, :683, :512], tall_mask)


def test_segmentation_metric_ignores_void_pixels() -> None:
    metric = SegmentationMetric(num_classes=2, ignore_index=255)
    target = torch.tensor([[[0, 1, 255, 1]]])
    prediction = torch.tensor([[[0, 0, 1, 1]]])
    metric.update(prediction, target)
    assert metric.confusion_matrix.tolist() == [[1, 0], [1, 1]]
    assert metric.compute()["pixel_accuracy"] == pytest.approx(2 / 3)


def test_ade_evaluation_uses_original_gt_before_collate_padding(tmp_path: Path) -> None:
    _write_ade_fixture(tmp_path)
    image_dir = tmp_path / "ADEChallengeData2016" / "images" / "validation"
    mask_dir = tmp_path / "ADEChallengeData2016" / "annotations" / "validation"
    Image.fromarray(
        np.array([[0, 150, 255, 1], [2, 3, 4, 5], [6, 7, 8, 9]], dtype=np.uint8),
        mode="L",
    ).save(mask_dir / "sample.png")
    Image.fromarray(np.zeros((1, 4, 3), dtype=np.uint8), mode="RGB").save(image_dir / "wide.jpg")
    Image.fromarray(np.array([[0, 1, 0, 1]], dtype=np.uint8), mode="L").save(mask_dir / "wide.png")

    class SpatialModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (images.shape[0], 150, images.shape[-2], images.shape[-1]),
                -10.0,
            )
            boundary = 1000
            logits[:, 0, :, :boundary] = 10.0
            logits[:, 1, :, boundary:] = 10.0
            return logits

    loaders = build_loaders(
        "ade20k",
        tmp_path,
        physical_batch_size=2,
        workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    metrics = evaluate(SpatialModel(), loaders.validation, device="cpu", task="ade20k")
    # The two original masks contain three class-0 pixels.  A resized/padded GT
    # would incorrectly count hundreds of thousands of pixels here.
    assert metrics["support"][0] == 3
    # The first image is narrower than the boundary and the second spans it;
    # squeezing the padded canvas before cropping would change this score.
    assert metrics["pixel_accuracy"] == pytest.approx(2 / 12)


def test_coco_flip_empty_boxes_and_contiguous_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pycocotools")
    _write_coco_fixture(tmp_path)
    monkeypatch.setattr("dense_transfer.data.random.random", lambda: 0.0)
    dataset = COCODataset(tmp_path, "train", train=True)
    image, target = dataset[0]
    assert image.shape == (3, 4, 5)
    assert target["labels"].tolist() == [1]
    assert target["category_ids"].tolist() == [17]
    assert torch.allclose(target["boxes"], torch.tensor([[2.0, 1.0, 4.0, 3.0]]))
    assert target["masks"].dtype == torch.uint8
    assert int(target["masks"].sum()) == 4
    assert target["image_id"].item() == 1
    _, empty_target = COCODataset(tmp_path, "val", train=False)[0]
    assert empty_target["boxes"].shape == (0, 4)
    assert empty_target["masks"].shape == (0, 4, 5)
    images, targets = coco_collate([(image, target)])
    assert isinstance(images, list)
    assert isinstance(targets, list)


def test_official_coco_metrics_on_perfect_fixture(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools")
    _write_coco_fixture(tmp_path)
    dataset = COCODataset(tmp_path, "train", train=False)
    _, target = dataset[0]
    prediction = {
        "image_id": 1,
        "boxes": target["boxes"],
        "labels": torch.tensor([1]),
        "scores": torch.tensor([1.0]),
        "masks": target["masks"],
    }
    metrics = evaluate_coco(dataset, [prediction])
    assert metrics["bbox_AP50"] == pytest.approx(1.0)
    assert metrics["segm_AP50"] == pytest.approx(1.0)

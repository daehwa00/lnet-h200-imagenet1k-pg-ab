# Copyright (c) 2026

"""Evaluation helpers for dense-transfer segmentation and COCO tasks.

Segmentation metrics use a streaming confusion matrix and ignore ADE's 255
void label.  Detection/instance-segmentation metrics are delegated to the
official ``pycocotools`` COCO evaluator; this module only converts the native
PyTorch target format into the result format expected by that evaluator.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

type SegmentationOutput = Tensor
type Prediction = Mapping[str, Any]


def _squeeze_segmentation_target(target: Tensor) -> Tensor:
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError(f"segmentation target must be [N,H,W], got {tuple(target.shape)}")
    return target.to(dtype=torch.int64)


def _class_predictions(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.ndim == 4:
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                f"prediction/target batch mismatch: {prediction.shape[0]} vs {target.shape[0]}"
            )
        prediction = prediction.argmax(dim=1)
    elif prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    elif (
        prediction.ndim == 3
        and target.shape[0] == 1
        and prediction.shape[-2:] == target.shape[-2:]
        and prediction.shape[0] != 1
    ):
        # Also accept one-image CHW logits, which are common in direct metric
        # unit tests and in simple CPU validation loops.
        prediction = prediction.unsqueeze(0).argmax(dim=1)
    if prediction.ndim != 3:
        raise ValueError(
            "segmentation prediction must be class ids [N,H,W] or logits [N,C,H,W], "
            f"got {tuple(prediction.shape)}"
        )
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            f"segmentation prediction/target shape mismatch: {tuple(prediction.shape)} "
            f"vs {tuple(target.shape)}"
        )
    return prediction.to(dtype=torch.int64)


def update_confusion_matrix(
    confusion_matrix: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    num_classes: int | None = None,
    ignore_index: int = 255,
) -> Tensor:
    """Return ``confusion_matrix`` after accumulating one segmentation batch.

    Rows represent ground-truth classes and columns represent predicted
    classes.  The operation is out-of-place with respect to autograd (metrics
    never retain graph references), but updates the supplied tensor in place so
    it is convenient in a streaming loop.
    """

    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("confusion_matrix must be square")
    classes = int(confusion_matrix.shape[0]) if num_classes is None else int(num_classes)
    if classes != confusion_matrix.shape[0]:
        raise ValueError("num_classes must match confusion_matrix")
    target = _squeeze_segmentation_target(target.detach())
    prediction = _class_predictions(prediction.detach(), target)
    target = target.to(device=confusion_matrix.device)
    prediction = prediction.to(device=confusion_matrix.device)
    valid = (target != int(ignore_index)) & (target >= 0) & (target < classes)
    valid &= (prediction >= 0) & (prediction < classes)
    if bool(valid.any()):
        encoded = target[valid] * classes + prediction[valid]
        histogram = torch.bincount(encoded, minlength=classes * classes)
        confusion_matrix.add_(histogram.reshape(classes, classes).to(confusion_matrix.dtype))
    return confusion_matrix


def segmentation_confusion_matrix(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    *,
    ignore_index: int = 255,
    dtype: torch.dtype = torch.int64,
) -> Tensor:
    """Compute a segmentation confusion matrix for one or more images."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    matrix = torch.zeros((num_classes, num_classes), dtype=dtype)
    return update_confusion_matrix(
        matrix,
        prediction,
        target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )


def _safe_mean(values: Tensor, valid: Tensor) -> float:
    if not bool(valid.any()):
        return 0.0
    return float(values[valid].mean().item())


def compute_segmentation_metrics(
    confusion_matrix: Tensor,
    *,
    include_confusion_matrix: bool = False,
) -> dict[str, Any]:
    """Compute common semantic-segmentation metrics from a confusion matrix."""

    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("confusion_matrix must be square")
    matrix = confusion_matrix.detach().to(dtype=torch.float64)
    true_positive = torch.diag(matrix)
    ground_truth = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    union = ground_truth + predicted - true_positive
    class_accuracy = torch.where(ground_truth > 0, true_positive / ground_truth, torch.zeros_like(ground_truth))
    class_iou = torch.where(union > 0, true_positive / union, torch.zeros_like(union))
    total = matrix.sum()
    pixel_accuracy = float(true_positive.sum().item() / total.item()) if total.item() else 0.0
    valid_accuracy = ground_truth > 0
    valid_iou = union > 0
    frequency = torch.where(total > 0, ground_truth / total, torch.zeros_like(ground_truth))
    metrics: dict[str, Any] = {
        "pixel_accuracy": pixel_accuracy,
        "pixel_acc": pixel_accuracy,
        "mean_accuracy": _safe_mean(class_accuracy, valid_accuracy),
        "mean_acc": _safe_mean(class_accuracy, valid_accuracy),
        "mean_iou": _safe_mean(class_iou, valid_iou),
        "miou": _safe_mean(class_iou, valid_iou),
        "frequency_weighted_iou": float((frequency * class_iou).sum().item()),
        "fw_iou": float((frequency * class_iou).sum().item()),
        "pixAcc": pixel_accuracy,
        "mAcc": _safe_mean(class_accuracy, valid_accuracy),
        "mIoU": _safe_mean(class_iou, valid_iou),
        "per_class_accuracy": class_accuracy.tolist(),
        "per_class_iou": class_iou.tolist(),
        "support": ground_truth.to(dtype=torch.int64).tolist(),
    }
    if include_confusion_matrix:
        metrics["confusion_matrix"] = confusion_matrix.detach().clone()
    return metrics


def segmentation_metrics(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    *,
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Compute segmentation metrics directly from logits/class ids and labels."""

    matrix = segmentation_confusion_matrix(
        prediction,
        target,
        num_classes,
        ignore_index=ignore_index,
    )
    return compute_segmentation_metrics(matrix, include_confusion_matrix=True)


class SegmentationMetric:
    """Streaming confusion-matrix accumulator for ADE validation."""

    def __init__(
        self,
        num_classes: int = 150,
        *,
        ignore_index: int = 255,
        device: torch.device | str | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=device,
        )

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    def update(self, prediction: Tensor, target: Tensor) -> None:
        update_confusion_matrix(
            self.confusion_matrix,
            prediction,
            target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def merge(self, other: SegmentationMetric | Tensor) -> None:
        matrix = other.confusion_matrix if isinstance(other, SegmentationMetric) else other
        if tuple(matrix.shape) != tuple(self.confusion_matrix.shape):
            raise ValueError("cannot merge confusion matrices with different shapes")
        self.confusion_matrix.add_(matrix.to(device=self.confusion_matrix.device, dtype=torch.int64))

    def compute(self, *, include_confusion_matrix: bool = False) -> dict[str, Any]:
        return compute_segmentation_metrics(
            self.confusion_matrix,
            include_confusion_matrix=include_confusion_matrix,
        )

    @property
    def matrix(self) -> Tensor:
        """Short alias retained for metric loops written against older code."""

        return self.confusion_matrix


SegmentationConfusionMatrix = SegmentationMetric


def _load_coco_modules() -> tuple[Any, Any]:
    try:
        from pycocotools import mask as mask_utils
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "COCO metrics require the official pycocotools package; install 'pycocotools'"
        ) from exc
    return mask_utils, COCOeval


def _first_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Tensor):
        flat = value.detach().cpu().reshape(-1)
        return default if flat.numel() == 0 else flat[0].item()
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return default if flat.size == 0 else flat[0].item()
    if isinstance(value, (list, tuple)):
        return default if not value else _first_value(value[0], default)
    return value


def _to_cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().cpu()
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _prediction_image_id(prediction: Prediction, fallback: int | None = None) -> int:
    value = prediction.get("image_id", prediction.get("id", fallback))
    if value is None:
        raise ValueError("COCO prediction is missing image_id")
    return int(_first_value(value))


def _label_category_ids(
    prediction: Prediction,
    labels: Tensor,
    label_to_category_id: Mapping[int, int] | None,
) -> list[int]:
    category_values = prediction.get("category_ids", prediction.get("category_id"))
    if category_values is not None:
        return [int(value) for value in _to_cpu_tensor(category_values).reshape(-1).tolist()]
    if label_to_category_id is None:
        return [int(value) for value in labels.reshape(-1).tolist()]
    category_ids: list[int] = []
    for label in labels.reshape(-1).tolist():
        try:
            category_ids.append(int(label_to_category_id[int(label)]))
        except KeyError as exc:
            raise KeyError(f"prediction contains unknown contiguous COCO label {label}") from exc
    return category_ids


def _prediction_masks(
    value: Any,
    *,
    height: int,
    width: int,
) -> list[np.ndarray]:
    masks = _to_cpu_tensor(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError(f"COCO masks must be [N,H,W] or [N,1,H,W], got {tuple(masks.shape)}")
    if tuple(masks.shape[-2:]) != (height, width):
        masks = F.interpolate(
            masks.to(dtype=torch.float32).unsqueeze(1),
            size=(height, width),
            mode="nearest",
        )[:, 0]
    masks = masks > 0.5 if masks.dtype.is_floating_point else masks > 0
    return [mask.numpy().astype(np.uint8, copy=False) for mask in masks]


def _prediction_rows(
    prediction: Prediction,
    *,
    coco: Any,
    label_to_category_id: Mapping[int, int] | None,
    mask_utils: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert one native prediction to COCO bbox and segmentation rows."""

    image_id = _prediction_image_id(prediction)
    if image_id not in coco.imgs:
        raise KeyError(f"prediction image_id {image_id} is absent from COCO annotations")
    info = coco.imgs[image_id]
    image_width = int(info["width"])
    image_height = int(info["height"])
    boxes_value = prediction.get("boxes", prediction.get("bbox", []))
    boxes = _to_cpu_tensor(boxes_value, dtype=torch.float32)
    if boxes.numel() == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"COCO boxes must have shape [N,4], got {tuple(boxes.shape)}")
    # Native targets use XYXY.  Already-encoded result rows use XYWH and are
    # accepted when a category_id/bbox is provided instead of ``boxes``.
    native_xyxy = "boxes" in prediction
    if native_xyxy:
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0::2].clamp_(0.0, float(image_width))
        boxes_xyxy[:, 1::2].clamp_(0.0, float(image_height))
    else:
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
    scores_value = prediction.get("scores")
    if scores_value is None:
        scores = torch.ones((boxes.shape[0],), dtype=torch.float32)
    else:
        scores = _to_cpu_tensor(scores_value, dtype=torch.float32).reshape(-1)
    labels_value = prediction.get("labels")
    if labels_value is None:
        labels = _to_cpu_tensor(prediction.get("category_ids", prediction.get("category_id", [])), dtype=torch.int64).reshape(-1)
    else:
        labels = _to_cpu_tensor(labels_value, dtype=torch.int64).reshape(-1)
    category_ids = _label_category_ids(prediction, labels, label_to_category_id)
    count = min(int(boxes_xyxy.shape[0]), int(scores.shape[0]), len(category_ids))
    boxes_xyxy = boxes_xyxy[:count]
    scores = scores[:count]
    category_ids = category_ids[:count]
    bbox_rows: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    for index, (box, score, category_id) in enumerate(zip(boxes_xyxy.tolist(), scores.tolist(), category_ids, strict=True)):
        x0, y0, x1, y1 = box
        x0 = max(0.0, min(float(x0), float(image_width)))
        y0 = max(0.0, min(float(y0), float(image_height)))
        x1 = max(0.0, min(float(x1), float(image_width)))
        y1 = max(0.0, min(float(y1), float(image_height)))
        if x1 <= x0 or y1 <= y0:
            continue
        valid_indices.append(index)
        bbox_rows.append(
            {
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(score),
            }
        )
    seg_rows: list[dict[str, Any]] = []
    masks_value = prediction.get("masks", prediction.get("mask"))
    if masks_value is not None:
        masks = _prediction_masks(masks_value, height=image_height, width=image_width)
        for output_index, source_index in enumerate(valid_indices):
            if source_index >= len(masks):
                break
            encoded = mask_utils.encode(np.asfortranarray(masks[source_index]))
            if isinstance(encoded.get("counts"), bytes):
                encoded["counts"] = encoded["counts"].decode("ascii")
            row = dict(bbox_rows[output_index])
            row["segmentation"] = encoded
            seg_rows.append(row)
    return bbox_rows, seg_rows


def _normalise_predictions(
    predictions: Sequence[Prediction] | Iterable[Prediction] | Mapping[int, Prediction],
) -> list[Prediction]:
    if isinstance(predictions, Mapping):
        values: list[Prediction] = []
        for image_id, prediction in predictions.items():
            if "image_id" not in prediction and "id" not in prediction:
                values.append({**prediction, "image_id": image_id})
            else:
                values.append(prediction)
        return values
    return list(predictions)


def _coco_object(dataset_or_coco: Any) -> tuple[Any, Mapping[int, int] | None]:
    coco = getattr(dataset_or_coco, "coco", dataset_or_coco)
    if not hasattr(coco, "loadRes") or not hasattr(coco, "imgs"):
        raise TypeError("expected a COCODataset or pycocotools COCO object")
    mapping = getattr(dataset_or_coco, "label_to_category_id", None)
    return coco, mapping


def _zero_coco_metrics(prefix: str) -> dict[str, float]:
    names = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")
    return {f"{prefix}_{name}": 0.0 for name in names}


def _evaluate_coco_type(
    coco: Any,
    rows: list[dict[str, Any]],
    iou_type: Literal["bbox", "segm"],
    *,
    coco_eval_class: Any,
) -> dict[str, float]:
    prefix = iou_type
    if not rows:
        return _zero_coco_metrics(prefix)
    try:
        coco_results = coco.loadRes(rows)
    except (IndexError, KeyError, TypeError, ValueError):
        return _zero_coco_metrics(prefix)
    evaluator = coco_eval_class(coco, coco_results, iou_type)
    # COCOeval prints its twelve summary lines.  Keep library callers quiet;
    # the returned named values are more useful to an experiment logger.
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = np.asarray(evaluator.stats, dtype=np.float64).reshape(-1)
    names = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")
    result = _zero_coco_metrics(prefix)
    for name, value in zip(names, stats.tolist(), strict=False):
        result[f"{prefix}_{name}"] = float(value) if np.isfinite(value) else 0.0
    return result


def evaluate_coco(
    dataset_or_coco: Any,
    predictions: Sequence[Prediction] | Iterable[Prediction] | Mapping[int, Prediction],
    *,
    iou_types: Sequence[Literal["bbox", "segm"]] = ("bbox", "segm"),
) -> dict[str, float]:
    """Evaluate native PyTorch predictions with official COCOeval.

    The returned dictionary contains flat keys such as ``bbox_AP50`` and
    ``segm_AP``.  Empty prediction sets are valid and return zero metrics.
    """

    coco, label_to_category_id = _coco_object(dataset_or_coco)
    mask_utils, coco_eval_class = _load_coco_modules()
    bbox_rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []
    for prediction in _normalise_predictions(predictions):
        bbox, seg = _prediction_rows(
            prediction,
            coco=coco,
            label_to_category_id=label_to_category_id,
            mask_utils=mask_utils,
        )
        bbox_rows.extend(bbox)
        seg_rows.extend(seg)
    metrics: dict[str, float] = {}
    requested = tuple(dict.fromkeys(iou_types))
    if "bbox" in requested:
        metrics.update(_evaluate_coco_type(coco, bbox_rows, "bbox", coco_eval_class=coco_eval_class))
    if "segm" in requested:
        metrics.update(_evaluate_coco_type(coco, seg_rows, "segm", coco_eval_class=coco_eval_class))
    return metrics


class COCOEvaluator:
    """Accumulate encoded COCO rows, not full-resolution float mask tensors."""

    def __init__(
        self,
        dataset_or_coco: Any,
        *,
        iou_types: Sequence[Literal["bbox", "segm"]] = ("bbox", "segm"),
    ) -> None:
        self.dataset_or_coco = dataset_or_coco
        self.iou_types = tuple(iou_types)
        self.rows: dict[str, list[dict[str, Any]]] = {"bbox": [], "segm": []}

    def reset(self) -> None:
        for rows in self.rows.values():
            rows.clear()

    def update(
        self,
        predictions: Prediction | Sequence[Prediction] | Iterable[Prediction],
        *,
        image_ids: Sequence[int] | None = None,
    ) -> None:
        values = [predictions] if isinstance(predictions, Mapping) else list(predictions)
        if image_ids is not None and len(image_ids) != len(values):
            raise ValueError("image_ids length must match predictions length")
        for index, prediction in enumerate(values):
            current = prediction
            if image_ids is not None and "image_id" not in current and "id" not in current:
                current = {**current, "image_id": int(image_ids[index])}
            encoded = coco_result_rows(self.dataset_or_coco, [current])
            for kind in self.rows:
                self.rows[kind].extend(encoded[kind])

    def compute(self) -> dict[str, float]:
        coco, _ = _coco_object(self.dataset_or_coco)
        _, evaluator = _load_coco_modules()
        result: dict[str, float] = {}
        for kind in dict.fromkeys(self.iou_types):
            result.update(_evaluate_coco_type(coco, self.rows[kind], kind, coco_eval_class=evaluator))
        return result

    def evaluate(self) -> dict[str, float]:
        return self.compute()


# Compatibility aliases used by small training/evaluation drivers.
CocoEvaluator = COCOEvaluator
evaluate_coco_predictions = evaluate_coco
coco_metrics = evaluate_coco
evaluate_segmentation = segmentation_metrics
compute_confusion_matrix = segmentation_confusion_matrix


def coco_result_rows(
    dataset_or_coco: Any,
    predictions: Sequence[Prediction] | Iterable[Prediction] | Mapping[int, Prediction],
) -> dict[str, list[dict[str, Any]]]:
    """Return official COCO bbox/segm result rows without running COCOeval."""

    coco, label_to_category_id = _coco_object(dataset_or_coco)
    mask_utils, _coco_eval_class = _load_coco_modules()
    bbox_rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []
    for prediction in _normalise_predictions(predictions):
        bbox, seg = _prediction_rows(
            prediction,
            coco=coco,
            label_to_category_id=label_to_category_id,
            mask_utils=mask_utils,
        )
        bbox_rows.extend(bbox)
        seg_rows.extend(seg)
    return {"bbox": bbox_rows, "segm": seg_rows}


__all__ = [
    "COCOEvaluator",
    "CocoEvaluator",
    "SegmentationConfusionMatrix",
    "SegmentationMetric",
    "coco_metrics",
    "coco_result_rows",
    "compute_confusion_matrix",
    "compute_segmentation_metrics",
    "evaluate_coco",
    "evaluate_coco_predictions",
    "evaluate_segmentation",
    "segmentation_confusion_matrix",
    "segmentation_metrics",
    "update_confusion_matrix",
]

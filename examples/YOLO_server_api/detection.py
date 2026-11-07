from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
lazy from sahi.predict import get_sliced_prediction

from examples.YOLO_server_api.config import get_inference_device
from examples.YOLO_server_api.config import USE_SAHI
from examples.YOLO_server_api.config import USE_TENSORRT


@dataclass(frozen=True)
class DetectionPolicy:
    conflicts: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_classes(cls, classes) -> DetectionPolicy:
        ids = {item.code: item.id for item in classes}
        return cls(
            tuple(
                (ids[positive], ids[negative])
                for positive, negative in (
                    ('helmet', 'no_helmet'),
                    ('safety_vest', 'no_safety_vest'),
                )
                if positive in ids and negative in ids
            ),
        )


def get_prediction_result(img: np.ndarray, model: Any) -> Any:
    """Generate prediction results using TensorRT, SAHI, or standard YOLO
    inference.

    Args:
        img: Input image as numpy array in BGR format.
        model: Loaded YOLO model (either Ultralytics or SAHI compatible).

    Returns:
        Prediction results from the model. Format varies based on inference
        method:
        - TensorRT: Ultralytics Results object
        - SAHI: SlicedPrediction object with object_prediction_list
        - Standard: Ultralytics Results object

    Raises:
        cv2.error: If the image cannot be processed or model inference fails.

    Note:
        This function is designed to be wrapped with asyncio.to_thread for
        non-blocking execution in async contexts.
    """
    # SAHI sliced inference path for better small object detection
    if USE_SAHI:
        return get_sliced_prediction(
            img,
            model,
            slice_height=370,
            slice_width=370,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )
    else:  # Ultralytics (TensorRT or standard) inference path
        # Ultralytics returns list[Results], we only need the first result
        # for single image
        predict_kwargs = {'source': img, 'verbose': False}
        if not USE_TENSORRT:
            predict_kwargs['device'] = get_inference_device()
        return model.predict(**predict_kwargs)[0]


def compile_detection_data(result: Any) -> list[list[float | int]]:
    """Standardise detection results from SAHI and Ultralytics into uniform
    format.

    Args:
        result:
            Detection result object from either SAHI or Ultralytics inference.

    Returns:
        List of detections in format [x1, y1, x2, y2, confidence, label_id].

    Note:
        - SAHI results have 'object_prediction_list' attribute
        - Ultralytics results have 'boxes' attribute
    """
    datas: list[list[float | int]] = []

    # Handle SAHI prediction results (EAFP)
    try:
        opl = result.object_prediction_list  # may raise AttributeError
        for obj in opl:
            label = int(obj.category.id)
            x1, y1, x2, y2 = (int(x) for x in obj.bbox.to_voc_bbox())
            conf = float(obj.score.value)
            datas.append([x1, y1, x2, y2, conf, label])
        return datas
    except AttributeError:
        pass

    # Handle Ultralytics prediction results. Convert tensors once instead of
    # calling .tolist() / .item() for every detection.
    return _compile_ultralytics_detection_data(result.boxes)


def _compile_ultralytics_detection_data(
    boxes: Any,
) -> list[list[float | int]]:
    """Convert Ultralytics boxes to the shared detection row format."""
    xyxy = boxes.xyxy.cpu().numpy()
    if xyxy.size == 0:
        return []

    xyxy = xyxy.reshape(1, 4) if xyxy.ndim == 1 else xyxy
    conf = boxes.conf.cpu().numpy().reshape(-1)
    labels = boxes.cls.cpu().numpy().reshape(-1)

    return [
        [
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            float(score),
            int(label),
        ]
        for (x1, y1, x2, y2), score, label in zip(
            xyxy,
            conf,
            labels,
            strict=False,
        )
    ]


def process_labels(
    datas: list[list[float | int]],
    policy: DetectionPolicy = DetectionPolicy(),
) -> list[list[float | int]]:
    """Resolve only semantic conflicts declared by the selected model."""
    datas = remove_overlapping_labels(datas, policy)
    return remove_completely_contained_labels(datas, policy)


def run_detection(
    img: np.ndarray,
    model_instance: Any,
    policy: DetectionPolicy = DetectionPolicy(),
) -> tuple[list[list[float | int]], dict[str, float]]:
    """Infer and filter a validated frame within its model lease."""

    inference_start = time.monotonic()
    result = get_prediction_result(img, model_instance)
    inference_time = time.monotonic() - inference_start

    post_start = time.monotonic()
    datas = compile_detection_data(result)
    datas = process_labels(datas, policy)
    post_time = time.monotonic() - post_start

    return datas, {'inference': inference_time, 'post': post_time}


def _add_overlaps(
    bad: set[int],
    i1: int,
    idxs2: list[int],
    datas: list[list[float | int]],
    thr: float = 0.5,
) -> None:
    """Add indices whose boxes overlap a reference above threshold."""
    d1 = datas[i1]
    area1 = area(d1[0], d1[1], d1[2], d1[3])
    for i2 in idxs2:
        if _overlap_ratio_with_area(d1, datas[i2], area1) > thr:
            bad.add(i2)


def _add_contained(
    bad: set[int],
    i1: int,
    idxs2: list[int],
    datas: list[list[float | int]],
) -> None:
    """Add indices for boxes contained by their conflicting pair."""
    d1 = datas[i1]
    for i2 in idxs2:
        d2 = datas[i2]
        # Check if candidate detection is contained within reference
        if contained(d2, d1):
            bad.add(i2)
        # Check if reference detection is contained within candidate
        elif contained(d1, d2):
            bad.add(i1)


def _add_conflicting_pair_indices(
    bad: set[int],
    idxs1: list[int],
    idxs2: list[int],
    datas: list[list[float | int]],
    add_fn: Callable[
        [set[int], int, list[int], list[list[float | int]]],
        None,
    ],
) -> None:
    """Add conflicting detection indices between two label groups."""
    for idx1 in idxs1:
        add_fn(bad, idx1, idxs2, datas)


def remove_overlapping_labels(
    datas: list[list[float | int]],
    policy: DetectionPolicy = DetectionPolicy(),
) -> list[list[float | int]]:
    """Remove overlapping negative labels only for declared conflicts."""
    bad: set[int] = set()
    for positive, negative in policy.conflicts:
        positives = [
            index for index, row in enumerate(datas) if int(row[5]) == positive
        ]
        negatives = [
            index for index, row in enumerate(datas) if int(row[5]) == negative
        ]
        _add_conflicting_pair_indices(
            bad,
            positives,
            negatives,
            datas,
            _add_overlaps,
        )

    return _without_indices(datas, bad)


def _overlap_ratio_with_area(
    b1: Sequence[float | int],
    b2: Sequence[float | int],
    area1: int,
) -> float:
    """Calculate IoU when the first box area is already known."""
    x1, y1, x2, y2 = (
        max(b1[0], b2[0]),  # Left edge of intersection
        max(b1[1], b2[1]),  # Top edge of intersection
        min(b1[2], b2[2]),  # Right edge of intersection
        min(b1[3], b2[3]),  # Bottom edge of intersection
    )
    if x2 < x1 or y2 < y1:
        return 0.0

    # Calculate intersection area
    inter = area(x1, y1, x2, y2)

    # Calculate union area (area of both boxes minus intersection)
    union_area = area1 + area(b2[0], b2[1], b2[2], b2[3]) - inter

    # Return IoU ratio (handle division by zero)
    return inter / float(union_area) if union_area > 0 else 0.0


def area(
    x1: float | int,
    y1: float | int,
    x2: float | int,
    y2: float | int,
) -> int:
    """Calculate the area of a bounding box defined by corner coordinates.

    Computes the area of a rectangular bounding box, handling edge cases where
    the coordinates may result in invalid (negative) dimensions.

    Args:
        x1: Left coordinate of the bounding box.
        y1: Top coordinate of the bounding box.
        x2: Right coordinate of the bounding box.
        y2: Bottom coordinate of the bounding box.

    Returns:
        Area as integer. Returns 0 for invalid bounding boxes.

    Note:
        Uses max(0, dimension + 1) to handle pixel-perfect area calculation
        and prevent negative areas from invalid coordinates.
    """
    # Calculate width and height, ensuring non-negative values
    width = max(0, x2 - x1 + 1)
    height = max(0, y2 - y1 + 1)
    # Ensure the return type is explicitly an integer
    return int(width * height)


def remove_completely_contained_labels(
    datas: list[list[float | int]],
    policy: DetectionPolicy = DetectionPolicy(),
) -> list[list[float | int]]:
    """Resolve contained boxes only for declared conflicting classes."""
    bad: set[int] = set()
    for positive, negative in policy.conflicts:
        positives = [
            index for index, row in enumerate(datas) if int(row[5]) == positive
        ]
        negatives = [
            index for index, row in enumerate(datas) if int(row[5]) == negative
        ]
        _add_conflicting_pair_indices(
            bad,
            positives,
            negatives,
            datas,
            _add_contained,
        )

    return _without_indices(datas, bad)


def contained(
    inner: Sequence[float | int],
    outer: Sequence[float | int],
) -> bool:
    """Check if one bounding box is completely contained within another.

    Determines whether the inner bounding box is entirely enclosed by the outer
    bounding box by comparing all four corner coordinates.

    Args:
        inner: Inner bounding box as [x1, y1, x2, y2].
        outer: Outer bounding box as [x1, y1, x2, y2].

    Returns:
        True if inner box is completely contained within outer box,
            False otherwise.

    Note:
        Uses inclusive comparison (<=, >=) to handle edge cases where boxes
        share boundary coordinates.
    """
    return (
        inner[0] >= outer[0]  # Inner left >= outer left
        and inner[1] >= outer[1]  # Inner top >= outer top
        and inner[2] <= outer[2]  # Inner right <= outer right
        and inner[3] <= outer[3]  # Inner bottom <= outer bottom
    )


def _without_indices(
    datas: list[list[float | int]],
    bad: set[int],
) -> list[list[float | int]]:
    """Return detections excluding bad indices without repeated list shifts."""
    if not bad:
        return datas
    return [detection for i, detection in enumerate(datas) if i not in bad]

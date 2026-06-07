from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from sahi.predict import get_sliced_prediction

from examples.YOLO_server_api.backend.config import get_inference_device
from examples.YOLO_server_api.backend.config import USE_SAHI
from examples.YOLO_server_api.backend.config import USE_TENSORRT


def convert_to_image(data: bytes) -> np.ndarray:
    """Convert raw bytes data to OpenCV BGR image array.

    Args:
        data: Raw image bytes data to be decoded.

    Returns:
        Decoded image as OpenCV BGR numpy array.

    Raises:
        cv2.error: If the image data cannot be decoded.
    """
    # Convert bytes to numpy array for image decoding
    npimg = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image is None:
        raise cv2.error('Unable to decode image bytes')
    return image


async def get_prediction_result(img: np.ndarray, model: Any) -> Any:
    """
    Generate prediction results using TensorRT, SAHI, or standard YOLO
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
    """
    Standardise detection results from SAHI and Ultralytics
    into uniform format.

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


async def process_labels(
    datas: list[list[float | int]],
) -> list[list[float | int]]:
    """
    Process detection labels by removing overlapping and contained detections.

    Applies a multi-stage filtering process to clean up detection results:
    1. Remove overlapping labels (e.g., hardhat vs no_hardhat conflicts)
    2. Remove completely contained labels (nested detections)
    3. Re-run overlap removal to catch any new conflicts

    Args:
        datas: List of detection data in format [x1, y1, x2, y2, conf, label].

    Returns:
        Cleaned detection data with conflicts resolved.

    Note:
        The double overlap removal ensures that removing contained labels
        doesn't create new overlapping conflicts.
    """
    datas = await remove_overlapping_labels(datas)
    return await remove_completely_contained_labels(datas)


# Global semaphore for controlling concurrency in inference
INFERENCE_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(4)


async def run_detection_from_bytes(
    img_bytes: bytes,
    model_instance: Any,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[list[list[float | int]], dict[str, float]]:
    """
    Run detection on image bytes with concurrency control.

    Args:
        img_bytes: The raw image bytes.
        model_instance: The loaded model instance.
        semaphore:
            The concurrency limit (optional).
            If not provided, use INFERENCE_SEMAPHORE.

        Returns:
                A tuple containing:
                - A list of detection results in the format
                    [x1, y1, x2, y2, conf, label].
                - A dictionary with timing information for 'inference'
                    and 'post' processing.
    """
    img = convert_to_image(img_bytes)

    # Concurrency limit: use HTTP's limit by default, WS can pass its own
    sem = semaphore or INFERENCE_SEMAPHORE

    inference_start = time.time()
    async with sem:
        result = await get_prediction_result(img, model_instance)
    inference_time = time.time() - inference_start

    post_start = time.time()
    datas = compile_detection_data(result)
    datas = await process_labels(datas)
    post_time = time.time() - post_start

    return datas, {'inference': inference_time, 'post': post_time}


def get_category_indices(
    datas: list[list[float | int]],
) -> dict[str, list[int]]:
    """Generate category indices for safety equipment detection filtering.

    Creates index mappings for different safety equipment categories to enable
    efficient conflict resolution between positive and negative detections.

    Args:
        datas: List of detection data in format [x1, y1, x2, y2, conf, label].

    Returns:
        Dictionary mapping category names to lists of detection indices:
        - 'hardhat': Indices of hard hat detections (label 0)
        - 'no_hardhat': Indices of no hard hat detections (label 2)
        - 'safety_vest': Indices of safety vest detections (label 7)
        - 'no_safety_vest': Indices of no safety vest detections (label 4)

    Note:
        Label IDs are based on the trained model's class definitions.
    """
    indices: dict[str, list[int]] = {
        'hardhat': [],
        'no_hardhat': [],
        'safety_vest': [],
        'no_safety_vest': [],
    }
    label_to_key = {
        0: 'hardhat',
        2: 'no_hardhat',
        7: 'safety_vest',
        4: 'no_safety_vest',
    }
    for i, detection in enumerate(datas):
        key = label_to_key.get(int(detection[5]))
        if key is not None:
            indices[key].append(i)
    return indices


async def _calc_and_filter(
    idxs1: list[int],
    idxs2: list[int],
    datas: list[list[float | int]],
    fn: Callable,
) -> set[int]:
    """Apply filtering function to calculate conflicting detection indices.

    Helper function that applies a conflict detection function between two sets
    of detection indices and accumulates the results.

    Args:
        idxs1: Indices of the first detection category.
        idxs2: Indices of the second detection category.
        datas: Complete detection data list.
        fn: Filtering function to apply (e.g., find_overlaps, find_contained).

    Returns:
        Set of detection indices that should be removed due to conflicts.

    Note:
        This function enables efficient batch processing of conflict detection
        between different category pairs.
    """
    bad: set[int] = set()
    for idx1 in idxs1:
        bad.update(await fn(idx1, idxs2, datas))
    return bad


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


async def remove_overlapping_labels(
    datas: list[list[float | int]],
) -> list[list[float | int]]:
    """
    Remove overlapping detections
    between conflicting safety equipment categories.

    Args:
        datas: List of detection data in format [x1, y1, x2, y2, conf, label].

    Returns:
        Filtered detection data with overlapping conflicts removed.

    Note:
        Uses intersection over union (IoU) threshold to determine overlaps.
        Memory cleanup with gc.collect() is performed after removal operations.
    """
    ci = get_category_indices(datas)
    bad: set[int] = set()

    # Find overlaps between hardhat and no_hardhat detections
    _add_conflicting_pair_indices(
        bad, ci['hardhat'], ci['no_hardhat'], datas, _add_overlaps,
    )
    # Find overlaps between safety_vest and no_safety_vest detections
    _add_conflicting_pair_indices(
        bad,
        ci['safety_vest'],
        ci['no_safety_vest'],
        datas,
        _add_overlaps,
    )

    return _without_indices(datas, bad)


async def find_overlaps(
    i1: int,
    idxs2: list[int],
    datas: list[list[float | int]],
    thr: float = 0.5,
) -> set[int]:
    """Find detections that overlap with a reference detection above threshold.

    Compares a reference detection against a list of candidate detections to
    identify those with overlap ratios exceeding the specified threshold.

    Args:
        i1: Index of the reference detection.
        idxs2: List of candidate detection indices to compare against.
        datas: Complete detection data list.
        thr: Overlap ratio threshold (default 0.5, meaning 50% overlap).

    Returns:
        Set of detection indices that overlap significantly with the reference.

    Note:
        Uses intersection over union (IoU) calculation
            for overlap determination.
    """
    bad: set[int] = set()
    _add_overlaps(bad, i1, idxs2, datas, thr)
    return bad


def overlap_ratio(
    b1: Sequence[float | int],
    b2: Sequence[float | int],
) -> float:
    """
    Calculate intersection over union (IoU) ratio between two bounding boxes.

    Args:
        b1: First bounding box as [x1, y1, x2, y2].
        b2: Second bounding box as [x1, y1, x2, y2].

    Returns:
        IoU ratio as float between 0.0 (no overlap) and 1.0 (complete overlap).

    Note:
        Uses the intersection area divided by union area formula.
        Handles edge cases where boxes don't overlap (returns 0.0).
    """
    return _overlap_ratio_values(b1, b2)


def _overlap_ratio_values(
    b1: Sequence[float | int],
    b2: Sequence[float | int],
) -> float:
    """Calculate IoU without allocating sliced box lists."""
    return _overlap_ratio_with_area(
        b1,
        b2,
        area(b1[0], b1[1], b1[2], b1[3]),
    )


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
    union_area = (
        area1
        + area(b2[0], b2[1], b2[2], b2[3])
        - inter
    )

    # Return IoU ratio (handle division by zero)
    return inter / float(union_area) if union_area > 0 else 0.0


def area(
    x1: float | int, y1: float | int,
    x2: float | int, y2: float | int,
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


async def remove_completely_contained_labels(
    datas: list[list[float | int]],
) -> list[list[float | int]]:
    """
    Remove detections that are completely contained within other detections.

    Args:
        datas: List of detection data in format [x1, y1, x2, y2, conf, label].

    Returns:
        Filtered detection data with contained detections removed.

    Note:
        Processes conflicting categories (hardhat vs no_hardhat, safety_vest vs
        no_safety_vest) to resolve containment conflicts between positive and
        negative detections.
    """
    ci = get_category_indices(datas)
    bad: set[int] = set()

    # Find contained detections between hardhat and no_hardhat categories
    _add_conflicting_pair_indices(
        bad, ci['hardhat'], ci['no_hardhat'], datas, _add_contained,
    )
    # Find contained detections between safety_vest
    # and no_safety_vest categories
    _add_conflicting_pair_indices(
        bad,
        ci['safety_vest'],
        ci['no_safety_vest'],
        datas,
        _add_contained,
    )

    return _without_indices(datas, bad)


async def find_contained(
    i1: int,
    idxs2: list[int],
    datas: list[list[float | int]],
) -> set[int]:
    """
    Find detections that have containment relationships
    with a reference detection.

    Args:
        i1: Index of the reference detection.
        idxs2: List of candidate detection indices to compare against.
        datas: Complete detection data list.

    Returns:
        Set of detection indices that have containment relationships (either
        direction) with the reference detection.

    Note:
        Checks both directions: reference contained in candidate, and candidate
        contained in reference, to identify all containment conflicts.
    """
    bad: set[int] = set()
    _add_contained(bad, i1, idxs2, datas)
    return bad


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
    return [
        detection
        for i, detection in enumerate(datas)
        if i not in bad
    ]

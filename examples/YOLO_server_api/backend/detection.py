from __future__ import annotations

import gc
from typing import Any
from typing import Callable

import cv2
import numpy as np
from sahi.predict import get_sliced_prediction

from examples.YOLO_server_api.backend.config import USE_SAHI


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
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)


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
        return model.predict(source=img, verbose=False)[0]


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

    # Handle SAHI prediction results
    if hasattr(result, 'object_prediction_list'):
        for obj in result.object_prediction_list:
            label = int(obj.category.id)
            x1, y1, x2, y2 = (int(x) for x in obj.bbox.to_voc_bbox())
            conf = float(obj.score.value)
            datas.append([x1, y1, x2, y2, conf, label])
        return datas

    # Handle Ultralytics prediction results
    boxes = result.boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        conf = float(boxes.conf[i].item())
        label = int(boxes.cls[i].item())
        datas.append([x1, y1, x2, y2, conf, label])
    return datas


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
    # First pass: remove overlapping detections
    datas = await remove_overlapping_labels(datas)
    # Remove completely contained detections
    datas = await remove_completely_contained_labels(datas)
    # Final pass: clean up any remaining overlaps
    datas = await remove_overlapping_labels(datas)
    return datas


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
    return {
        'hardhat': [i for i, d in enumerate(datas) if d[5] == 0],
        'no_hardhat': [i for i, d in enumerate(datas) if d[5] == 2],
        'safety_vest': [i for i, d in enumerate(datas) if d[5] == 7],
        'no_safety_vest': [i for i, d in enumerate(datas) if d[5] == 4],
    }


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
        bad |= await fn(idx1, idxs2, datas)
    return bad


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
    bad = set()

    # Find overlaps between hardhat and no_hardhat detections
    bad |= await _calc_and_filter(
        ci['hardhat'], ci['no_hardhat'], datas, find_overlaps,
    )
    # Find overlaps between safety_vest and no_safety_vest detections
    bad |= await _calc_and_filter(
        ci['safety_vest'], ci['no_safety_vest'], datas, find_overlaps,
    )

    # Remove conflicting detections in reverse order to maintain indices
    for i in sorted(bad, reverse=True):
        datas.pop(i)

    # Force garbage collection to free memory from removed detections
    gc.collect()
    return datas


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
    return {
        i2
        for i2 in idxs2
        if overlap_ratio(datas[i1][:4], datas[i2][:4]) > thr
    }


def overlap_ratio(
    b1: list[float | int],
    b2: list[float | int],
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
    # Calculate intersection boundaries
    x1, y1, x2, y2 = (
        max(b1[0], b2[0]),  # Left edge of intersection
        max(b1[1], b2[1]),  # Top edge of intersection
        min(b1[2], b2[2]),  # Right edge of intersection
        min(b1[3], b2[3]),  # Bottom edge of intersection
    )

    # Calculate intersection area
    inter = area(x1, y1, x2, y2)

    # Calculate union area (area of both boxes minus intersection)
    union_area = area(*b1) + area(*b2) - inter

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
    bad = set()

    # Find contained detections between hardhat and no_hardhat categories
    bad |= await _calc_and_filter(
        ci['hardhat'], ci['no_hardhat'], datas, find_contained,
    )
    # Find contained detections between safety_vest
    # and no_safety_vest categories
    bad |= await _calc_and_filter(
        ci['safety_vest'], ci['no_safety_vest'], datas, find_contained,
    )

    # Remove contained detections in reverse order to maintain indices
    for i in sorted(bad, reverse=True):
        datas.pop(i)
    return datas


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
    res = set()
    for i2 in idxs2:
        # Check if candidate detection is contained within reference
        if contained(datas[i2][:4], datas[i1][:4]):
            res.add(i2)
        # Check if reference detection is contained within candidate
        elif contained(datas[i1][:4], datas[i2][:4]):
            res.add(i1)
    return res


def contained(
    inner: list[float | int],
    outer: list[float | int],
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

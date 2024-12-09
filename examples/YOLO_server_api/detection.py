from __future__ import annotations

import gc
from typing import Any

import cv2
import numpy as np
from sahi.predict import get_sliced_prediction

from .models import DetectionModelManager

model_loader = DetectionModelManager()


async def convert_to_image(data: bytes) -> np.ndarray:
    """
    Converts raw image bytes into an OpenCV image format.

    Args:
        data (bytes): The image data in bytes.

    Returns:
        np.ndarray: The decoded image in OpenCV format.
    """
    # Convert image bytes to a NumPy array
    npimg = np.frombuffer(data, np.uint8)
    # Decode the NumPy array into an OpenCV image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img


async def get_prediction_result(
    img: np.ndarray,
    model: DetectionModelManager,
) -> Any:
    """
    Generates sliced predictions for an image using the specified model.

    Args:
        img (np.ndarray): The image in OpenCV format.
        model (DetectionModelManager): The object detection model instance.

    Returns:
        Any: The prediction result from the model.
    """
    # Use the SAHI library's get_sliced_prediction function for detection
    return get_sliced_prediction(
        img,
        model,
        slice_height=370,
        slice_width=370,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
    )


def compile_detection_data(result: Any) -> list[list[float | int]]:
    """
    Compiles detection data from the model's prediction results.

    Args:
        result (Any): The result of the model prediction.

    Returns:
        list[list[float | int]]: Detection data including bounding boxes,
        confidence, and label IDs.
    """
    datas = []
    # Extract bounding box, confidence, and label ID for each prediction
    for object_prediction in result.object_prediction_list:
        label = int(object_prediction.category.id)
        x1, y1, x2, y2 = (int(x) for x in object_prediction.bbox.to_voc_bbox())
        confidence = float(object_prediction.score.value)
        datas.append([x1, y1, x2, y2, confidence, label])
    return datas


async def process_labels(
    datas: list[list[float | int]],
) -> list[list[float | int]]:
    """
    Processes detection data to remove overlapping and contained labels.

    Args:
        datas (list[list[float | int]]): The detection data to process.

    Returns:
        list[list[float | int]]: The processed detection data.
    """
    # Remove overlapping and completely contained labels
    datas = await remove_overlapping_labels(datas)
    datas = await remove_completely_contained_labels(datas)
    datas = await remove_overlapping_labels(datas)
    return datas


async def remove_overlapping_labels(
    datas: list[list[float | int]],
) -> list[list[float | int]]:
    """
    Removes overlapping labels based on predefined thresholds.

    Args:
        datas (list[list[float | int]]): The detection data to process.

    Returns:
        list[list[float | int]]: The detection data
        with overlapping labels removed.
    """
    # Organise data by category indices for efficient processing
    category_indices = get_category_indices(datas)

    # Identify overlapping labels to remove
    to_remove = set()
    to_remove.update(
        await find_overlaps(
            category_indices['hardhat'],
            category_indices['no_hardhat'],
            datas,
            0.8,
        ),
    )
    to_remove.update(
        await find_overlaps(
            category_indices['safety_vest'],
            category_indices['no_safety_vest'],
            datas,
            0.8,
        ),
    )

    # Remove overlapping entries from the data list
    for index in sorted(to_remove, reverse=True):
        datas.pop(index)

    # Run garbage collection to free memory
    gc.collect()
    return datas


def get_category_indices(
    datas: list[list[float | int]],
) -> dict[str, list[int]]:
    """
    Organises detection data by category indices for quicker access.

    Args:
        datas (list[list[float | int]]): The detection data.

    Returns:
        dict[str, list[int]]:
            A dictionary mapping category names to lists of indices.
    """
    # Create a dictionary with lists of indices for each category
    return {
        'hardhat': [i for i, d in enumerate(datas) if d[5] == 0],
        'no_hardhat': [i for i, d in enumerate(datas) if d[5] == 2],
        'safety_vest': [i for i, d in enumerate(datas) if d[5] == 7],
        'no_safety_vest': [i for i, d in enumerate(datas) if d[5] == 4],
    }


async def find_overlaps(
    indices1: list[int],
    indices2: list[int],
    datas: list[list[float | int]],
    threshold: float,
) -> set[int]:
    """
    Finds overlaps between two sets of category indices.

    Args:
        indices1 (list[int]): The first set of indices.
        indices2 (list[int]): The second set of indices.
        datas (list[list[float | int]]): The detection data.
        threshold (float): The overlap threshold.

    Returns:
        set[int]: Indices of overlapping labels to be removed.
    """
    to_remove = set()
    # Check for overlaps between two sets of indices
    for index1 in indices1:
        to_remove.update(
            await find_overlapping_indices(
                index1, indices2, datas, threshold,
            ),
        )
    return to_remove


async def find_overlapping_indices(
    index1: int,
    indices2: list[int],
    datas: list[list[float | int]], threshold: float,
) -> set[int]:
    """
    Finds overlapping indices for a single detection index.

    Args:
        index1 (int): The index of the first detection.
        indices2 (list[int]): The indices of potential overlapping detections.
        datas (list[list[float | int]]): The detection data.
        threshold (float): The overlap threshold.

    Returns:
        set[int]: Indices of overlapping labels.
    """
    # Calculate overlaps for a specific detection
    # and return those exceeding the threshold
    return {
        index2 for index2 in indices2
        if calculate_overlap(
            [int(x) for x in datas[index1][:4]],
            [int(x) for x in datas[index2][:4]],
        ) > threshold
    }


def calculate_overlap(bbox1: list[int], bbox2: list[int]) -> float:
    """
    Calculates the overlap between two bounding boxes.

    Args:
        bbox1 (list[int]): The first bounding box.
        bbox2 (list[int]): The second bounding box.

    Returns:
        float: The overlap percentage.
    """
    # Calculate intersection coordinates and area
    x1, y1, x2, y2 = calculate_intersection(bbox1, bbox2)
    intersection_area = calculate_area(x1, y1, x2, y2)
    # Calculate area of each bounding box
    bbox1_area = calculate_area(*bbox1)
    bbox2_area = calculate_area(*bbox2)

    # Calculate the overlap percentage
    overlap_percentage = intersection_area / \
        float(bbox1_area + bbox2_area - intersection_area)
    gc.collect()
    return overlap_percentage


def calculate_intersection(
    bbox1: list[int],
    bbox2: list[int],
) -> tuple[int, int, int, int]:
    """
    Calculates the intersection coordinates of two bounding boxes.

    Args:
        bbox1 (list[int]): The first bounding box.
        bbox2 (list[int]): The second bounding box.

    Returns:
        tuple[int, int, int, int]: The intersection coordinates.
    """
    # Determine the coordinates for the intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    return x1, y1, x2, y2


def calculate_area(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Calculates the area of a bounding box.

    Args:
        x1 (int): The x-coordinate of the top-left corner.
        y1 (int): The y-coordinate of the top-left corner.
        x2 (int): The x-coordinate of the bottom-right corner.
        y2 (int): The y-coordinate of the bottom-right corner.

    Returns:
        int: The area of the bounding box.
    """
    # Calculate area, ensuring no negative dimensions
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def is_contained(inner_bbox: list[int], outer_bbox: list[int]) -> bool:
    """
    Checks if one bounding box is fully contained within another.

    Args:
        inner_bbox (list[int]): The inner bounding box.
        outer_bbox (list[int]): The outer bounding box.

    Returns:
        bool: True if the inner bounding box is fully contained in
        the outer bounding box.
    """
    # Verify if each coordinate of inner_bbox is within outer_bbox
    return (
        inner_bbox[0] >= outer_bbox[0]
        and inner_bbox[2] <= outer_bbox[2]
        and inner_bbox[1] >= outer_bbox[1]
        and inner_bbox[3] <= outer_bbox[3]
    )


async def remove_completely_contained_labels(
    datas: list[list[float | int]],
) -> list[list[float | int]]:
    """
    Removes labels that are fully contained within other labels.

    Args:
        datas (list[list[float | int]]): The detection data.

    Returns:
        list[list[float | int]]: The detection data
        with contained labels removed.
    """
    # Get indices of each category for comparison
    category_indices = get_category_indices(datas)

    # Identify labels that are contained within others
    to_remove = set()
    to_remove.update(
        await find_contained_labels(
            category_indices['hardhat'],
            category_indices['no_hardhat'],
            datas,
        ),
    )
    to_remove.update(
        await find_contained_labels(
            category_indices['safety_vest'],
            category_indices['no_safety_vest'],
            datas,
        ),
    )

    # Remove fully contained labels
    for index in sorted(to_remove, reverse=True):
        datas.pop(index)

    return datas


async def find_contained_labels(
    indices1: list[int],
    indices2: list[int],
    datas: list[list[float | int]],
) -> set[int]:
    """
    Finds labels that are fully contained within other labels.

    Args:
        indices1 (list[int]): The indices of the first category.
        indices2 (list[int]): The indices of the second category.
        datas (list[list[float | int]]): The detection data.

    Returns:
        set[int]: Indices of contained labels to be removed.
    """
    to_remove = set()
    # Check for containment of labels across two sets of indices
    for index1 in indices1:
        to_remove.update(await find_contained_indices(index1, indices2, datas))
    return to_remove


async def find_contained_indices(
    index1: int,
    indices2: list[int],
    datas: list[list[float | int]],
) -> set[int]:
    """
    Finds indices of detections that are fully contained within others.

    Args:
        index1 (int): The index of the first detection.
        indices2 (list[int]): The indices of potential containing detections.
        datas (list[list[float | int]]): The detection data.

    Returns:
        set[int]: Indices of contained labels.
    """
    to_remove = set()
    # Determine if one detection is fully contained within another
    for index2 in indices2:
        to_remove.update(await check_containment(index1, index2, datas))
    return to_remove


async def check_containment(
    index1: int,
    index2: int,
    datas: list[list[float | int]],
) -> set[int]:
    """
    Checks if one detection is fully contained within another.

    Args:
        index1 (int): The index of the first detection.
        index2 (int): The index of the second detection.
        datas (list[list[float | int]]): The detection data.

    Returns:
        set[int]: Indices of contained labels.
    """
    to_remove = set()
    # Verify if either index1 or index2 is contained within the other
    if is_contained(
        [int(x) for x in datas[index2][:4]],
        [int(x) for x in datas[index1][:4]],
    ):
        to_remove.add(index2)
    elif is_contained(
        [int(x) for x in datas[index1][:4]],
        [int(x) for x in datas[index2][:4]],
    ):
        to_remove.add(index1)
    return to_remove

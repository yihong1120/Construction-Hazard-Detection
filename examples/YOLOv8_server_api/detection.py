from __future__ import annotations

import gc

import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sahi.predict import get_sliced_prediction

from .models import DetectionModelManager

detection_blueprint = Blueprint('detection', __name__)
limiter = Limiter(key_func=get_remote_address)
model_loader = DetectionModelManager()  # 创建 DetectionModelManager 实例


@detection_blueprint.route('/detect', methods=['POST'])
@jwt_required()
@limiter.limit('3000 per minute')
def detect():
    data = request.files['image'].read()
    model_key = request.args.get('model', default='yolov8n', type=str)
    model = model_loader.get_model(model_key)  # 从 DetectionModelManager 获取模型

    img = convert_to_image(data)
    result = get_prediction_result(img, model)

    datas = compile_detection_data(result)
    datas = process_labels(datas)

    return jsonify(datas)


def convert_to_image(data):
    """
    Convert string data to an image.

    Args:
        data (bytes): Image data in bytes.

    Returns:
        numpy.ndarray: Decoded image.
    """
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img


def get_prediction_result(img, model):
    """
    Get the prediction result from the model.

    Args:
        img (numpy.ndarray): Input image.
        model: Detection model.

    Returns:
        Result: Prediction result.
    """
    return get_sliced_prediction(
        img,
        model,
        slice_height=376,
        slice_width=376,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
    )


def compile_detection_data(result):
    """
    Compile detection data in YOLOv8 format.

    Args:
        result: Prediction result.

    Returns:
        list: Compiled detection data.
    """
    datas = []
    for object_prediction in result.object_prediction_list:
        label = int(object_prediction.category.id)
        x1, y1, x2, y2 = (int(x) for x in object_prediction.bbox.to_voc_bbox())
        confidence = float(object_prediction.score.value)
        datas.append([x1, y1, x2, y2, confidence, label])
    return datas


def process_labels(datas):
    """
    Process detection labels to remove overlaps and contained labels.

    Args:
        datas (list): Detection data.

    Returns:
        list: Processed detection data.
    """
    datas = remove_overlapping_labels(datas)
    datas = remove_completely_contained_labels(datas)
    datas = remove_overlapping_labels(datas)
    return datas


def remove_overlapping_labels(datas):
    """
    Removes overlapping labels for Hardhat and Safety Vest categories.

    Args:
        datas (list): A list of detection data in YOLOv8 format.

    Returns:
        list: A list of detection data with overlapping labels removed.
    """
    hardhat_indices, no_hardhat_indices, safety_vest_indices, no_safety_vest_indices = get_category_indices(datas)

    to_remove = set()
    to_remove.update(find_overlaps(hardhat_indices, no_hardhat_indices, datas, 0.8))
    to_remove.update(find_overlaps(safety_vest_indices, no_safety_vest_indices, datas, 0.8))

    for index in sorted(to_remove, reverse=True):
        datas.pop(index)

    gc.collect()
    return datas


def get_category_indices(datas):
    """
    Get indices of different categories in the detection data.

    Args:
        datas (list): A list of detection data in YOLOv8 format.

    Returns:
        tuple: Indices of Hardhat, NO-Hardhat, Safety Vest, and NO-Safety Vest detections.
    """
    hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 0]
    no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2]
    safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7]
    no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4]
    return hardhat_indices, no_hardhat_indices, safety_vest_indices, no_safety_vest_indices


def find_overlaps(indices1, indices2, datas, threshold):
    """
    Find overlapping labels between two sets of indices.

    Args:
        indices1 (list): First set of indices.
        indices2 (list): Second set of indices.
        datas (list): Detection data.
        threshold (float): Overlap threshold.

    Returns:
        set: Indices of overlapping labels to remove.
    """
    to_remove = set()
    for index1 in indices1:
        for index2 in indices2:
            if overlap_percentage(datas[index1][:4], datas[index2][:4]) > threshold:
                to_remove.add(index2)
    return to_remove


def overlap_percentage(bbox1, bbox2):
    """
    Calculates the percentage of overlap between two bounding boxes.

    Args:
        bbox1 (list): The first bounding box [x1, y1, x2, y2].
        bbox2 (list): The second bounding box [x1, y1, x2, y2].

    Returns:
        float: The percentage of overlap between the two bounding boxes.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    overlap_percentage = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    gc.collect()
    return overlap_percentage


def is_contained(inner_bbox, outer_bbox):
    """
    Determines if one bounding box is completely contained within another.

    Args:
        inner_bbox (list): The inner bounding box [x1, y1, x2, y2].
        outer_bbox (list): The outer bounding box [x1, y1, x2, y2].

    Returns:
        bool: True if inner box is fully within outer box, False otherwise.
    """
    return (
        inner_bbox[0] >= outer_bbox[0]
        and inner_bbox[2] <= outer_bbox[2]
        and inner_bbox[1] >= outer_bbox[1]
        and inner_bbox[3] <= outer_bbox[3]
    )


def remove_completely_contained_labels(datas):
    """
    Removes completely contained labels for Hardhat and Safety Vest categories.

    Args:
        datas (list): A list of detection data in YOLOv8 format.

    Returns:
        list: Detection data with fully contained labels removed.
    """
    hardhat_indices, no_hardhat_indices, safety_vest_indices, no_safety_vest_indices = get_category_indices(datas)

    to_remove = set()
    to_remove.update(find_contained_labels(hardhat_indices, no_hardhat_indices, datas))
    to_remove.update(find_contained_labels(safety_vest_indices, no_safety_vest_indices, datas))

    for index in sorted(to_remove, reverse=True):
        datas.pop(index)

    return datas


def find_contained_labels(indices1, indices2, datas):
    """
    Find completely contained labels between two sets of indices.

    Args:
        indices1 (list): First set of indices.
        indices2 (list): Second set of indices.
        datas (list): Detection data.

    Returns:
        set: Indices of completely contained labels to remove.
    """
    to_remove = set()
    for index1 in indices1:
        for index2 in indices2:
            if is_contained(datas[index2][:4], datas[index1][:4]):
                to_remove.add(index2)
            elif is_contained(datas[index1][:4], datas[index2][:4]):
                to_remove.add(index1)
    return to_remove
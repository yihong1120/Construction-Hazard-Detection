from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from .model_loader import MODELS
import numpy as np
import cv2
import gc
from sahi.predict import get_sliced_prediction

detection_blueprint = Blueprint('detection', __name__)
limiter = Limiter(key_func=get_remote_address)

@detection_blueprint.route('/detect', methods=['POST'])
@jwt_required()
@limiter.limit("3000 per minute")
def detect():
    data = request.files['image'].read()
    model_key = request.args.get('model', default='yolov8n', type=str)
    model = MODELS[model_key]
    
    # Convert string data to numpy array
    npimg = np.frombuffer(data, np.uint8)
    # Convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = get_sliced_prediction(
        img,
        model,
        slice_height=384,
        slice_width=384,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3
    )

    # Compile detection data in YOLOv8 format
    datas = []
    for object_prediction in result.object_prediction_list:
        label = int(object_prediction.category.id)
        x1, y1, x2, y2 = [int(x) for x in object_prediction.bbox.to_voc_bbox()]
        confidence = float(object_prediction.score.value)
        datas.append([x1, y1, x2, y2, confidence, label])

    # Remove overlapping labels for Hardhat and Safety Vest categories
    datas = remove_overlapping_labels(datas)

    # Remove completely contained labels for Hardhat and Safety Vest categories
    datas = remove_completely_contained_labels(datas)

    return jsonify(datas)

def remove_overlapping_labels(datas):
    """
    Removes overlapping labels for Hardhat and Safety Vest categories.

    Args:
        datas (list): A list of detection data in YOLOv8 format.

    Returns:
        list: A list of detection data with overlapping labels removed.    
    """
    hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 0]  # Indices of Hardhat detections
    no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2]  # Indices of NO-Hardhat detections
    safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7]  # Indices of Safety Vest detections
    no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4]  # Indices of NO-Safety Vest detections

    to_remove = set()
    for hardhat_index in hardhat_indices:
        for no_hardhat_index in no_hardhat_indices:
            if overlap_percentage(datas[hardhat_index][:4], datas[no_hardhat_index][:4]) > 0.8:
                to_remove.add(no_hardhat_index)

    for safety_vest_index in safety_vest_indices:
        for no_safety_vest_index in no_safety_vest_indices:
            if overlap_percentage(datas[safety_vest_index][:4], datas[no_safety_vest_index][:4]) > 0.8:
                to_remove.add(no_safety_vest_index)

    for index in sorted(to_remove, reverse=True):
        datas.pop(index)

    gc.collect()
    return datas

def overlap_percentage(bbox1, bbox2):
    """
    Calculates the percentage of overlap between two bounding boxes.
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
        bool: True if the inner bounding box is completely contained within the outer bounding box, False otherwise.
    """
    return (inner_bbox[0] >= outer_bbox[0] and inner_bbox[2] <= outer_bbox[2] and
            inner_bbox[1] >= outer_bbox[1] and inner_bbox[3] <= outer_bbox[3])

def remove_completely_contained_labels(datas):
    """
    Removes completely contained labels for Hardhat and Safety Vest categories.

    Args:
        datas (list): A list of detection data in YOLOv8 format.

    Returns:
        list: A list of detection data with completely contained labels removed.
    """
    hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 0]  # Indices of Hardhat detections
    no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2]  # Indices of NO-Hardhat detections
    safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7]  # Indices of Safety Vest detections
    no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4]  # Indices of NO-Safety Vest detections

    to_remove = set()
    # Check hardhats
    for hardhat_index in hardhat_indices:
        for no_hardhat_index in no_hardhat_indices:
            if is_contained(datas[no_hardhat_index][:4], datas[hardhat_index][:4]):
                to_remove.add(no_hardhat_index)
            elif is_contained(datas[hardhat_index][:4], datas[no_hardhat_index][:4]):
                to_remove.add(hardhat_index)

    # Check safety vests
    for safety_vest_index in safety_vest_indices:
        for no_safety_vest_index in no_safety_vest_indices:
            if is_contained(datas[no_safety_vest_index][:4], datas[safety_vest_index][:4]):
                to_remove.add(no_safety_vest_index)
            elif is_contained(datas[safety_vest_index][:4], datas[no_safety_vest_index][:4]):
                to_remove.add(safety_vest_index)

    for index in sorted(to_remove, reverse=True):
        datas.pop(index)

    return datas

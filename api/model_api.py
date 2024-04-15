from flask import Flask, request, jsonify
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import cv2
import gc
import json

app = Flask(__name__)

# Load models (assuming they are pre-downloaded in the 'models/' directory)
MODELS = {
    # 'yolov8n': AutoDetectionModel.from_pretrained("yolov8", model_path='models/best_yolov8n.pt'),
    # 'yolov8s': AutoDetectionModel.from_pretrained("yolov8", model_path='models/best_yolov8s.pt'),
    # 'yolov8m': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8m.pt'),
    'yolov8l': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8l.pt'),
    # 'yolov8x': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8x.pt')
}

@app.route('/detect', methods=['POST'])
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
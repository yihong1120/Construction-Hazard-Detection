from flask import Flask, request, jsonify
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import cv2
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

    return jsonify(datas)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
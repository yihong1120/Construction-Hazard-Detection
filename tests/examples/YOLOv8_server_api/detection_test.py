from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from flask import Flask
from flask_jwt_extended import create_access_token, JWTManager

from examples.YOLOv8_server_api.detection import detection_blueprint
from examples.YOLOv8_server_api.detection import is_contained
from examples.YOLOv8_server_api.detection import calculate_overlap
from examples.YOLOv8_server_api.detection import remove_completely_contained_labels
from examples.YOLOv8_server_api.detection import remove_overlapping_labels

app = Flask(__name__)
# Change this in your real application
app.config['JWT_SECRET_KEY'] = 'super-secret'
jwt = JWTManager(app)
app.register_blueprint(detection_blueprint)


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_detection_route(client):
    # Create JWT token
    with app.app_context():
        access_token = create_access_token(identity='testuser')

    # Load test image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = BytesIO(buffer.tobytes())

    # Test detection endpoint
    response = client.post(
        '/detect',
        headers={'Authorization': f'Bearer {access_token}'},
        content_type='multipart/form-data',
        data={'image': (img_bytes, 'test.jpg')},
    )

    assert response.status_code == 200
    assert isinstance(response.json, list)


def test_remove_overlapping_labels():
    datas = [
        [10, 10, 50, 50, 0.9, 0],  # Hardhat
        [10, 10, 50, 50, 0.8, 2],  # NO-Hardhat
        [100, 100, 150, 150, 0.9, 7],  # Safety Vest
        [100, 100, 150, 150, 0.8, 4],  # NO-Safety Vest
    ]

    updated_datas = remove_overlapping_labels(datas)
    assert len(updated_datas) == 2
    assert all(d[5] in [0, 7] for d in updated_datas)


def test_remove_completely_contained_labels():
    datas = [
        [10, 10, 50, 50, 0.9, 0],  # Hardhat
        [15, 15, 45, 45, 0.8, 2],  # NO-Hardhat contained within Hardhat
        [100, 100, 150, 150, 0.9, 7],  # Safety Vest
        [105, 105, 145, 145, 0.8, 4],  # NO-Safety Vest contained within Safety Vest
    ]

    updated_datas = remove_completely_contained_labels(datas)
    assert len(updated_datas) == 2
    assert all(d[5] in [0, 7] for d in updated_datas)


def test_calculate_overlap():
    bbox1 = [10, 10, 50, 50]
    bbox2 = [30, 30, 70, 70]
    overlap = calculate_overlap(bbox1, bbox2)
    assert overlap > 0


def test_is_contained():
    outer_bbox = [10, 10, 50, 50]
    inner_bbox = [20, 20, 30, 30]
    assert is_contained(inner_bbox, outer_bbox)
    assert not is_contained(outer_bbox, inner_bbox)

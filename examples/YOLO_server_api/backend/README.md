
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Server API (Backend)

This directory contains a FastAPI-based backend for performing object detection with YOLO models, along with optional model file management and integration points for user authentication/authorisation (via `examples.auth`).

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Folder Structure](#folder-structure)
4. [Setup & Installation](#setup--installation)
5. [Running the Server](#running-the-server)
6. [Endpoints](#endpoints)
   - [Detection Endpoint](#detection-endpoint)
   - [Model Management Endpoints](#model-management-endpoints)
7. [Authentication & User Management Integration](#authentication--user-management-integration)
   - [Token Retrieval & Refresh](#token-retrieval--refresh)
8. [Additional Notes](#additional-notes)

## Overview

This backend provides:

- **Object Detection** using YOLO models (see [`detection.py`](./detection.py)).
- **Model Management** for uploading and updating YOLO `.pt` files at runtime (see [`model_files.py`](./model_files.py) and [`models.py`](./models.py)).
- **FastAPI Routers** for cleanly organised endpoints (see [`routers.py`](./routers.py)).
- References external modules in `examples.auth` for:
  - **JWT-based authentication**
  - **Role-based authorisation**
  - **Custom rate limiting**

## Features

- **Multiple YOLO Models**: Dynamically manage different YOLO model weights (`yolo11n`, `yolo11s`, etc.).
- **Watchdog Monitoring**: Automatic model reloading if a `.pt` file is modified on disk.
- **Async Detection Pipeline**: Efficient handling of uploaded images and detection results.
- **Selective Overlap Removal**: Post-processing to remove overlapping or contained bounding boxes.

## Folder Structure

```
examples/YOLO_server_api/backend/
├── app.py            # Main FastAPI application
├── detection.py      # Core detection logic (processing images, bounding box filtering)
├── model_files.py    # Logic for updating/retrieving .pt model files
├── models.py         # DetectionModelManager & Watchdog-based auto-reload
├── routers.py        # FastAPI routers for detection & model management
└── README.md         # You're here
```

## Setup & Installation

1. **Install Dependencies**
   Make sure you have installed all requirements (including `fastapi`, `uvicorn`, `watchdog`, `sahi`, and YOLO dependencies). If you are using the larger repository’s `requirements.txt`, ensure you have:

   ```bash
   pip install -r requirements.txt
   ```

2. **Model Files**
   - By default, `.pt` files are expected in `models/pt/` (relative to the current working directory).
   - The [`DetectionModelManager`](./models.py) will watch for changes in this directory.

3. **Optional**: Configure environment variables or `.env` if you use them (for Redis, database, or JWT secret—handled by `examples.auth`, not this folder).

## Running the Server

1. **Navigate** to the repository root or a suitable project directory.
2. **Start** the application:

   ```bash
   python examples/YOLO_server_api/backend/app.py
   ```
   or
   ```bash
   uvicorn examples.YOLO_server_api.backend.app:app --host 127.0.0.1 --port 8000
   ```

3. **Check** that the server is running on <http://127.0.0.1:8000>.

## Endpoints

All main endpoints are defined in [`routers.py`](./routers.py). Below are a few highlights:

### Detection Endpoint

- **`POST /detect`**
  Upload an image for YOLO-based object detection.

  **Example cURL**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/detect" \
       -H "Authorization: Bearer <ACCESS_TOKEN>" \
       -F "image=@/path/to/image.jpg" \
       -F "model=yolo11n"
  ```
  **Response**: A JSON array of bounding-box data, each item is `[x1, y1, x2, y2, confidence, label_id]`.

### Model Management Endpoints

- **`POST /model_file_update`**
  Upload a new `.pt` file to replace a specific YOLO model.

  **Example cURL**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/model_file_update" \
       -H "Authorization: Bearer <ACCESS_TOKEN_WITH_ADMIN_OR_MODEL_MANAGE_ROLE>" \
       -F "model=yolo11n" \
       -F "file=@/path/to/new_best_yolo11n.pt"
  ```

- **`POST /get_new_model`**
  Retrieve a model file (Base64-encoded) if it’s updated after a specified timestamp.

  **Example cURL**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/get_new_model" \
       -H "Content-Type: application/json" \
       -d '{
            "model": "yolo11n",
            "last_update_time": "2025-01-01T00:00:00"
           }'
  ```

## Authentication & User Management Integration

- **Authentication**: This backend references `examples.auth.jwt_config.jwt_access` for JWT verification.
- **Role-based checks**: Some endpoints (e.g., model uploads) require `admin` or `model_manage` roles, enforced via `examples.auth.jwt_config` and FastAPI dependencies.
- **Rate limiting**: The detection endpoint uses `custom_rate_limiter` from `examples.auth.cache`.

If you wish to disable authentication or rate limiting, you can remove those dependencies from `routers.py` or replace them with your own logic.

### Token Retrieval & Refresh

1. **Login**: To obtain your `ACCESS_TOKEN` and `REFRESH_TOKEN`, send a `POST` request to:
   ```
   POST http://127.0.0.1:8000/login
   ```
   **Request Body** (JSON):
   ```json
   {
     "username": "your_username",
     "password": "your_password"
   }
   ```
   **Response** (example):
   ```json
   {
     "access_token": "<JWT_ACCESS_TOKEN>",
     "refresh_token": "<JWT_REFRESH_TOKEN>",
     "role": "user",
     "username": "your_username",
     "user_id": 1
   }
   ```

2. **Use the Access Token**: For subsequent requests (e.g., `/detect`), include:
   ```
   Authorization: Bearer <ACCESS_TOKEN>
   ```
   in the request header.

3. **Refresh Token**: If your `ACCESS_TOKEN` expires, send a `POST` request to:
   ```
   POST http://127.0.0.1:8000/refresh
   ```
   **Request Body** (JSON):
   ```json
   {
     "refresh_token": "<YOUR_REFRESH_TOKEN>"
   }
   ```
   **Response** (example):
   ```json
   {
     "access_token": "<NEW_JWT_ACCESS_TOKEN>",
     "refresh_token": "<NEW_JWT_REFRESH_TOKEN>",
     "message": "Token refreshed successfully."
   }
   ```
   Use the new `ACCESS_TOKEN` in your `Authorization` header and (optionally) replace the old `REFRESH_TOKEN` locally if you wish.

## Additional Notes

- **Model Reloading**: [`DetectionModelManager`](./models.py) uses Watchdog to watch `.pt` files. When a file changes (e.g., replaced with a new version), the corresponding model is automatically reloaded in memory.
- **Bounding Box Filtering**: [`detection.py`](./detection.py) demonstrates removing overlapping or fully contained bounding boxes for categories like `hardhat` vs. `no_hardhat`.
- **Performance**: For high-throughput scenarios, consider optimising concurrency settings and GPU usage. The code uses `device='cuda:0'` by default if available.
- **Deployment**: You can deploy via Docker, behind an Nginx reverse proxy, or any standard ASGI hosting environment. Make sure to handle volumes for the `models/pt/` directory if you rely on file-based updates.

🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLOv8 Server API example

This section an example implementation of a YOLOv8 Server API, designed to facilitate object detection using the YOLOv8 model. This guide provides information on how to use, configure, and understand the features of this API.

## Usage

3. **Run the server:**
    ```sh
    python app.py
    ```

    or

    ```sh
    gunicorn -w 1 -b 0.0.0.0:8000 "examples.YOLOv8_server_api.app:app"
    ```

4. **Send a request to the API:**
    - Use tools like `curl`, Postman, or your web browser to send a request to the server.
    - Example using `curl`:
        ```sh
        curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/detect
        ```

## Features

- **Authentication**: Secure the API with authentication mechanisms.
- **Caching**: Improve performance with caching of detection results.
- **Model Download**: Automated downloading and loading of the YOLOv8 model.
- **Configuration**: Flexible configuration options to customise the API.
- **Object Detection**: Perform object detection on uploaded images using YOLOv8.
- **Error Handling**: Robust error handling to manage different scenarios gracefully.

## Configuration

The API can be configured through the `config.py` file. Below are some of the key configuration options available:

- **Server Settings**:
  - `HOST`: The hostname to run the server on. Default is `0.0.0.0`.
  - `PORT`: The port to run the server on. Default is `8000`.

- **Model Settings**:
  - `MODEL_PATH`: Path to the YOLOv8 model file.
  - `CONFIDENCE_THRESHOLD`: Confidence threshold for object detection.

- **Cache Settings**:
  - `CACHE_ENABLED`: Enable or disable caching. Default is `True`.
  - `CACHE_EXPIRY`: Cache expiry time in seconds. Default is `3600`.

- **Authentication Settings**:
  - `AUTH_ENABLED`: Enable or disable authentication. Default is `True`.
  - `SECRET_KEY`: Secret key for JWT authentication.

## File Overview

- **app.py**: Main application file that starts the server and defines the API endpoints.
- **auth.py**: Handles authentication mechanisms.
- **cache.py**: Implements caching functionalities.
- **config.py**: Contains configuration settings for the API.
- **detection.py**: Performs object detection using the YOLOv8 model.
- **model_downloader.py**: Handles downloading and loading of the YOLOv8 model.
- **models.py**: Defines data models and schemas.
- **security.py**: Implements security features.

Ensure to review and adjust the configuration settings in `config.py` to suit your specific requirements.

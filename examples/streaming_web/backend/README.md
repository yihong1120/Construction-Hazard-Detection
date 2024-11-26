
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web backend example

This section provides an example implementation of a Streaming Web backend application, designed to facilitate real-time camera feeds and updates. This guide provides information on how to use, configure, and understand the features of this application.

## Usage

1. **Run the server:**
    ```sh
    python app.py
    ```

    or

    ```sh
    uvicorn examples.streaming_web.backend.app:sio_app --host 127.0.0.1 --port 8000
    ```

2. **Open your web browser and navigate to:**
    ```sh
    http://localhost:8000
    ```

## Features

- **Real-Time Streaming**: Display real-time camera feeds with automatic updates.
- **WebSocket Integration**: Utilises WebSocket for efficient real-time communication.
- **Dynamic Content Loading**: Automatically updates camera images without refreshing the page.
- **Responsive Design**: Adapts to various screen sizes for a seamless user experience.
- **Customisable Layout**: Adjust layout and styles using CSS.
- **Rate Limiting**: Implements rate limiting to prevent abuse of the API.
- **Error Handling**: Comprehensive error handling for various exceptions.
- **Static File Serving**: Serves static files such as HTML, CSS, and JavaScript.

## Configuration

The application can be configured through the following files:

- **app.py**: Main application file that starts the server and defines the routes, middleware, and WebSocket integration.
- **routes.py**: Defines the web routes and their respective handlers, including API endpoints for fetching labels, handling WebSocket connections, and processing webhooks.
- **sockets.py**: Manages WebSocket connections, handling events such as connection, disconnection, and updates. It also includes background tasks for updating images.
- **utils.py**: Utility functions for the application, including encoding/decoding values and sending frame data over WebSocket.
- **.env**: Environment variables for configuration, such as Redis connection details.

## File Overview

### app.py
The main entry point of the application that starts the server and sets up routes, middleware, and WebSocket integration.

- **Lifespan Management**: Uses `@asynccontextmanager` to manage application startup and shutdown tasks, such as initializing and closing the Redis connection.
- **CORS Middleware**: Configures CORS to allow cross-origin requests.
- **Static Files**: Serves static files from the specified directory.
- **Socket.IO Integration**: Initializes and configures the Socket.IO server for real-time communication.

### routes.py
Defines the various web routes and their respective request handlers.

- **API Endpoints**:
  - `/api/labels`: Fetches available labels from Redis.
  - `/api/ws/labels/{label}`: WebSocket endpoint for streaming updated frames for a specific label.
  - `/api/ws/stream/{label}/{key}`: WebSocket endpoint for streaming data for a single camera.
  - `/api/webhook`: Processes incoming webhook requests.
  - `/api/upload`: Handles file uploads and saves them to the designated folder.

### sockets.py
Manages WebSocket connections, handling events such as connection, disconnection, and updates. It also includes background tasks for updating images.

- **Socket.IO Events**:
  - `connect`: Handles new client connections.
  - `disconnect`: Handles client disconnections.
  - `error`: Handles errors during WebSocket communication.
- **Background Tasks**:
  - `update_images`: Periodically fetches and emits updated images to connected clients.

### utils.py
Contains utility functions used across the application for various tasks.

- **RedisManager**: Manages asynchronous Redis operations for fetching labels, keys, and image data.
  - `get_labels`: Fetches unique label and stream_name combinations from Redis keys.
  - `get_keys_for_label`: Retrieves Redis keys that match a given label-stream_name pattern.
  - `fetch_latest_frames`: Fetches the latest frames for each Redis stream.
  - `fetch_latest_frame_for_key`: Fetches the latest frame and warnings for a specific Redis key.
- **Utils**: Contains static methods for encoding/decoding values and sending frame data over WebSocket.
  - `encode`: Encodes a value into a URL-safe Base64 string.
  - `decode`: Decodes a URL-safe Base64 string.
  - `send_frames`: Sends the latest frames to the WebSocket client.

### .env
Contains environment variables for configuring the Redis connection, such as `REDIS_HOST`, `REDIS_PORT`, and `REDIS_PASSWORD`.

## Environment Variables

The application uses the following environment variables for configuration:

- `REDIS_HOST`: The hostname of the Redis server (default: `127.0.0.1`).
- `REDIS_PORT`: The port of the Redis server (default: `6379`).
- `REDIS_PASSWORD`: The password for the Redis server (default: empty).

Ensure to review and adjust the configuration settings in the respective files to suit your specific requirements.

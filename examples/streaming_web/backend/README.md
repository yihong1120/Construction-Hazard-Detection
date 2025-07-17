
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web Backend

This directory contains the backend logic for a streaming web application built with FastAPI, allowing users to upload frames (images) and retrieve them via WebSocket streams in real-time.

Below is an overview of each file within `examples/streaming_web/backend/` and instructions on how to run the application, followed by best practices and advice for production usage.

## File Structure

```
examples/streaming_web/backend/
├── __init__.py
├── app.py
├── redis_service.py
├── routers.py
├── schemas.py
└── utils.py
```

### `app.py`
- **Purpose**: Defines and configures the main FastAPI application.
- **Key Highlights**:
  - Creates a `FastAPI` instance using a custom lifespan (`global_lifespan`) to manage start-up and shut-down tasks (see `examples.auth.lifespan`).
  - Configures CORS middleware to allow cross-origin requests (this may need restricting in production).
  - Includes routers for streaming web services (`streaming_web_router`).
  - Provides a `main()` function, which uses `uvicorn.run()` to start the application.

**Usage**:

```bash
uvicorn examples.streaming_web.backend.app:app --host 127.0.0.1 --port 8000
```

or, to use the `main()` function:

```bash
python examples/streaming_web/backend/app.py
```

### `redis_service.py`
- **Purpose**: Handles interactions with Redis for storing and retrieving frame data, along with metadata.
- **Key Functions**:
  1. `scan_for_labels(rds: redis.Redis) -> list[str]`: Scans Redis keys to find distinct labels (base64-encoded in the key, decoded for output). Excludes labels containing `'test'`.
  2. `get_keys_for_label(rds: redis.Redis, label: str) -> list[str]`: Retrieves all keys from Redis associated with a particular label.
  3. `fetch_latest_frames(rds: redis.Redis, last_ids: dict[str, str]) -> list[dict[str, str | bytes | int]]`: Fetches the latest frames for each key in `last_ids`. Returns metadata and raw frame bytes.
  4. `fetch_latest_frame_for_key(rds: redis.Redis, redis_key: str, last_id: str) -> dict[str, str | bytes | int] | None`: Fetches the most recent frame for a single key, ignoring frames older than `last_id`.
  5. `store_to_redis(...) -> None`: Stores an uploaded frame and associated data into Redis Streams (`xadd`).

**Key Notes**:
- Uses base64 encoding/decoding for labels and stream names (`Utils.encode()`).
- Stores frame data in Redis with a maximum length of 10 items (`maxlen=10`).
- Defines a delimiter (`DELIMITER = b'--IMG--'`) for separating JSON metadata and binary frame data in WebSocket communication.
- All functions are type-annotated and documented using Google-style docstrings and British English comments.

### `routers.py`
- **Purpose**: Defines FastAPI endpoints for uploading frames (`POST /frames`), retrieving labels (`GET /labels`), and streaming frames over WebSocket.
- **Endpoints**:
  1. `GET /labels`: Rate-limited to 60 calls/minute, retrieves all labels from Redis using `scan_for_labels`. Returns a `LabelListResponse` model.
  2. `POST /frames`: JWT-protected endpoint for uploading frames and metadata. Returns a `FramePostResponse` model.
  3. `WebSocket /ws/labels/{label}`: Streams all latest frames for a label to the client, using binary messages (header + DELIMITER + frame bytes).
  4. `WebSocket /ws/stream/{label}/{key}`: Streams frames on a **pull-based** mechanism – a client sends `{"action": "pull"}` to retrieve the latest frame, or `{"action": "ping"}` for a keepalive response.
  5. `WebSocket /ws/frames`: Allows authenticated clients to upload frames and metadata via WebSocket.

**WebSocket Interaction**:
- **`/ws/labels/{label}`**:
  - Continuously fetches and sends new frames from all keys belonging to `label`.
  - If no keys exist for that label, it sends an error and closes the connection.
  - Frames are sent as binary: JSON header + DELIMITER + frame bytes.

- **`/ws/stream/{label}/{key}`**:
  - Awaits JSON commands: `{"action": "pull"}` or `{"action": "ping"}`.
  - On `pull`, sends the latest frame as binary (header + DELIMITER + frame bytes).
  - On `ping`, responds with a JSON `{"action": "pong"}`.
  - On unknown action, responds with a JSON error.

- **`/ws/frames`**:
  - Authenticated endpoint for uploading frames and metadata via WebSocket.
  - Expects binary messages: JSON header + DELIMITER + frame bytes.
  - Returns JSON status messages for success or error.

All endpoints and WebSocket handlers are fully type-annotated and documented using Google-style docstrings and British English comments.

### `schemas.py`
- **Purpose**: Contains Pydantic models for structured request and response data.
- **Models**:
  1. `LabelListResponse`: Returns a list of labels in JSON format. Fully type-annotated and documented.
  2. `FramePostResponse`: Indicates the status of a frame upload operation (e.g., `"ok"`) and a message. Fully type-annotated and documented.

These models ensure a consistent and well-defined JSON response structure in all endpoints.

### `utils.py`
- **Purpose**: Provides miscellaneous helper functions, such as base64 encoding/decoding and a utility for sending frames via WebSocket, with full type hints and Google-style docstrings.
- **Key Functions**:
  - `encode(value: str) -> str`: URL-safe Base64-encodes a string.
  - `is_base64(value: str) -> bool`: Checks if a string is valid URL-safe Base64.
  - `decode(value: str) -> str`: Decodes a URL-safe Base64 string if valid, otherwise returns the original string.
  - `send_frames(websocket: WebSocket, label: str, updated_data: list[dict[str, str | bytes | int]]) -> None`: Sends JSON data over a given `WebSocket`, bundling the label and data in a JSON format.

## Running the Application

1. **Install Dependencies**
   Ensure you have installed FastAPI, Uvicorn, Redis (or a Redis client library such as `redis-py`), and other required packages:
   ```bash
   pip install fastapi uvicorn redis fastapi-limiter
   ```

2. **Start Redis**
   You will need a Redis server running locally or accessible over the network.
   ```bash
   redis-server
   ```
   or use Docker if you prefer:
   ```bash
   docker run -p 6379:6379 redis
   ```

3. **Launch the FastAPI App**
   From the project root, run:
   ```bash
   uvicorn examples.streaming_web.backend.app:app --host 127.0.0.1 --port 8000
   ```
   You can then access the OpenAPI docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Testing

Unit tests and integration tests are located in the `tests/` directory. All tests use full type hints, Google-style docstrings, and British English comments for clarity and maintainability.

```bash
pytest --cov=examples.streaming_web.backend --cov-report=term-missing
```

The tests cover:
- **`redis_service_test.py`**: Ensures correct behaviour in storing and retrieving frames from Redis.
- **`routers_test.py`**: Validates endpoints, including WebSocket functionality, using FastAPI’s `TestClient`. All test code is type-annotated and documented.

## Production Considerations

1. **CORS Restrictions**:
   In production, replace `allow_origins=['*']` with a specific list of allowed domains to enhance security.

2. **Rate Limiting**:
   FastAPI-Limiter is configured to limit certain endpoints. Review your rate-limit configuration (times/seconds) to avoid performance bottlenecks or excessive resource usage.

3. **Authentication**:
   Endpoints such as `POST /frames` rely on JWT authentication (`jwt_access`) from `examples.auth.jwt_config`. Verify your token handling is secure in production.

4. **Secure Connection**:
   If running over the internet, set up TLS/SSL for secure communication to protect API calls and WebSocket connections.

5. **Scaling**:
   - Use a production-ready server (e.g., Gunicorn with Uvicorn workers) behind a load balancer.
   - Redis Streams can be scaled with clustering if your application deals with a high volume of images.

## Contact & Further Information

For further details regarding authentication flows (`examples/auth`), user management, or advanced streaming use cases (e.g., employing WebRTC or chunked streaming), explore the broader repository or contact the development team.


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
  - Includes routers for:
    - **Authentication** (`auth_router`)
    - **User management** (`user_management_router`)
    - **Streaming web services** (`streaming_web_router`)
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
  1. `scan_for_labels(rds)`: Scans Redis keys to find distinct labels. Labels are stored as base64-encoded substrings in the key, which are then decoded. Labels containing `'test'` are excluded.
  2. `get_keys_for_label(rds, label)`: Retrieves all keys from Redis associated with a particular label.
  3. `fetch_latest_frames(rds, last_ids)`: Fetches the latest frames for each key in `last_ids`. If a new frame is found, it returns the associated metadata and raw frame bytes.
  4. `fetch_latest_frame_for_key(rds, redis_key, last_id)`: Similar to the above but fetches the most recent frame for a single key, ignoring frames older than `last_id`.
  5. `store_to_redis(rds, site, stream_name, frame_bytes, warnings_json, cone_polygons_json, ...)`: Stores an uploaded frame and associated data into Redis, using Redis Streams (`xadd`).

**Key Notes**:
- Uses base64 encoding and decoding for labels and stream names (`Utils.encode()`).
- Stores frame data in Redis with a maximum length of 10 items (`maxlen=10`), meaning older frames are trimmed if the stream exceeds that length.
- Defines a delimiter (`DELIMITER = b'--IMG--'`) for separating JSON metadata and binary frame data in WebSocket communication.

### `routers.py`
- **Purpose**: Defines FastAPI endpoints for uploading frames (`POST /frames`), retrieving labels (`GET /labels`), and streaming frames over WebSocket.
- **Endpoints**:
  1. `GET /labels`: Rate-limited to 60 calls/minute, retrieves all labels from Redis using `scan_for_labels`.
  2. `POST /frames`: Rate-limited via JWT authentication, accepts an image file and associated JSON strings (warnings, polygons, etc.), which are then saved to Redis.
  3. `WebSocket /ws/labels/{label}`: Streams the latest frames in near real-time for **all** keys under a given label.
  4. `WebSocket /ws/stream/{label}/{key}`: Streams frames on a **pull-based** mechanism – a client sends `{"action": "pull"}` to retrieve the latest frame, or `{"action": "ping"}` for a keepalive response.

**WebSocket Interaction**:
- **`/ws/labels/{label}`**:
  - Continuously fetches and sends new frames from all keys belonging to `label`.
  - Metadata is encoded as JSON (header) and concatenated with the raw frame bytes, separated by the `DELIMITER`.
  - If no keys exist for that label, it sends an error and closes the connection.

- **`/ws/stream/{label}/{key}`**:
  - Awaits JSON commands.
    - `ping` → responds with `pong`.
    - `pull` → fetches the latest frame from Redis, if present, and sends it.
    - Unknown actions → returns a JSON error response.

### `schemas.py`
- **Purpose**: Contains Pydantic models for structured request and response data.
- **Models**:
  1. `LabelListResponse`: Returns a list of labels in JSON format.
  2. `FramePostResponse`: Indicates the status of a frame upload operation (e.g., `"ok"`) and a message.

These models help ensure a consistent and well-defined JSON response structure in our endpoints.

### `utils.py`
- **Purpose**: Provides miscellaneous helper functions, such as base64 encoding/decoding and a utility for sending frames via WebSocket (currently not used in the main routes, but available for future expansions).
- **Key Functions**:
  - `encode(value)`: URL-safe Base64-encodes a string.
  - `is_base64(value)`: Checks if a string is valid URL-safe Base64.
  - `decode(value)`: Decodes the string from Base64 if valid, otherwise returns the original.
  - `send_frames(websocket, label, updated_data)`: Sends JSON data over a given `WebSocket`, bundling the label and data in a JSON format.

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

Unit tests and integration tests are located in the `tests/` directory. For example:

```bash
pytest --cov=examples.streaming_web.backend --cov-report=term-missing
```

The tests cover:
- **`redis_service_test.py`**: Ensuring correct behaviour in storing and retrieving frames from Redis.
- **`routers_test.py`**: Validates endpoints, including WebSocket functionality, using FastAPI’s `TestClient`.

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

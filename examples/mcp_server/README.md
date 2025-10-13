
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Construction Hazard Detection MCP Server

A comprehensive Model Context Protocol (MCP) server implementation that provides AI agent tools for construction safety monitoring and hazard detection.

## Features

### 🔍 Object Detection & Tracking
- **inference.detect_frame**: YOLO-based object detection with configurable confidence thresholds
- Real-time object tracking capabilities
- Support for multiple image formats (base64, file paths)

### ⚠️ Safety Violation Analysis
- **hazard.detect_violations**: Intelligent safety violation detection
- Working hour filtering and site-specific configuration
- Polygon-based safety zone analysis
- Customizable violation rules and thresholds

### 📊 Violation Record Management
- **violations.search**: Advanced violation record querying with filters
- **violations.get**: Retrieve specific violation details by ID
- **violations.get_image**: Access violation images in multiple formats
- **violations.my_sites**: Get user's accessible construction sites

### 📱 Multi-Platform Notifications
- **notify.line_push**: LINE Messaging API integration
- **notify.telegram_send**: Telegram Bot API support
- **notify.broadcast_send**: Custom broadcast notifications
- Image attachment support for all platforms

### 💾 Data Persistence
- **record.send_violation**: Upload violation records with metadata
- **record.batch_send_violations**: Bulk violation record processing
- **record.sync_pending**: Synchronize offline cached records
- **record.get_statistics**: Monitor upload queue and statistics

### 🎥 Live Streaming
- **streaming.start_detection**: Continuous stream monitoring
- **streaming.stop_detection**: Stream management controls
- **streaming.capture_frame**: Single frame capture
- **streaming.get_status**: Stream status monitoring

### 🤖 Model Management
- **model.fetch**: Download and cache ML models
- **model.update**: Update models to latest versions
- **model.list_available**: Browse available model catalog
- **model.get_local**: Manage locally cached models

### 🔧 Utility Functions
- **utils.calculate_polygon_area**: Geometry calculations
- **utils.point_in_polygon**: Spatial analysis
- **utils.bbox_intersection**: Bounding box operations
- **utils.validate_detections**: Data validation

## Installation

### Option 1: Use within existing project (Recommended)

If you're running the MCP server within the Construction-Hazard-Detection project:

```bash
# Install main project dependencies first
pip install -r requirements.txt

# Install MCP-specific dependencies
pip install -r mcp_server/requirements.txt
```

### Option 2: Standalone installation

For independent deployment of just the MCP server:

```bash
cd mcp_server/
# Uncomment the conditional dependencies in requirements.txt first
pip install -r requirements.txt
```

### Dependencies Note

The MCP server is designed to reuse the main project's dependencies to avoid conflicts:
- **Shared**: FastAPI, Pydantic, NumPy, Pillow, python-dotenv, Redis, etc.
- **MCP-specific**: FastMCP, httpx, jsonschema, structlog

## Configuration

The server supports multiple transport modes via environment variables, with streamable-http as the default:

```bash
# Transport configuration
MCP_TRANSPORT=streamable-http  # Default: "streamable-http", options: "stdio", "sse", "streamable-http"
MCP_HOST=0.0.0.0              # Host for HTTP-based transports
MCP_PORT=8000                 # Port for HTTP-based transports

# API endpoints
VIOLATION_RECORD_API_URL=http://localhost:3000/api
VIOLATION_RECORDS_USERNAME=your_username
VIOLATION_RECORDS_PASSWORD=your_password

# Notification services
LINE_CHANNEL_ACCESS_TOKEN=your_line_token
TELEGRAM_BOT_TOKEN=your_telegram_token
BROADCAST_URL=http://localhost:8080/broadcast

# Timeouts
# Per-request timeout used by outbound HTTP calls (seconds)
API_REQUEST_TIMEOUT=30
```

## Usage

### Run as MCP Server

```bash
# Default: streamable-http transport (recommended for web integration)
python -m mcp_server

# With stdio transport (for direct MCP client integration)
MCP_TRANSPORT=stdio python -m mcp_server

# With SSE transport (for server-sent events)
MCP_TRANSPORT=sse python -m mcp_server

# Access the server (for HTTP-based transports)
# HTTP: http://localhost:8000/mcp
# SSE: http://localhost:8000/sse
```

### Example Tool Usage

```python
# Detect objects in construction site image
result = await inference_detect_frame(
    image_base64="data:image/jpeg;base64,/9j/4AAQ...",
    confidence_threshold=0.6,
    track_objects=True
)

# Analyze for safety violations
violations = await hazard_detect_violations(
    detections=result["detections"],
    image_width=1920,
    image_height=1080,
    working_hour_only=True
)

# Send violation notification
if violations["violations_detected"]:
    await notify_line_push(
        recipient_id="user123",
        message=violations["warning_message"],
        image_base64="data:image/jpeg;base64,/9j/4AAQ..."
    )

    # Record violation
    await record_send_violation(
        image_base64="data:image/jpeg;base64,/9j/4AAQ...",
        detections=result["detections"],
        warning_message=violations["warning_message"],
        site_id="construction_site_001"
    )
```

## Architecture

The MCP server is built using the FastMCP framework and follows a modular architecture:

```
mcp_server/
├── __init__.py          # Package initialization
├── server.py            # Main MCP server with tool registration
├── config.py            # Configuration management
├── schemas.py           # JSON schemas for validation
└── tools/               # Tool implementations
    ├── inference.py     # Object detection tools
    ├── hazard.py        # Safety violation analysis
    ├── violations.py    # Violation record management
    ├── notify.py        # Multi-platform notifications
    ├── record.py        # Data persistence tools
    ├── streaming.py     # Live stream processing
    ├── model.py         # ML model management
    └── utils.py         # Utility functions
```

## Integration

This MCP server integrates with the existing Construction-Hazard-Detection system by wrapping its core modules:

- **src/live_stream_detection.py** → inference tools
- **src/danger_detector.py** → hazard analysis tools
- **src/violation_sender.py** → record management tools
- **src/notifiers/** → notification tools
- **src/model_fetcher.py** → model management tools
- **src/utils.py** → utility tools

## Development

### Adding New Tools

1. Create tool class in appropriate `tools/` module
2. Add JSON schema in `schemas.py`
3. Register tool function in `server.py`
4. Update documentation

### Testing

```bash
# Run tests
pytest tests/

# Test specific tool
pytest tests/test_inference.py -v
```

## License

This project is licensed under the same terms as the parent Construction-Hazard-Detection repository.

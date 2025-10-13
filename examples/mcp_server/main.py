from __future__ import annotations

import asyncio
import logging

from fastmcp import FastMCP

from examples.mcp_server.config import get_transport_config
from examples.mcp_server.schemas import DetectionLikeDict
from examples.mcp_server.schemas import HazardResponse
from examples.mcp_server.schemas import InferenceResponse
from examples.mcp_server.tools.hazard import HazardTools
from examples.mcp_server.tools.inference import InferenceTools
from examples.mcp_server.tools.model import ModelTools
from examples.mcp_server.tools.notify import NotifyTools
from examples.mcp_server.tools.record import RecordTools
from examples.mcp_server.tools.streaming import StreamingTools
from examples.mcp_server.tools.utils import UtilsTools
from examples.mcp_server.tools.violations import ViolationsTools


# Configure logging for predictable, structured output across tools
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Initialise FastMCP server
mcp = FastMCP('construction-hazard-detection')

# Initialise tool instances lazily used by MCP tool handlers
inference_tools = InferenceTools()
hazard_tools = HazardTools()
violations_tools = ViolationsTools()
notify_tools = NotifyTools()
record_tools = RecordTools()
streaming_tools = StreamingTools()
model_tools = ModelTools()
utils_tools = UtilsTools()


# === INFERENCE TOOLS ===

@mcp.tool(
    name='inference_detect_frame',
    description='Detect objects in image using YOLO model',
)
async def inference_detect_frame(
    image_base64: str,
    confidence_threshold: float = 0.5,
    track_objects: bool = False,
) -> InferenceResponse:
    """Detect objects in an image using a YOLO model.

    Args:
        image_base64: Base64-encoded image data.
        confidence_threshold: Minimum confidence threshold for detections
            (0.0–1.0).
        track_objects: Whether to enable object tracking.

    Returns:
        dict[str, Any]: A mapping with detections, tracked objects and meta.
    """
    return await inference_tools.detect_frame(
        image_base64=image_base64,
        confidence_threshold=confidence_threshold,
        track_objects=track_objects,
    )


# === HAZARD DETECTION TOOLS ===

@mcp.tool(
    name='hazard_detect_violations',
    description='Analyse detections for safety violations',
)
async def hazard_detect_violations(
    detections: list[DetectionLikeDict] | list[list[float]],
    image_width: int,
    image_height: int,
    working_hour_only: bool = True,
    site_config: dict | None = None,
) -> HazardResponse:
    """Analyse detections for safety violations.

    Args:
        detections: List of detection objects with bbox, class and confidence.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        working_hour_only: Whether to filter by working hours.
        site_config: Optional site-specific configuration.

    Returns:
        dict[str, Any]: Violation analysis with warnings and messages.
    """
    return await hazard_tools.detect_violations(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
        working_hour_only=working_hour_only,
        site_config=site_config,
    )


# === VIOLATIONS MANAGEMENT TOOLS ===

@mcp.tool(name='violations_search', description='Violations Search')
async def violations_search(
    site_id: int | None = None,
    keyword: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Search violation records with filters."""
    return await violations_tools.search(
        site_id=site_id,
        keyword=keyword,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )


@mcp.tool(name='violations_get', description='Violations Get')
async def violations_get(
    violation_id: int,
) -> dict:
    """Get specific violation record by ID."""
    return await violations_tools.get(violation_id=violation_id)


@mcp.tool(name='violations_get_image', description='Violations Get Image')
async def violations_get_image(
    image_path: str,
    as_base64: bool = False,
) -> dict:
    """Get violation image by ID."""
    return await violations_tools.get_image(
        image_path=image_path,
        as_base64=as_base64,
    )


@mcp.tool(
    name='violations_get_image_by_id',
    description='Get violation image by violation_id (convenience)',
)
async def violations_get_image_by_id(
    violation_id: int,
    as_base64: bool = False,
) -> dict:
    """Get violation image by violation id.

    This fetches the details first to resolve the image_path.
    """
    return await violations_tools.get_image_by_violation_id(
        violation_id=violation_id,
        as_base64=as_base64,
    )


@mcp.tool(name='violations_my_sites', description='Violations My Sites')
async def violations_my_sites() -> dict:
    """Get user's accessible construction sites."""
    sites = await violations_tools.my_sites()
    return {'sites': sites}


# === NOTIFICATION TOOLS ===

@mcp.tool(name='notify_line_push', description='Notify Line Push')
async def notify_line_push(
    recipient_id: str,
    message: str,
    image_base64: str | None = None,
) -> dict:
    """Send notification via LINE Messaging API."""
    return await notify_tools.line_push(
        recipient_id=recipient_id,
        message=message,
        image_base64=image_base64,
    )


@mcp.tool(name='notify_broadcast_send', description='Notify Broadcast Send')
async def notify_broadcast_send(
    message: str,
    broadcast_url: str | None = None,
) -> dict:
    """Send broadcast notification."""
    return await notify_tools.broadcast_send(
        message=message,
        broadcast_url=broadcast_url,
    )


@mcp.tool(name='notify_telegram_send', description='Notify Telegram Send')
async def notify_telegram_send(
    chat_id: str,
    message: str,
    image_base64: str | None = None,
) -> dict:
    """Send notification via Telegram Bot API."""
    return await notify_tools.telegram_send(
        chat_id=chat_id,
        message=message,
        image_base64=image_base64,
    )


# === RECORD MANAGEMENT TOOLS ===

@mcp.tool(name='record_send_violation', description='Record Send Violation')
async def record_send_violation(
    image_base64: str,
    detections: list[dict],
    warning_message: str,
    timestamp: str | None = None,
    site_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Send violation record to database/API."""
    return await record_tools.send_violation(
        image_base64=image_base64,
        detections=detections,
        warning_message=warning_message,
        timestamp=timestamp,
        site_id=site_id,
        metadata=metadata,
    )


@mcp.tool(
    name='record_batch_send_violations',
    description='Record Batch Send Violations',
)
async def record_batch_send_violations(
    violations: list[dict],
) -> dict:
    """Send multiple violation records in batch."""
    return await record_tools.batch_send_violations(violations=violations)


@mcp.tool()
async def record_sync_pending() -> dict:
    """Synchronize pending violation records."""
    return await record_tools.sync_pending_records()


@mcp.tool()
async def record_get_statistics() -> dict:
    """Get upload statistics and queue status."""
    return await record_tools.get_upload_statistics()


# === STREAMING TOOLS ===

@mcp.tool(
    name='streaming_start_detection',
    description='Streaming Start Detection',
)
async def streaming_start_detection(
    stream_url: str,
    stream_id: str | None = None,
    detection_interval: float = 1.0,
    save_detections: bool = True,
) -> dict:
    """Start continuous detection on video stream."""
    return await streaming_tools.start_detection_stream(
        stream_url=stream_url,
        stream_id=stream_id,
        detection_interval=detection_interval,
        save_detections=save_detections,
    )


@mcp.tool(
    name='streaming_stop_detection',
    description='Streaming Stop Detection',
)
async def streaming_stop_detection(
    stream_id: str,
) -> dict:
    """Stop continuous detection on stream."""
    return await streaming_tools.stop_detection_stream(stream_id=stream_id)


@mcp.tool()
async def streaming_get_status(
    stream_id: str | None = None,
) -> dict:
    """Get status of detection streams."""
    return await streaming_tools.get_stream_status(stream_id=stream_id)


@mcp.tool(
    name='streaming_capture_frame',
    description='Streaming Capture Frame',
)
async def streaming_capture_frame(
    stream_url: str,
    frame_format: str = 'base64',
) -> dict:
    """Capture single frame from video stream."""
    return await streaming_tools.capture_frame(
        stream_url=stream_url,
        frame_format=frame_format,
    )


# === MODEL MANAGEMENT TOOLS ===

@mcp.tool(
    name='model_fetch',
    description='Model Fetch',
)
async def model_fetch(
    model_name: str,
    model_version: str | None = None,
    force_download: bool = False,
) -> dict:
    """Fetch and download ML model."""
    return await model_tools.fetch_model(
        model_name=model_name,
        model_version=model_version,
        force_download=force_download,
    )


@mcp.tool()
async def model_list_available() -> dict:
    """List available models from repository."""
    return await model_tools.list_available_models()


@mcp.tool(
    name='model_update',
    description='Model Update',
)
async def model_update(
    model_name: str,
    target_version: str | None = None,
) -> dict:
    """Update model to latest or specific version."""
    return await model_tools.update_model(
        model_name=model_name,
        target_version=target_version,
    )


@mcp.tool()
async def model_get_local() -> dict:
    """Get list of locally cached models."""
    return await model_tools.get_local_models()


# === UTILITY TOOLS ===

@mcp.tool(
    name='utils_calculate_polygon_area',
    description='Utils Calculate Polygon Area',
)
async def utils_calculate_polygon_area(
    polygon_points: list[list[float]],
) -> dict:
    """Calculate area of a polygon."""
    return await utils_tools.calculate_polygon_area(
        polygon_points=polygon_points,
    )


@mcp.tool(
    name='utils_point_in_polygon',
    description='Utils Point In Polygon',
)
async def utils_point_in_polygon(
    point: list[float],
    polygon_points: list[list[float]],
) -> dict:
    """Check if point is inside polygon."""
    return await utils_tools.point_in_polygon(
        point=point,
        polygon_points=polygon_points,
    )


@mcp.tool(
    name='utils_bbox_intersection',
    description='Utils Bbox Intersection',
)
async def utils_bbox_intersection(
    bbox1: list[float],
    bbox2: list[float],
) -> dict:
    """Calculate intersection of two bounding boxes."""
    return await utils_tools.bbox_intersection(
        bbox1=bbox1,
        bbox2=bbox2,
    )


@mcp.tool()
async def utils_validate_detections(
    detections: list[dict],
    image_width: int,
    image_height: int,
) -> dict:
    """Validate detection data format and coordinates."""
    return await utils_tools.validate_detection_data(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
    )


async def run_server():
    """Run the MCP server with configured transport."""
    try:
        # Get transport configuration
        transport_config = get_transport_config()

        logger.info('Starting Construction Hazard Detection MCP Server')
        logger.info(f"Transport: {transport_config['transport']}")

        if transport_config['transport'] == 'stdio':
            # Run with stdio transport
            await mcp.run_stdio_async()
        elif transport_config['transport'] == 'sse':
            # Run with SSE transport - no timeout limits
            # Remove keep-alive timeout
            # Remove graceful shutdown timeout
            # Remove HTTP event size limit
            uvicorn_config = {
                'timeout_keep_alive': 0,
                'timeout_graceful_shutdown': None,
                'h11_max_incomplete_event_size': None,
            }
            await mcp.run_sse_async(
                host=transport_config['host'],
                port=transport_config['port'],
                uvicorn_config=uvicorn_config,
                stateless_http=True,
            )
        elif transport_config['transport'] == 'streamable-http':
            # Run with streamable HTTP transport - no timeout limits
            # Remove keep-alive timeout
            # Remove graceful shutdown timeout
            # Remove HTTP event size limit
            uvicorn_config = {
                'timeout_keep_alive': 0,
                'timeout_graceful_shutdown': None,
                'h11_max_incomplete_event_size': None,
            }
            await mcp.run_http_async(
                transport='streamable-http',
                host=transport_config['host'],
                port=transport_config['port'],
                uvicorn_config=uvicorn_config,
                stateless_http=True,
            )
        elif transport_config['transport'] == 'http':
            # Run with standard HTTP transport - no timeout limits
            # Remove keep-alive timeout
            # Remove graceful shutdown timeout
            # Remove HTTP event size limit
            uvicorn_config = {
                'timeout_keep_alive': 0,
                'timeout_graceful_shutdown': None,
                'h11_max_incomplete_event_size': None,
            }
            await mcp.run_http_async(
                transport='http',
                host=transport_config['host'],
                port=transport_config['port'],
                uvicorn_config=uvicorn_config,
                stateless_http=True,
            )
        else:
            raise ValueError(
                f"Unsupported transport type: {transport_config['transport']}",
            )

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(run_server())

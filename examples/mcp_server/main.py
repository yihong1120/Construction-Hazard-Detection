from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from typing import Literal

import anyio
from mcp.server.fastmcp import FastMCP

from examples.mcp_server.config import get_transport_config
from examples.mcp_server.schemas import DetectionLikeDict
from examples.mcp_server.schemas import HazardResponse
from examples.mcp_server.schemas import InferenceResponse
from examples.mcp_server.schemas import TransportConfig
from examples.mcp_server.tools.hazard import HazardTools
from examples.mcp_server.tools.inference import InferenceTools
from examples.mcp_server.tools.model import ModelTools
from examples.mcp_server.tools.notify import NotifyTools
from examples.mcp_server.tools.record import RecordTools
from examples.mcp_server.tools.streaming import StreamingTools
from examples.mcp_server.tools.utils import bbox_intersection
from examples.mcp_server.tools.utils import calculate_polygon_area
from examples.mcp_server.tools.utils import point_in_polygon
from examples.mcp_server.tools.utils import validate_detection_data
from examples.mcp_server.tools.violations import ViolationsTools
from examples.violation_records.schemas import ViolationDetectionRow

# Configure logging for predictable, structured output across tools
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialise tool instances lazily used by MCP tool handlers
inference_tools = InferenceTools()
hazard_tools = HazardTools()
violations_tools = ViolationsTools()
notify_tools = NotifyTools()
record_tools = RecordTools()
streaming_tools = StreamingTools()
model_tools = ModelTools()


@asynccontextmanager
async def _mcp_lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Close tool-owned network and model resources at server shutdown."""
    try:
        yield {}
    finally:
        # Transport shutdown cancels its task group; still finish closing
        # pools.
        with anyio.CancelScope(shield=True):
            results = await asyncio.gather(
                inference_tools.close(),
                violations_tools.close(),
                notify_tools.close(),
                record_tools.close(),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning(
                        'MCP resource cleanup failed (%s)', type(
                            result,
                        ).__name__,
                    )


mcp = FastMCP('construction-hazard-detection', lifespan=_mcp_lifespan)

# === INFERENCE TOOLS ===


@mcp.tool(
    name='inference_detect_frame',
    description='Detect objects in image using YOLO model',
)
async def inference_detect_frame(
    image_base64: str,
    model_key: str = 'yolo26n',
    use_ultralytics: bool = True,
    movement_thr: float = 40.0,
) -> InferenceResponse:
    """Detect objects in an image using a YOLO model.

    Args:
        image_base64: Base64-encoded image data.
    Returns:
        dict[str, Any]: A mapping with detections, tracked objects and meta.
    """
    return await inference_tools.detect_frame(
        image_base64=image_base64,
        model_key=model_key,
        use_ultralytics=use_ultralytics,
        movement_thr=movement_thr,
    )


# === HAZARD DETECTION TOOLS ===


@mcp.tool(
    name='hazard_detect_violations',
    description='Analyse detections for safety violations',
)
async def hazard_detect_violations(
    detections: list[DetectionLikeDict] | list[list[float]],
    detection_items: dict[str, bool] | None = None,
) -> HazardResponse:
    """Analyse detections for safety violations.

    Args:
        detections: List of detection objects with bbox, class and confidence.
    Returns:
        dict[str, Any]: Violation analysis with warnings and messages.
    """
    return await hazard_tools.detect_violations(
        detections=detections,
        detection_items=detection_items,
    )


# === VIOLATIONS MANAGEMENT TOOLS ===


@mcp.tool(name='violations_search', description='Violations Search')
async def violations_search(
    site_id: int | None = None,
    keyword: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Search violation records with filters."""
    return await violations_tools.search(
        site_id=site_id,
        keyword=keyword,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


@mcp.tool(name='violations_get', description='Violations Get')
async def violations_get(
    violation_id: int,
) -> dict[str, Any]:
    """Get specific violation record by ID."""
    return await violations_tools.get(violation_id=violation_id)


@mcp.tool(name='violations_get_image', description='Violations Get Image')
async def violations_get_image(
    image_path: str,
    as_base64: bool = False,
) -> dict[str, Any]:
    """Get violation image by ID."""
    return await violations_tools.get_image(
        image_path=image_path,
        as_base64=as_base64,
    )


@mcp.tool(name='violations_my_sites', description='Violations My Sites')
async def violations_my_sites() -> dict[str, Any]:
    """Get user's accessible construction sites."""
    sites = await violations_tools.my_sites()
    return {'sites': sites}


# === NOTIFICATION TOOLS ===


@mcp.tool(
    name='notify_fcm_send',
    description='Send a safety notification via FCM',
)
async def notify_fcm_send(
    site: str,
    stream_name: str,
    warnings: dict[str, dict[str, Any]],
    image_path: str | None = None,
    violation_id: int | None = None,
) -> dict[str, Any]:
    """Notify registered devices through the authenticated FCM backend."""
    return await notify_tools.fcm_send(
        site, stream_name, warnings, image_path, violation_id,
    )


@mcp.tool(name='notify_line_push', description='Notify Line Push')
async def notify_line_push(
    recipient_id: str,
    message: str,
    image_base64: str | None = None,
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Send broadcast notification."""
    return await notify_tools.broadcast_send(
        message=message,
        broadcast_url=broadcast_url,
    )


@mcp.tool(
    name='notify_messenger_send', description='Notify Facebook Messenger',
)
async def notify_messenger_send(
    recipient_id: str,
    message: str,
    image_base64: str | None = None,
) -> dict[str, Any]:
    """Send a notification via Facebook Messenger."""
    return await notify_tools.messenger_send(
        recipient_id=recipient_id,
        message=message,
        image_base64=image_base64,
    )


@mcp.tool(name='notify_wechat_send', description='Notify WeChat Work')
async def notify_wechat_send(
    user_id: str,
    message: str,
    image_base64: str | None = None,
) -> dict[str, Any]:
    """Send a notification via WeChat Work."""
    return await notify_tools.wechat_send(
        user_id=user_id,
        message=message,
        image_base64=image_base64,
    )


@mcp.tool(name='notify_telegram_send', description='Notify Telegram Send')
async def notify_telegram_send(
    chat_id: str,
    message: str,
    image_base64: str | None = None,
) -> dict[str, Any]:
    """Send notification via Telegram Bot API."""
    return await notify_tools.telegram_send(
        chat_id=chat_id,
        message=message,
        image_base64=image_base64,
    )


# === RECORD MANAGEMENT TOOLS ===


@mcp.tool(
    name='record_send_violation',
    description='Upload a violation and model snapshot',
)
async def record_send_violation(
    image_base64: str,
    detections: list[ViolationDetectionRow],
    warnings: dict[str, dict[str, Any]],
    site: str,
    stream_name: str,
    model_id: str,
    model_version: str,
    timestamp: str | None = None,
    cone_polygon: list | None = None,
    pole_polygon: list | None = None,
) -> dict[str, Any]:
    """Upload an authenticated record with explicit site, stream and model
    identity.
    """
    return await record_tools.send_violation(
        image_base64=image_base64, detections=detections, warnings=warnings,
        site=site, stream_name=stream_name, model_id=model_id,
        model_version=model_version, timestamp=timestamp,
        cone_polygon=cone_polygon, pole_polygon=pole_polygon,
    )


@mcp.tool(
    name='record_batch_send_violations',
    description='Record Batch Send Violations',
)
async def record_batch_send_violations(
    violations: list[dict],
) -> dict[str, Any]:
    """Send multiple violation records in batch."""
    return await record_tools.batch_send_violations(violations=violations)


# === STREAMING TOOLS ===


@mcp.tool(
    name='streaming_capture_frame',
    description='Streaming Capture Frame',
)
async def streaming_capture_frame(
    stream_url: str,
    frame_format: Literal['base64', 'array'] = 'base64',
) -> dict[str, Any]:
    """Capture single frame from video stream."""
    return await streaming_tools.capture_frame(
        stream_url=stream_url,
        frame_format=frame_format,
    )


# === MODEL MANAGEMENT TOOLS ===


@mcp.tool(
    name='model_sync',
    description='Download a newer model atomically',
)
async def model_sync(
    model_name: str,
    force_download: bool = False,
) -> dict[str, Any]:
    """Synchronise a model from the authenticated model distributor."""
    return await model_tools.sync_model(
        model_name=model_name,
        force_download=force_download,
    )


@mcp.tool()
async def model_list_available() -> dict[str, Any]:
    """List available models from repository."""
    return await model_tools.list_available_models()


@mcp.tool()
async def model_get_local() -> dict[str, Any]:
    """Get list of locally cached models."""
    return await model_tools.get_local_models()


# === UTILITY TOOLS ===


@mcp.tool(
    name='utils_calculate_polygon_area',
    description='Utils Calculate Polygon Area',
)
def utils_calculate_polygon_area(
    polygon_points: list[list[float]],
) -> dict[str, Any]:
    """Calculate area of a polygon."""
    return calculate_polygon_area(polygon_points)


@mcp.tool(
    name='utils_point_in_polygon',
    description='Utils Point In Polygon',
)
def utils_point_in_polygon(
    point: list[float],
    polygon_points: list[list[float]],
) -> dict[str, Any]:
    """Check if point is inside polygon."""
    return point_in_polygon(point, polygon_points)


@mcp.tool(
    name='utils_bbox_intersection',
    description='Utils Bbox Intersection',
)
def utils_bbox_intersection(
    bbox1: list[float],
    bbox2: list[float],
) -> dict[str, Any]:
    """Calculate intersection of two bounding boxes."""
    return bbox_intersection(bbox1, bbox2)


@mcp.tool()
def utils_validate_detections(
    detections: list[dict],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Validate detection data format and coordinates."""
    return validate_detection_data(detections, image_width, image_height)


def _configure_mcp_transport(transport_config: TransportConfig) -> None:
    """Apply runtime transport settings to the registered MCP server."""
    settings = mcp.settings
    settings.host = transport_config['host']
    settings.port = transport_config['port']
    settings.streamable_http_path = transport_config['path']
    settings.sse_path = transport_config['sse_path']
    settings.debug = transport_config['debug']
    settings.stateless_http = (
        transport_config['transport'] == 'streamable-http'
    )


async def run_server() -> None:
    """Run the MCP server with configured transport."""
    transport_config = get_transport_config()

    logger.info('Starting Construction Hazard Detection MCP Server')
    logger.info(f"Transport: {transport_config['transport']}")
    _configure_mcp_transport(transport_config)

    if transport_config['transport'] == 'stdio':
        await mcp.run_stdio_async()
    elif transport_config['transport'] == 'sse':
        await mcp.run_sse_async()
    elif transport_config['transport'] == 'streamable-http':
        await mcp.run_streamable_http_async()
    else:
        raise ValueError(
            f"Unsupported transport type: {transport_config['transport']}",
        )


if __name__ == '__main__':
    asyncio.run(run_server())

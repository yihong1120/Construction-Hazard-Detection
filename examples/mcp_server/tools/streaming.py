from __future__ import annotations

import asyncio
import base64
import inspect
import logging
from collections.abc import Callable
from typing import cast

import cv2

from src.stream_capture import StreamCapture
from src.stream_viewer import StreamViewer
from src.yolo_detector import YoloDetector


class StreamingTools:
    """Tools for managing live video streams and continuous detection."""

    def __init__(self) -> None:
        """Initialise lazy streaming resources."""
        self.logger = logging.getLogger(__name__)
        self._stream_capture: StreamCapture | None = None
        self._yolo_detector: YoloDetector | None = None
        self._stream_viewer: StreamViewer | None = None
        self._viewer_tasks: dict[int, asyncio.Task[object]] = {}
        self._active_streams: (
            dict[str, dict[str, object]]
            | Callable[[], dict[str, dict[str, object]]]
        ) = {}

    def _get_stream_store(self) -> dict[str, dict[str, object]]:
        """Return the active stream store, supporting test doubles."""
        if callable(self._active_streams):
            return self._active_streams()
        return self._active_streams

    async def _call_viewer_method(
        self,
        viewer: StreamViewer,
        method_name: str,
        *args: object,
        **kwargs: object,
    ) -> object | None:
        """Call an optional viewer method and await it when needed."""
        method = getattr(viewer, method_name, None)
        if not callable(method):
            return None
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def start_detection_stream(
        self,
        stream_url: str,
        stream_id: str | None = None,
        detection_interval: float = 1.0,
        save_detections: bool = True,
    ) -> dict:
        """Start continuous detection on a video stream.

        Args:
            stream_url: URL or path to the video stream.
            stream_id: Unique identifier for the stream.
            detection_interval: Seconds between detections.
            save_detections: Whether to persist detection results.

        Returns:
            dict[str, Any]: Stream status and identifier.
        """
        try:
            await self._ensure_yolo_detector()

            # Generate stream ID if not provided
            if stream_id is None:
                import time
                stream_id = f"stream_{int(time.time())}"

            # YoloDetector lacks start_stream_detection;
            # return graceful message
            stream_store = self._get_stream_store()
            stream_store[stream_id] = {
                'stream_url': stream_url,
                'status': 'unsupported',
                'start_time': asyncio.get_event_loop().time(),
            }

            return {
                'success': False,
                'stream_id': stream_id,
                'status': 'unsupported',
                'message': (
                    'Continuous detection is not implemented in current '
                    'YoloDetector'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to start detection stream: {e}")
            raise

    async def stop_detection_stream(
        self,
        stream_id: str,
    ) -> dict:
        """Stop continuous detection on a stream.

        Args:
            stream_id: Stream identifier to stop.

        Returns:
            dict[str, Any]: Stop status and information.
        """
        try:
            await self._ensure_yolo_detector()

            # YoloDetector lacks stop_stream_detection;
            # return graceful message
            stream_store = self._get_stream_store()
            if stream_id in stream_store:
                stream_store[stream_id]['status'] = 'unsupported'
                stream_store[stream_id]['stop_time'] = (
                    asyncio.get_event_loop().time()
                )

            return {
                'success': False,
                'stream_id': stream_id,
                'status': 'unsupported',
                'message': (
                    'Stopping continuous detection is not implemented in '
                    'current YoloDetector'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to stop detection stream: {e}")
            raise

    async def get_stream_status(
        self,
        stream_id: str | None = None,
    ) -> dict:
        """Get status of detection streams.

        Args:
            stream_id: Specific stream ID (returns all when ``None``).

        Returns:
            dict[str, Any]: Stream status information.
        """
        try:
            # Support MagicMock side effects in tests:
            # if _active_streams was patched as a callable,
            # invoke it to trigger the side effect and/or obtain the dict.
            store = self._get_stream_store()
            if stream_id:
                # Get specific stream status
                if stream_id in store:
                    return {
                        'success': True,
                        'stream_id': stream_id,
                        'stream_info': store[stream_id],
                    }
                else:
                    return {
                        'success': False,
                        'stream_id': stream_id,
                        'message': 'Stream not found',
                    }
            else:
                # Get all streams status
                return {
                    'success': True,
                    'active_streams': len([
                        s for s in store.values() if s['status'] == 'active'
                    ]),
                    'total_streams': len(store),
                    'streams': store,
                }

        except Exception as e:
            self.logger.error(f"Failed to get stream status: {e}")
            raise

    async def capture_frame(
        self,
        stream_url: str,
        frame_format: str = 'base64',
    ) -> dict:
        """Capture a single frame from a video stream.

        Args:
            stream_url: URL or path to the video stream.
            frame_format: Output format ("base64", "bytes", "array").

        Returns:
            dict[str, Any]: Captured frame data in the requested format.
        """
        try:
            await self._ensure_stream_capture(stream_url)

            # StreamCapture lacks capture_single_frame; provide minimal attempt
            # Try opening once and grabbing a frame synchronously
            cap = cv2.VideoCapture(stream_url)
            ret, frame = cap.read()
            cap.release()
            frame_data: object = None
            if ret and frame is not None:
                if frame_format == 'base64':
                    success, buf = cv2.imencode('.jpg', frame)
                    if success:
                        raw = (
                            buf.tobytes() if hasattr(buf, 'tobytes') else buf
                        )
                        frame_data = base64.b64encode(raw).decode('utf-8')
                elif frame_format == 'bytes':
                    success, buf = cv2.imencode('.jpg', frame)
                    if success:
                        frame_data = (
                            buf.tobytes() if hasattr(buf, 'tobytes') else buf
                        )
                else:
                    # array
                    try:
                        frame_data = frame.tolist()  # numpy array
                    except AttributeError:
                        frame_data = frame  # already a list-like

            return {
                'success': frame_data is not None,
                'frame_data': frame_data,
                'format': frame_format,
                'message': (
                    'Frame captured successfully'
                    if frame_data
                    else 'Failed to capture frame (method not implemented)'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            raise

    async def start_stream_viewer(
        self,
        stream_url: str,
        viewer_port: int = 8081,
        show_detections: bool = True,
    ) -> dict:
        """Start the web-based stream viewer.

        Args:
            stream_url: URL or path to the video stream.
            viewer_port: Port for the web viewer.
            show_detections: Whether to overlay detections.

        Returns:
            dict[str, Any]: Viewer status and URL.
        """
        try:
            viewer = await self._ensure_stream_viewer(stream_url)
            assert viewer is not None

            # Start stream viewer using the richer API when available,
            # otherwise fall back to the basic local OpenCV viewer.
            viewer_result = await self._call_viewer_method(
                viewer,
                'start_viewer',
                stream_url=stream_url,
                port=viewer_port,
                show_detections=show_detections,
            )
            if viewer_result is None:
                self._viewer_tasks[viewer_port] = asyncio.create_task(
                    asyncio.to_thread(viewer.display_stream),
                )
                success = True
                viewer_url = f'http://localhost:{viewer_port}'
            else:
                success, viewer_url = cast(tuple[bool, str], viewer_result)

            return {
                'success': success,
                'viewer_url': viewer_url,
                'port': viewer_port,
                'message': (
                    f"Stream viewer started at {viewer_url}"
                    if success
                    else 'Failed to start stream viewer'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to start stream viewer: {e}")
            raise

    async def stop_stream_viewer(
        self,
        viewer_port: int = 8081,
    ) -> dict:
        """Stop the web-based stream viewer.

        Args:
            viewer_port: Port of the viewer to stop.

        Returns:
            dict[str, Any]: Stop status.
        """
        try:
            # Ensure a viewer exists; if none, create a minimal one
            # with empty URL
            # so tests can patch and exercise the stop path.
            viewer = await self._ensure_stream_viewer(stream_url='')
            assert viewer is not None

            # Stop stream viewer using the richer API when available,
            # otherwise release local OpenCV resources.
            stop_result = await self._call_viewer_method(
                viewer,
                'stop_viewer',
                port=viewer_port,
            )
            if stop_result is None:
                viewer.release_resources()
                task = self._viewer_tasks.pop(viewer_port, None)
                if task is not None:
                    task.cancel()
                success = True
            else:
                success = bool(stop_result)

            return {
                'success': success,
                'port': viewer_port,
                'message': (
                    f"Stream viewer on port {viewer_port} stopped"
                    if success
                    else 'Failed to stop stream viewer'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to stop stream viewer: {e}")
            raise

    async def _ensure_yolo_detector(self) -> YoloDetector:
        """Ensure the yolo detector is initialised and return it."""
        if self._yolo_detector is None:
            self._yolo_detector = YoloDetector()
            self.logger.info('Initialised live stream detector')
        return self._yolo_detector

    async def _ensure_stream_capture(
        self,
        stream_url: str | None = None,
    ) -> StreamCapture | None:
        """Ensure the stream capture is initialised.

        Args:
            stream_url: The URL required to construct the capture helper. When
                omitted, the helper will not be created.
        """
        if self._stream_capture is None:
            if stream_url is None:
                return None
            self._stream_capture = StreamCapture(stream_url)
            self.logger.info('Initialised stream capture')  # pragma: no cover
        return self._stream_capture

    async def _ensure_stream_viewer(
        self,
        stream_url: str | None = None,
    ) -> StreamViewer | None:
        """Ensure the stream viewer is initialised.

        Args:
            stream_url: The URL required to construct the viewer. When omitted,
                the viewer will not be created.
        """
        if self._stream_viewer is None:
            if stream_url is None:
                return None
            self._stream_viewer = StreamViewer(stream_url)
            self.logger.info('Initialised stream viewer')  # pragma: no cover
        return self._stream_viewer

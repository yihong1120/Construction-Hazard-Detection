from __future__ import annotations

import asyncio
import base64
from typing import Literal


class StreamingTools:
    """Capture one frame without creating a viewer or background stream."""

    async def capture_frame(
        self,
        stream_url: str,
        frame_format: Literal['base64', 'array'] = 'base64',
    ) -> dict:
        """Keep OpenCV I/O off the event loop; return JSON-compatible data."""
        if frame_format not in {'base64', 'array'}:
            raise ValueError('frame_format must be base64 or array')
        if not stream_url.strip():
            raise ValueError('stream_url is required')
        return await asyncio.to_thread(
            self._capture_frame, stream_url, frame_format,
        )

    @staticmethod
    def _capture_frame(stream_url: str, frame_format: str) -> dict:
        import cv2

        cap = cv2.VideoCapture(
            stream_url, cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
            ],
        )
        try:
            read, frame = cap.read()
            frame_data = None
            if read and frame is not None:
                if frame_format == 'array':
                    frame_data = frame.tolist()
                else:
                    encoded, buffer = cv2.imencode('.jpg', frame)
                    if encoded:
                        frame_data = base64.b64encode(
                            buffer.tobytes(),
                        ).decode('ascii')
            return {
                'success': frame_data is not None,
                'frame_data': frame_data, 'format': frame_format,
            }
        finally:
            cap.release()

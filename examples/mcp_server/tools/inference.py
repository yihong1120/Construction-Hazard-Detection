from __future__ import annotations

import asyncio
import base64
import logging

import cv2
import numpy as np

from examples.mcp_server.config import get_env_int
from examples.mcp_server.schemas import InferenceResponse
from src.async_http_client import AsyncHttpClientOwner
from src.local_yolo_detector import LocalYoloDetector


_REMOTE_IMAGE_TIMEOUT_SECONDS = max(
    1,
    get_env_int('MCP_REMOTE_IMAGE_TIMEOUT_SECONDS', 20),
)
_MAX_REMOTE_IMAGE_BYTES = max(
    1,
    get_env_int('MCP_MAX_REMOTE_IMAGE_BYTES', 10 * 1024 * 1024),
)


class InferenceTools:
    """Tools for object detection and tracking using YOLO models.

    An instance lazily initialises the underlying detector upon first use to
    minimise import side-effects and improve start-up performance.
    """

    def __init__(self) -> None:
        """Initialise lazy inference resources."""
        self.logger = logging.getLogger(__name__)
        self._detector: LocalYoloDetector | None = None
        self._http_client = AsyncHttpClientOwner(
            timeout=_REMOTE_IMAGE_TIMEOUT_SECONDS,
        )

    async def detect_frame(
        self,
        image_base64: str | None = None,
        image_url: str | None = None,
        model_key: str = 'yolo26n',
        use_ultralytics: bool = True,
        movement_thr: float = 40.0,
    ) -> InferenceResponse:
        """Detect objects in a single image frame.

        Args:
            image_base64: Base64-encoded image data. Provide either this or
                ``image_url``.
            image_url: URL pointing to an image resource. Provide either this
                or ``image_base64``.
            model_key: Identifier for the YOLO model to use.
            use_ultralytics: Prefer the Ultralytics engine locally.
            movement_thr: Movement threshold in pixels for tracking heuristics.

        Returns:
            dict[str, Any]: A mapping containing ``detections``, ``tracked``
            and a ``meta`` section describing the run.
        """
        if not image_base64 and not image_url:
            raise ValueError(
                'Either image_base64 or image_url must be provided',
            )

        frame = await self._load_image(image_base64, image_url)
        if frame is None:
            raise ValueError('Failed to load image')

        if self._detector is None:
            await self._init_detector(
                model_key,
                use_ultralytics,
                movement_thr,
            )
        detector = self._detector
        assert detector is not None

        detections, tracked = await detector.generate_detections(frame)

        return {
            'detections': detections,
            'tracked': tracked,
            'meta': {
                'model_key': model_key,
                'engine': 'ultralytics' if use_ultralytics else 'sahi',
                'tracker': 'ultralytics_builtin',
                # [width, height]
                'frame_size': list(frame.shape[:2][::-1]),
            },
        }

    async def _load_image(
        self,
        image_base64: str | None,
        image_url: str | None,
    ) -> np.ndarray | None:
        """Load an image from base64 or from a remote URL.

        The loader attempts to decode a data URL prefix when present.

        Args:
            image_base64: Base64 image string, optionally prefixed by a data
                URL scheme.
            image_url: HTTP(S) URL for a remote image.

        Returns:
            A decoded OpenCV image (BGR), or ``None`` when decoding fails.
        """
        try:
            if image_base64:
                # Decode base64 image
                if ',' in image_base64:
                    image_base64 = image_base64.split(
                        ',', 1,
                    )[1]  # Remove data URL prefix

                if len(image_base64) > ((_MAX_REMOTE_IMAGE_BYTES + 2) // 3) * 4:
                    raise ValueError('Base64 image exceeds size limit')
                image_bytes = base64.b64decode(image_base64, validate=True)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = await asyncio.to_thread(
                    cv2.imdecode,
                    nparr,
                    cv2.IMREAD_COLOR,
                )

            elif image_url:
                client = await self._http_client._get_client()
                async with client.stream('GET', image_url) as response:
                    response.raise_for_status()
                    content_length = response.headers.get('content-length')
                    if (
                        content_length is not None
                        and int(content_length) > _MAX_REMOTE_IMAGE_BYTES
                    ):
                        raise ValueError('Remote image exceeds size limit')
                    chunks: list[bytes] = []
                    total_bytes = 0
                    async for chunk in response.aiter_bytes():
                        total_bytes += len(chunk)
                        if total_bytes > _MAX_REMOTE_IMAGE_BYTES:
                            raise ValueError('Remote image exceeds size limit')
                        chunks.append(chunk)
                nparr = np.frombuffer(b''.join(chunks), np.uint8)
                frame = await asyncio.to_thread(
                    cv2.imdecode,
                    nparr,
                    cv2.IMREAD_COLOR,
                )

            else:
                return None

            if frame is None:
                raise ValueError('Failed to decode image')

            return frame

        except Exception:
            self.logger.exception('Failed to load image')
            return None

    async def _init_detector(
        self, model_key: str, use_ultralytics: bool,
        movement_thr: float,
    ) -> LocalYoloDetector:
        """Initialise the detection system.

        The underlying detector is imported dynamically to avoid heavy import
        costs for users who only interact with configuration or non-inference
        tools.
        """
        self._detector = LocalYoloDetector(
            model_key=model_key,
            use_ultralytics=use_ultralytics,
            movement_thr=movement_thr,
            fps=get_env_int('TARGET_FPS', 1),
            max_id_keep=get_env_int('MAX_ID_KEEP', 10),
        )

        self.logger.info(
            f"Initialised detector: model={model_key}",
        )
        return self._detector

    async def close(self) -> None:
        """Clean up resources.

        Ensures that any underlying async resources are closed correctly.
        """
        if self._detector:
            await self._detector.close()
            self._detector = None
        await self._http_client.close()

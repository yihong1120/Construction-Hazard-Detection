from __future__ import annotations

import base64
import logging

import cv2
import httpx
import numpy as np

from examples.mcp_server.config import get_env_int
from examples.mcp_server.schemas import InferenceResponse
from src.local_yolo_detector import LocalYoloDetector


class InferenceTools:
    """Tools for object detection and tracking using YOLO models.

    An instance lazily initialises the underlying detector upon first use to
    minimise import side-effects and improve start-up performance.
    """

    def __init__(self) -> None:
        """Initialise lazy inference resources."""
        self.logger = logging.getLogger(__name__)
        self._detector: LocalYoloDetector | None = None

    async def detect_frame(
        self,
        image_base64: str | None = None,
        image_url: str | None = None,
        # Compatibility params accepted but currently not used directly
        confidence_threshold: float = 0.5,
        track_objects: bool = False,
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
            confidence_threshold: Minimum confidence to report (currently
                surfaced in metadata only; the underlying engine may apply its
                own thresholding).
            track_objects: Whether to enable tracking in the detector (metadata
                only if unsupported by the current engine).
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
                'confidence_threshold': confidence_threshold,
                'track_objects': track_objects,
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

                image_bytes = base64.b64decode(image_base64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            elif image_url:
                # Download image from URL. Timeout removed to support very slow
                # networks and large media.
                async with httpx.AsyncClient(timeout=None) as client:
                    response = await client.get(image_url)
                    response.raise_for_status()

                nparr = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            else:
                return None

            if frame is None:
                raise ValueError('Failed to decode image')

            return frame

        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
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

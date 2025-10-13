from __future__ import annotations

import base64
import logging

import cv2
import httpx
import numpy as np

from examples.mcp_server.config import get_env_int
from examples.mcp_server.config import get_env_var
from examples.mcp_server.schemas import InferenceResponse
from src.live_stream_detection import LiveStreamDetector


class InferenceTools:
    """Tools for object detection and tracking using YOLO models.

    An instance lazily initialises the underlying detector upon first use to
    minimise import side-effects and improve start-up performance.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._detector = None
        self._shared_token = {
            'access_token': '',
            'refresh_token': '',
            'is_refreshing': False,
        }

    async def detect_frame(
        self,
        image_base64: str | None = None,
        image_url: str | None = None,
        # Compatibility params accepted but currently not used directly
        confidence_threshold: float = 0.5,
        track_objects: bool = False,
        use_remote: bool = False,
        model_key: str = 'yolo11n',
        use_ultralytics: bool = True,
        remote_tracker: str = 'centroid',
        remote_cost_threshold: float = 0.7,
        ws_frame_size: tuple[int, int] | None = None,
        use_jpeg_ws: bool = True,
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
            use_remote: Use a remote detector (e.g. via WebSocket) rather than
                a local engine.
            model_key: Identifier for the YOLO model to use.
            use_ultralytics: Prefer the Ultralytics engine locally.
            remote_tracker: Tracking algorithm used remotely.
            remote_cost_threshold: Cost threshold for association/matching.
            ws_frame_size: Optional frame size when sending frames remotely.
            use_jpeg_ws: Prefer JPEG encoding for WebSocket transfer.
            movement_thr: Movement threshold in pixels for tracking heuristics.

        Returns:
            dict[str, Any]: A mapping containing ``detections``, ``tracked``
            and a ``meta`` section describing the run.
        """
        try:
            # Validate inputs
            if not image_base64 and not image_url:
                raise ValueError(
                    'Either image_base64 or image_url must be provided',
                )

            # Load image
            frame = await self._load_image(image_base64, image_url)
            if frame is None:
                raise ValueError('Failed to load image')

            # Initialise the detector lazily on first use to keep imports light
            if self._detector is None:
                await self._init_detector(
                    use_remote,
                    model_key,
                    use_ultralytics,
                    remote_tracker,
                    remote_cost_threshold,
                    ws_frame_size,
                    use_jpeg_ws,
                    movement_thr,
                )

            # Perform detection
            detections, tracked = await self._detector.generate_detections(
                frame,
            )

            return {
                'detections': detections,
                'tracked': tracked,
                'meta': {
                    'model_key': model_key,
                    'engine': (
                        'remote'
                        if use_remote
                        else ('ultralytics' if use_ultralytics else 'sahi')
                    ),
                    'tracker': (
                        remote_tracker if use_remote else 'ultralytics_builtin'
                    ),
                    'confidence_threshold': confidence_threshold,
                    'track_objects': track_objects,
                    # [width, height]
                    'frame_size': list(frame.shape[:2][::-1]),
                },
            }

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            raise

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
                if frame is None:
                    # Fallback: try Pillow
                    try:
                        from io import BytesIO
                        from PIL import Image
                        img = Image.open(BytesIO(image_bytes)).convert('RGB')
                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        frame = None

            elif image_url:
                # Download image from URL. Timeout removed to support very slow
                # networks and large media.
                async with httpx.AsyncClient(timeout=None) as client:
                    response = await client.get(image_url)
                    response.raise_for_status()

                nparr = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    # Fallback: try Pillow
                    try:
                        from io import BytesIO
                        from PIL import Image
                        img = Image.open(
                            BytesIO(response.content),
                        ).convert('RGB')
                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        frame = None

            else:
                return None

            if frame is None:
                raise ValueError('Failed to decode image')

            return frame

        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return None

    async def _init_detector(
        self, use_remote: bool, model_key: str, use_ultralytics: bool,
        remote_tracker: str, remote_cost_threshold: float,
        ws_frame_size: tuple[int, int] | None, use_jpeg_ws: bool,
        movement_thr: float,
    ):
        """Initialise the detection system.

        The underlying detector is imported dynamically to avoid heavy import
        costs for users who only interact with configuration or non-inference
        tools.
        """
        # Get API configuration
        api_url = get_env_var('DETECT_API_URL')

        # Create detector
        self._detector = LiveStreamDetector(
            api_url=api_url,
            model_key=model_key,
            detect_with_server=use_remote,
            shared_token=self._shared_token,
            use_ultralytics=use_ultralytics,
            movement_thr=movement_thr,
            fps=get_env_int('TARGET_FPS', 1),
            max_id_keep=get_env_int('MAX_ID_KEEP', 10),
            ws_frame_size=ws_frame_size,
            use_jpeg_ws=use_jpeg_ws,
            remote_tracker=remote_tracker,
            remote_cost_threshold=remote_cost_threshold,
        )

        self.logger.info(
            f"Initialised detector: remote={use_remote}, model={model_key}",
        )

    async def close(self) -> None:
        """Clean up resources.

        Ensures that any underlying async resources are closed correctly.
        """
        if self._detector:
            await self._detector.close()
            self._detector = None

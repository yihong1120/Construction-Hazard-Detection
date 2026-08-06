"""Optional local YOLO and SAHI inference for interactive tools.

The production stream pipeline uses :class:`src.yolo_detector.YoloDetector`,
which submits frames to the shared worker.  This module intentionally owns
the heavyweight local model loading used by the MCP single-frame tool and the
desktop preview CLI.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from dotenv import load_dotenv

from src.ultralytics_args import parse_quantize_value
from src.ultralytics_args import precision_kwargs

load_dotenv()

YOLO: Any = None


class _LazyAutoDetectionModel:
    """Import SAHI's model factory only when sliced prediction is used."""

    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        """Load SAHI's detection model factory on first use."""
        from sahi import AutoDetectionModel as detection_model

        return detection_model.from_pretrained(*args, **kwargs)


def get_sliced_prediction(*args: Any, **kwargs: Any) -> Any:
    """Run SAHI sliced prediction, importing SAHI lazily."""
    from sahi.predict import get_sliced_prediction as prediction

    return prediction(*args, **kwargs)


AutoDetectionModel: Any = _LazyAutoDetectionModel


def _yolo_class() -> Any:
    """Return the Ultralytics YOLO class, importing it only when needed."""
    global YOLO
    if YOLO is None:
        from ultralytics import YOLO as yolo_class

        YOLO = yolo_class
    return YOLO


def _sahi_detection_model() -> Any:
    """Return the SAHI detection model class, importing it lazily."""
    return AutoDetectionModel


def _sahi_sliced_prediction() -> Any:
    """Return SAHI sliced prediction function, importing it lazily."""
    return get_sliced_prediction


class LocalYoloDetector:
    """Perform local Ultralytics or SAHI inference for interactive tooling."""

    def __init__(
        self,
        model_key: str = 'yolo26n',
        output_folder: str | None = None,
        use_ultralytics: bool = True,
        movement_thr: float = 40.0,
        fps: int = 1,
        max_id_keep: int = 10,
    ) -> None:
        """Initialise a local model and its lightweight tracking state."""
        self.model_key = model_key
        self.output_folder = output_folder
        self.use_ultralytics = use_ultralytics
        self.local_device = os.getenv('DETECT_LOCAL_DEVICE', 'cuda:0')
        self.local_imgsz = int(os.getenv('DETECT_LOCAL_IMGSZ', '640'))
        self.local_half = os.getenv(
            'DETECT_LOCAL_HALF',
            'true',
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        self.local_quantize = parse_quantize_value(
            os.getenv('DETECT_LOCAL_QUANTIZE'),
        )
        self._logger = logging.getLogger(__name__)
        self.movement_thr = movement_thr
        self.movement_thr_sq = movement_thr * movement_thr
        self.fps = fps
        self.max_id_keep = max_id_keep
        self.frame_count = 0
        self.prev_centers: dict[int, tuple[float, float]] = {}
        self.prev_centers_last_seen: dict[int, int] = {}

        if use_ultralytics:
            self.ultralytics_model = _yolo_class()(
                f'models/pt/best_{model_key}.pt',
            )
        else:
            self.model = _sahi_detection_model().from_pretrained(
                'yolo26',
                model_path=str(Path('models/pt') / f'best_{model_key}.pt'),
                device=self.local_device,
            )

    async def _detect_local(self, frame: np.ndarray) -> list[list[float]]:
        """Run one local inference without Ultralytics tracking state."""
        if self.use_ultralytics:
            result = self.ultralytics_model(frame)
            boxes = result[0].boxes
            return [
                [
                    *map(float, boxes.xyxy[index].tolist()),
                    float(boxes.conf[index].item()),
                    int(boxes.cls[index].item()),
                ]
                for index in range(len(boxes))
            ]

        sahi_result: Any = _sahi_sliced_prediction()(
            frame,
            self.model,
            slice_height=376,
            slice_width=376,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )
        return [
            [
                *map(int, obj.bbox.to_voc_bbox()),
                float(obj.score.value),
                int(obj.category.id),
            ]
            for obj in sahi_result.object_prediction_list
        ]

    async def generate_detections(
        self,
        frame: np.ndarray,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Generate local detections and tracking annotations."""
        self.frame_count += 1
        if not self.use_ultralytics:
            sahi_detections = await self._detect_local(frame)
            return sahi_detections, [
                row + [-1, 0]
                for row in sahi_detections
            ]

        try:
            results = self.ultralytics_model.track(
                frame,
                persist=True,
                verbose=False,
                device=self.local_device,
                imgsz=self.local_imgsz,
                **precision_kwargs(self.local_half, self.local_quantize),
            )
        except Exception as exc:
            if not self._is_cuda_oom(exc):
                raise
            self._logger.error(
                'Local YOLO CUDA out of memory for model %s. %s',
                self.model_key,
                'Returning empty detections.',
            )
            self._release_local_model()
            self._cleanup_prev_centers()
            return [], []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._cleanup_prev_centers()
            return [], []

        box_data = getattr(boxes, 'data', None)
        if isinstance(box_data, (np.ndarray, torch.Tensor)):
            if hasattr(box_data, 'cpu'):
                box_data = box_data.cpu()
            box_rows = box_data.tolist()
        else:
            xyxy_rows = boxes.xyxy.tolist()
            confidences = boxes.conf.tolist()
            class_ids = boxes.cls.tolist()
            track_ids = None if boxes.id is None else boxes.id.tolist()
            box_rows = [
                [
                    *coordinates,
                    *([] if track_ids is None else [track_ids[index]]),
                    confidences[index],
                    class_ids[index],
                ]
                for index, coordinates in enumerate(xyxy_rows)
            ]

        detections: list[list[float]] = []
        tracked: list[list[float]] = []
        for row in box_rows:
            xyxy = row[:4]
            confidence = float(row[-2])
            class_id = int(row[-1])
            track_id = int(row[-3]) if len(row) == 7 else -1
            is_moving = 0

            if track_id != -1:
                center = (
                    (xyxy[0] + xyxy[2]) * 0.5,
                    (xyxy[1] + xyxy[3]) * 0.5,
                )
                previous = self.prev_centers.get(track_id)
                if previous:
                    distance_sq = (
                        (center[0] - previous[0]) ** 2
                        + (center[1] - previous[1]) ** 2
                    )
                    is_moving = int(distance_sq > self.movement_thr_sq)
                self.prev_centers[track_id] = center
                self.prev_centers_last_seen[track_id] = self.frame_count

            detections.append(xyxy + [confidence, class_id])
            tracked.append(xyxy + [confidence, class_id, track_id, is_moving])

        self._cleanup_prev_centers()
        return detections, tracked

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        """Return whether an exception represents CUDA memory exhaustion."""
        message = str(exc).lower()
        return (
            'out of memory' in message
            and ('cuda' in message or 'accelerator' in message)
        )

    def _release_local_model(self) -> None:
        """Release local CUDA resources after an out-of-memory error."""
        if hasattr(self, 'ultralytics_model'):
            del self.ultralytics_model
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _cleanup_prev_centers(self) -> None:
        """Discard stale local track centres periodically."""
        if self.frame_count % 10 != 0:
            return
        expired_ids = [
            track_id
            for track_id, last_seen in self.prev_centers_last_seen.items()
            if self.frame_count - last_seen > self.max_id_keep
        ]
        for track_id in expired_ids:
            self.prev_centers.pop(track_id, None)
            self.prev_centers_last_seen.pop(track_id, None)

    async def run_detection(self, stream_url: str) -> None:
        """Display local detections for a stream until ``q`` is pressed."""
        capture = cv2.VideoCapture(stream_url)
        if not capture.isOpened():
            raise ValueError('Failed to open stream.')
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    await asyncio.sleep(1)
                    continue
                _, tracked = await self.generate_detections(frame)
                display = frame.copy()
                for row in tracked:
                    x1, y1, x2, y2, _, _, track_id, moving = row
                    cv2.rectangle(
                        display,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        display,
                        f'ID{track_id} M{moving}',
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                cv2.imshow('Stream', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    async def close(self) -> None:
        """Release the local model when an interactive caller is finished."""
        self._release_local_model()


async def main() -> None:
    """Run the local interactive preview CLI."""
    parser = argparse.ArgumentParser(
        description='Live stream detection with local YOLO inference',
    )
    parser.add_argument('--url', type=str, required=True)
    parser.add_argument('--model_key', type=str, default='yolo26n')
    parser.add_argument('--use_ultralytics', action='store_true')
    args = parser.parse_args()
    detector = LocalYoloDetector(
        model_key=args.model_key,
        use_ultralytics=args.use_ultralytics,
    )
    await detector.run_detection(args.url)


if __name__ == '__main__':
    asyncio.run(main())

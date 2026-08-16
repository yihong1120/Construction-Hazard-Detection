from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import NotRequired
from typing import TypeAlias
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


CLASS_NAMES: dict[int, str] = {
    0: 'hardhat',
    1: 'mask',
    2: 'no-hardhat',
    3: 'no-mask',
    4: 'no-safety-vest',
    5: 'person',
    6: 'safety-cone',
    7: 'safety-vest',
    8: 'machinery',
    9: 'utility-pole',
    10: 'vehicle',
}

CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    'hardhat': (76, 175, 80),
    'helmet': (76, 175, 80),
    'safety-vest': (76, 175, 80),
    'vest': (76, 175, 80),
    'no-hardhat': (244, 67, 54),
    'no_hardhat': (244, 67, 54),
    'no-safety-vest': (244, 67, 54),
    'no_vest': (244, 67, 54),
    'person': (255, 152, 0),
    'machinery': (255, 171, 64),
    'vehicle': (255, 255, 0),
    'car': (255, 255, 0),
    'utility-pole': (33, 150, 243),
    'utility_pole': (33, 150, 243),
    'safety-cone': (33, 150, 243),
    'cone': (33, 150, 243),
}
WARNING_RGB: tuple[int, int, int] = (244, 67, 54)

CLASS_LABELS: dict[str, dict[str, str]] = {
    'en': {
        'hardhat': 'hardhat',
        'mask': 'mask',
        'no-hardhat': 'no hardhat',
        'no-mask': 'no mask',
        'no-safety-vest': 'no safety vest',
        'person': 'person',
        'safety-cone': 'safety cone',
        'safety-vest': 'safety vest',
        'machinery': 'machinery',
        'utility-pole': 'utility pole',
        'vehicle': 'vehicle',
        'danger': 'danger',
        'unknown': 'unknown',
    },
    'zh-TW': {
        'hardhat': '安全帽',
        'mask': '口罩',
        'no-hardhat': '未戴安全帽',
        'no-mask': '未戴口罩',
        'no-safety-vest': '未穿安全背心',
        'person': '人員',
        'safety-cone': '交通錐',
        'safety-vest': '安全背心',
        'machinery': '機具',
        'utility-pole': '電桿',
        'vehicle': '車輛',
        'danger': '危險',
        'unknown': '未知',
    },
    'zh-CN': {
        'hardhat': '安全帽',
        'mask': '口罩',
        'no-hardhat': '未戴安全帽',
        'no-mask': '未戴口罩',
        'no-safety-vest': '未穿安全背心',
        'person': '人员',
        'safety-cone': '交通锥',
        'safety-vest': '安全背心',
        'machinery': '机具',
        'utility-pole': '电杆',
        'vehicle': '车辆',
        'danger': '危险',
        'unknown': '未知',
    },
    'ja': {
        'hardhat': 'ヘルメット',
        'mask': 'マスク',
        'no-hardhat': 'ヘルメットなし',
        'no-mask': 'マスクなし',
        'no-safety-vest': '安全ベストなし',
        'person': '作業員',
        'safety-cone': 'カラーコーン',
        'safety-vest': '安全ベスト',
        'machinery': '重機',
        'utility-pole': '電柱',
        'vehicle': '車両',
        'danger': '危険',
        'unknown': '不明',
    },
    'vi': {
        'hardhat': 'mu bao ho',
        'mask': 'khau trang',
        'no-hardhat': 'khong mu bao ho',
        'no-mask': 'khong khau trang',
        'no-safety-vest': 'khong ao bao ho',
        'person': 'nguoi',
        'safety-cone': 'coc an toan',
        'safety-vest': 'ao bao ho',
        'machinery': 'may moc',
        'utility-pole': 'cot dien',
        'vehicle': 'xe',
        'danger': 'nguy hiem',
        'unknown': 'khong ro',
    },
    'id': {
        'hardhat': 'helm',
        'mask': 'masker',
        'no-hardhat': 'tanpa helm',
        'no-mask': 'tanpa masker',
        'no-safety-vest': 'tanpa rompi',
        'person': 'orang',
        'safety-cone': 'kerucut',
        'safety-vest': 'rompi',
        'machinery': 'mesin',
        'utility-pole': 'tiang listrik',
        'vehicle': 'kendaraan',
        'danger': 'bahaya',
        'unknown': 'tidak dikenal',
    },
    'fr': {
        'hardhat': 'casque',
        'mask': 'masque',
        'no-hardhat': 'sans casque',
        'no-mask': 'sans masque',
        'no-safety-vest': 'sans gilet',
        'person': 'personne',
        'safety-cone': 'cone',
        'safety-vest': 'gilet',
        'machinery': 'machine',
        'utility-pole': 'poteau',
        'vehicle': 'vehicule',
        'danger': 'danger',
        'unknown': 'inconnu',
    },
    'th': {
        'hardhat': 'หมวกนิรภัย',
        'mask': 'หน้ากาก',
        'no-hardhat': 'ไม่สวมหมวก',
        'no-mask': 'ไม่สวมหน้ากาก',
        'no-safety-vest': 'ไม่สวมเสื้อสะท้อนแสง',
        'person': 'คนงาน',
        'safety-cone': 'กรวยจราจร',
        'safety-vest': 'เสื้อสะท้อนแสง',
        'machinery': 'เครื่องจักร',
        'utility-pole': 'เสาไฟ',
        'vehicle': 'ยานพาหนะ',
        'danger': 'อันตราย',
        'unknown': 'ไม่ทราบ',
    },
}
SUPPORTED_LABEL_LANGUAGES: tuple[str, ...] = tuple(CLASS_LABELS.keys())

WARNING_LABELS: dict[str, dict[str, str]] = {
    'en': {
        'warning_no_hardhat': 'No hardhat',
        'warning_no_mask': 'No mask',
        'warning_no_safety_vest': 'No safety vest',
        'warning_close_to_machinery': 'Too close to machinery',
        'warning_close_to_vehicle': 'Too close to vehicle',
        'warning_people_in_controlled_area': 'In restricted area',
        'warning_people_in_utility_pole_controlled_area': 'In pole area',
        'detect_machinery_close_to_pole': 'Machinery near pole',
    },
    'zh-TW': {
        'warning_no_hardhat': '未戴安全帽',
        'warning_no_mask': '未戴口罩',
        'warning_no_safety_vest': '未穿安全背心',
        'warning_close_to_machinery': '靠近機具',
        'warning_close_to_vehicle': '靠近車輛',
        'warning_people_in_controlled_area': '進入管制區',
        'warning_people_in_utility_pole_controlled_area': '進入電桿管制區',
        'detect_machinery_close_to_pole': '機具靠近電桿',
    },
    'zh-CN': {
        'warning_no_hardhat': '未戴安全帽',
        'warning_no_mask': '未戴口罩',
        'warning_no_safety_vest': '未穿安全背心',
        'warning_close_to_machinery': '靠近机具',
        'warning_close_to_vehicle': '靠近车辆',
        'warning_people_in_controlled_area': '进入管制区',
        'warning_people_in_utility_pole_controlled_area': '进入电杆管制区',
        'detect_machinery_close_to_pole': '机具靠近电杆',
    },
    'ja': {
        'warning_no_hardhat': 'ヘルメットなし',
        'warning_no_mask': 'マスクなし',
        'warning_no_safety_vest': '安全ベストなし',
        'warning_close_to_machinery': '重機に接近',
        'warning_close_to_vehicle': '車両に接近',
        'warning_people_in_controlled_area': '立入禁止区域内',
        'warning_people_in_utility_pole_controlled_area': '電柱区域内',
        'detect_machinery_close_to_pole': '重機が電柱に接近',
    },
    'vi': {
        'warning_no_hardhat': 'Không đội mũ bảo hộ',
        'warning_no_mask': 'Không đeo khẩu trang',
        'warning_no_safety_vest': 'Không mặc áo bảo hộ',
        'warning_close_to_machinery': 'Quá gần máy móc',
        'warning_close_to_vehicle': 'Quá gần xe',
        'warning_people_in_controlled_area': 'Trong khu vực hạn chế',
        'warning_people_in_utility_pole_controlled_area': (
            'Trong khu vực cột điện'
        ),
        'detect_machinery_close_to_pole': 'Máy móc gần cột điện',
    },
    'id': {
        'warning_no_hardhat': 'Tanpa helm',
        'warning_no_mask': 'Tanpa masker',
        'warning_no_safety_vest': 'Tanpa rompi',
        'warning_close_to_machinery': 'Terlalu dekat mesin',
        'warning_close_to_vehicle': 'Terlalu dekat kendaraan',
        'warning_people_in_controlled_area': 'Di area terbatas',
        'warning_people_in_utility_pole_controlled_area': (
            'Di area tiang listrik'
        ),
        'detect_machinery_close_to_pole': 'Mesin dekat tiang listrik',
    },
    'fr': {
        'warning_no_hardhat': 'Sans casque',
        'warning_no_mask': 'Sans masque',
        'warning_no_safety_vest': 'Sans gilet',
        'warning_close_to_machinery': 'Trop près de la machine',
        'warning_close_to_vehicle': 'Trop près du véhicule',
        'warning_people_in_controlled_area': 'Zone restreinte',
        'warning_people_in_utility_pole_controlled_area': 'Zone du poteau',
        'detect_machinery_close_to_pole': 'Machine près du poteau',
    },
    'th': {
        'warning_no_hardhat': 'ไม่สวมหมวกนิรภัย',
        'warning_no_mask': 'ไม่สวมหน้ากาก',
        'warning_no_safety_vest': 'ไม่สวมเสื้อสะท้อนแสง',
        'warning_close_to_machinery': 'ใกล้เครื่องจักรเกินไป',
        'warning_close_to_vehicle': 'ใกล้ยานพาหนะเกินไป',
        'warning_people_in_controlled_area': 'อยู่ในพื้นที่ควบคุม',
        'warning_people_in_utility_pole_controlled_area': 'อยู่ในพื้นที่เสาไฟ',
        'detect_machinery_close_to_pole': 'เครื่องจักรใกล้เสาไฟ',
    },
}

LANGUAGE_ALIASES: dict[str, str] = {
    'en-gb': 'en',
    'en-us': 'en',
    'zh': 'zh-TW',
    'zh-hant': 'zh-TW',
    'zh-tw': 'zh-TW',
    'zh-hk': 'zh-TW',
    'zh-mo': 'zh-TW',
    'zh-hans': 'zh-CN',
    'zh-cn': 'zh-CN',
    'zh-sg': 'zh-CN',
    'jp': 'ja',
    'ja-jp': 'ja',
    'vi-vn': 'vi',
    'id-id': 'id',
    'fr-fr': 'fr',
    'fr-ca': 'fr',
    'th-th': 'th',
}

DETECTION_WARNING_KEYS: dict[str, str] = {
    'no-hardhat': 'warning_no_hardhat',
    'no_hardhat': 'warning_no_hardhat',
    'no-helmet': 'warning_no_hardhat',
    'no_helmet': 'warning_no_hardhat',
    'no-mask': 'warning_no_mask',
    'no_mask': 'warning_no_mask',
    'no-safety-vest': 'warning_no_safety_vest',
    'no_safety_vest': 'warning_no_safety_vest',
    'no-vest': 'warning_no_safety_vest',
    'no_vest': 'warning_no_safety_vest',
}

TrackingDetection: TypeAlias = list[float]
TrackingDetections: TypeAlias = list[TrackingDetection]
WarningBoundingBox: TypeAlias = list[float | int]


class WarningDetails(TypedDict):
    """Represent one detector warning and optional proximity evidence.

    Attributes:
        count: Number of active instances for the warning type.
        person_bboxes: Optional person boxes associated with a proximity warning.
        person_track_ids: Optional tracking identifiers for affected people.
    """

    count: int
    person_bboxes: NotRequired[list[WarningBoundingBox]]
    person_track_ids: NotRequired[list[str]]


WarningPayload: TypeAlias = dict[str, WarningDetails]
PolygonCoordinates: TypeAlias = list[list[float]]
PolygonCollection: TypeAlias = list[PolygonCoordinates]


_overlay_parse_cache_size = max(
    16,
    int(os.getenv('STREAMING_OVERLAY_PARSE_CACHE_SIZE', '256')),
)
_overlay_max_labels = max(
    0,
    int(os.getenv('STREAMING_OVERLAY_MAX_LABELS', '40')),
)
_overlay_draw_labels = (
    os.getenv('STREAMING_OVERLAY_DRAW_LABELS', 'true').lower() == 'true'
)
_overlay_label_warnings_only = (
    os.getenv(
        'STREAMING_OVERLAY_LABEL_WARNINGS_ONLY',
        'false',
    ).lower() == 'true'
)
_overlay_draw_warning_summary = (
    os.getenv(
        'STREAMING_OVERLAY_DRAW_WARNING_SUMMARY',
        'true',
    ).lower() == 'true'
)
_overlay_draw_warning_status = (
    os.getenv(
        'STREAMING_OVERLAY_DRAW_WARNING_STATUS',
        'false',
    ).lower() == 'true'
)
_overlay_max_warning_summary_items = max(
    1,
    int(os.getenv('STREAMING_OVERLAY_MAX_WARNING_SUMMARY_ITEMS', '5')),
)
_overlay_text_bitmap_cache_size = max(
    128,
    int(os.getenv('STREAMING_OVERLAY_TEXT_BITMAP_CACHE_SIZE', '4096')),
)


@dataclass(frozen=True)
class DetectionOverlay:
    """Represent one normalised detection ready for drawing.

    Attributes:
        class_name: Canonical detector class name.
        confidence: Detector confidence score.
        bbox: Clipped pixel bounding box in left, top, right, bottom order.
        track_id: Optional detector tracking identifier.
        is_warning: Whether the detection should use warning presentation.
    """

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: str | None = None
    is_warning: bool = False


@dataclass(frozen=True)
class _RenderedTextBitmap:
    """Represent cached multilingual text pixels for ROI drawing.

    Attributes:
        bgra: Four-channel text pixels in OpenCV channel order.
        width: Bitmap width in pixels.
        height: Bitmap height in pixels.
    """

    bgra: np.ndarray
    width: int
    height: int


def normalise_overlay_mode(value: str | None) -> str:
    """Normalise a user-supplied overlay mode to a supported value.

    Args:
        value: Optional raw mode from a request or configuration.

    Returns:
        ``backend`` for recognised truthy overlay modes; otherwise ``none``.
    """
    mode = (value or 'none').strip().lower()
    if mode in {'1', 'true', 'yes', 'on', 'backend', 'annotated'}:
        return 'backend'
    return 'none'


def normalise_label_language(value: str | None) -> str:
    """Normalise an overlay label language for live rendering.

    Args:
        value: Optional language code or recognised language alias.

    Returns:
        Supported canonical language code, defaulting to English.
    """
    language = (value or 'en').strip().replace('_', '-')
    if language in CLASS_LABELS:
        return language
    alias = LANGUAGE_ALIASES.get(language.lower())
    if alias in CLASS_LABELS:
        return alias
    base_language = language.split('-', 1)[0].lower()
    if base_language in CLASS_LABELS:
        return base_language
    return 'en'


def render_overlay_frame(
    frame_bytes: bytes,
    detection_items_json: str = '[]',
    warnings_json: str = '{}',
    cone_polygons_json: str = '[]',
    pole_polygons_json: str = '[]',
    overlay_mode: str = 'none',
    label_language: str = 'en',
    min_confidence: float = 0.4,
    box_thickness: int = 2,
) -> bytes:
    """Draw backend overlays on an encoded JPEG or PNG frame.

    Args:
        frame_bytes: Encoded source image bytes.
        detection_items_json: JSON tracked detection rows.
        warnings_json: JSON detector-warning payload.
        cone_polygons_json: JSON controlled-area cone polygons.
        pole_polygons_json: JSON utility-pole controlled-area polygons.
        overlay_mode: Requested overlay mode.
        label_language: Requested label language.
        min_confidence: Minimum detection confidence to draw.
        box_thickness: Requested bounding-box line thickness.

    Returns:
        Original bytes when no overlay can be rendered, otherwise JPEG bytes.
    """
    if normalise_overlay_mode(overlay_mode) != 'backend':
        return frame_bytes

    frame = _decode_image(frame_bytes)
    if frame is None:
        return frame_bytes

    frame = render_overlay_array(
        frame,
        detection_items=_parse_tracking_detections(detection_items_json),
        warnings=_parse_warning_payload(warnings_json),
        cone_polygons=_parse_polygon_collection(cone_polygons_json),
        pole_polygons=_parse_polygon_collection(pole_polygons_json),
        overlay_mode='backend',
        label_language=label_language,
        min_confidence=min_confidence,
        box_thickness=box_thickness,
    )

    success, encoded = cv2.imencode(
        '.jpg',
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not success:
        return frame_bytes
    return encoded.tobytes()


def render_overlay_array(
    frame: np.ndarray,
    detection_items: TrackingDetections,
    warnings: WarningPayload,
    cone_polygons: PolygonCollection,
    pole_polygons: PolygonCollection,
    overlay_mode: str = 'backend',
    label_language: str = 'en',
    min_confidence: float = 0.4,
    box_thickness: int = 2,
) -> np.ndarray:
    """Draw backend overlays directly on a BGR frame.

    Args:
        frame: Mutable BGR image array to annotate in place.
        detection_items: Decoded tracked detection rows.
        warnings: Decoded detector-warning payload.
        cone_polygons: Decoded controlled-area cone polygons.
        pole_polygons: Decoded utility-pole controlled-area polygons.
        overlay_mode: Requested overlay mode.
        label_language: Requested label language.
        min_confidence: Minimum detection confidence to draw.
        box_thickness: Requested bounding-box line thickness.

    Returns:
        The supplied frame after any requested drawing operations.
    """
    if normalise_overlay_mode(overlay_mode) != 'backend':
        return frame
    if frame is None or frame.size == 0:
        return frame

    label_language = normalise_label_language(label_language)
    # Geometry is rendered before detections so boxes and labels remain legible.
    _draw_polygon_data(
        frame,
        (
            (cone_polygons, (255, 64, 129), (233, 30, 99), 0.4),
            (pole_polygons, (68, 138, 255), (68, 138, 255), 0.4),
        ),
    )
    detection_warning_counts = _draw_detections_from_data(
        frame,
        detection_items,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        min_confidence=min_confidence,
        warnings=warnings,
        label_language=label_language,
        box_thickness=box_thickness,
    )
    if _overlay_draw_warning_summary:
        _draw_warning_summary(
            frame,
            warnings,
            label_language,
            detection_warning_counts=detection_warning_counts,
        )
    return frame


def _decode_image(frame_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG or PNG bytes into an OpenCV BGR frame.

    Args:
        frame_bytes: Encoded image bytes.

    Returns:
        Decoded BGR frame, or ``None`` when decoding produces no pixels.
    """
    buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    return frame


def _parse_tracking_detections(value: str) -> TrackingDetections:
    """Decode tracked YOLO rows emitted by the detector pipeline.

    Args:
        value: JSON array of trusted tracking rows.

    Returns:
        Decoded tracking rows.
    """
    return json.loads(value)


def _parse_warning_payload(value: str) -> WarningPayload:
    """Decode the warning payload emitted by the danger detector.

    Args:
        value: JSON warning payload.

    Returns:
        Decoded warning map keyed by warning type.
    """
    return json.loads(value)


def _parse_polygon_collection(value: str) -> PolygonCollection:
    """Decode one detector polygon collection.

    Args:
        value: JSON polygon collection.

    Returns:
        Decoded polygon coordinate lists.
    """
    return json.loads(value)


@lru_cache(maxsize=_overlay_parse_cache_size)
def _detections_for_overlay(
    value: str,
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    warnings_json: str,
) -> tuple[DetectionOverlay, ...]:
    """Return cached detections normalised for overlay drawing.

    Args:
        value: JSON tracked detection rows.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.
        min_confidence: Minimum detection confidence to retain.
        warnings_json: JSON detector-warning payload.

    Returns:
        Cached normalised detections including explicit warning targets.
    """
    data = _parse_tracking_detections(value)
    warnings = _parse_warning_payload(warnings_json)
    warning_targets = _warning_targets(warnings, frame_width, frame_height)
    detections = tuple(
        _iter_detections_from_data(
            data,
            frame_width=frame_width,
            frame_height=frame_height,
            warning_classes=_warning_classes(warnings),
            warning_targets=warning_targets,
            min_confidence=min_confidence,
        ),
    )
    return detections + _warning_target_overlays(warning_targets)


def _detections_from_data(
    data: TrackingDetections,
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    warnings: WarningPayload,
) -> tuple[DetectionOverlay, ...]:
    """Normalise detections from already-decoded metadata.

    Args:
        data: Decoded tracked detection rows.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.
        min_confidence: Minimum detection confidence to retain.
        warnings: Decoded detector-warning payload.

    Returns:
        Normalised detections including explicit warning targets.
    """
    warning_targets = _warning_targets(warnings, frame_width, frame_height)
    detections = tuple(
        _iter_detections_from_data(
            data,
            frame_width=frame_width,
            frame_height=frame_height,
            warning_classes=_warning_classes(warnings),
            warning_targets=warning_targets,
            min_confidence=min_confidence,
        ),
    )
    return detections + _warning_target_overlays(warning_targets)


def _draw_detections_from_data(
    frame: np.ndarray,
    data: TrackingDetections,
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    warnings: WarningPayload,
    label_language: str,
    box_thickness: int,
) -> dict[str, int]:
    """Draw decoded detections and return inferred warning counts.

    Args:
        frame: Mutable BGR image array to annotate.
        data: Decoded tracked detection rows.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        min_confidence: Minimum detection confidence to draw.
        warnings: Decoded detector-warning payload.
        label_language: Canonical label language.
        box_thickness: Requested bounding-box line thickness.

    Returns:
        Count of drawn detections for each inferred warning key.
    """
    warning_targets = _warning_targets(warnings, frame_width, frame_height)
    warning_classes = _warning_classes(warnings)
    warning_counts: dict[str, int] = {}
    label_count = 0

    for detection in _iter_detections_from_data(
        data,
        frame_width=frame_width,
        frame_height=frame_height,
        warning_classes=warning_classes,
        warning_targets=warning_targets,
        min_confidence=min_confidence,
    ):
        label_count = _draw_detection_and_update_counts(
            frame,
            detection,
            label_language,
            box_thickness,
            label_count,
            warning_counts,
        )

    for detection in _warning_target_overlays(warning_targets):
        label_count = _draw_detection_and_update_counts(
            frame,
            detection,
            label_language,
            box_thickness,
            label_count,
            warning_counts,
        )

    return warning_counts


def _draw_detection_and_update_counts(
    frame: np.ndarray,
    detection: DetectionOverlay,
    label_language: str,
    box_thickness: int,
    label_count: int,
    warning_counts: dict[str, int],
) -> int:
    """Draw one detection and update warning-summary state.

    Args:
        frame: Mutable BGR image array to annotate.
        detection: Normalised detection to draw.
        label_language: Canonical label language.
        box_thickness: Requested bounding-box line thickness.
        label_count: Number of labels already drawn.
        warning_counts: Mutable inferred warning-count map.

    Returns:
        Updated number of labels drawn.
    """
    draw_label = _should_draw_label(detection, label_count)
    _draw_detection(
        frame,
        detection,
        label_language,
        box_thickness,
        draw_label=draw_label,
    )
    _add_detection_warning_count(warning_counts, detection)
    return label_count + int(draw_label)


def _warning_target_overlays(
    warning_targets: set[tuple[int, int, int, int]],
) -> tuple[DetectionOverlay, ...]:
    """Build direct red person boxes for algorithm-marked warning targets.

    Args:
        warning_targets: Clipped person boxes associated with warnings.

    Returns:
        Synthetic warning detections sorted by bounding box.
    """
    return tuple(
        DetectionOverlay(
            class_name='person',
            confidence=1.0,
            bbox=bbox,
            is_warning=True,
        )
        for bbox in sorted(warning_targets)
    )


def _iter_detections_from_data(
    data: TrackingDetections,
    frame_width: int,
    frame_height: int,
    warning_classes: set[str],
    warning_targets: set[tuple[int, int, int, int]],
    min_confidence: float,
) -> Iterator[DetectionOverlay]:
    """Yield valid detections above the configured confidence threshold.

    Args:
        data: Decoded tracked detection rows.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        warning_classes: Classes that should receive warning presentation.
        warning_targets: Explicit person boxes associated with warnings.
        min_confidence: Minimum detection confidence to retain.

    Yields:
        Normalised valid detections that meet the confidence threshold.
    """
    for item in data:
        detection = _parse_tracking_detection(
            item,
            frame_width=frame_width,
            frame_height=frame_height,
            warning_classes=warning_classes,
            warning_targets=warning_targets,
        )
        if detection is None or detection.confidence < min_confidence:
            continue
        yield detection


def _parse_tracking_detection(
    item: TrackingDetection,
    frame_width: int,
    frame_height: int,
    warning_classes: set[str],
    warning_targets: set[tuple[int, int, int, int]],
) -> DetectionOverlay | None:
    """Parse one tracked YOLO row into clipped overlay coordinates.

    Args:
        item: Trusted tracking row with geometry, confidence, class, and ID.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        warning_classes: Classes that should receive warning presentation.
        warning_targets: Explicit person boxes associated with warnings.

    Returns:
        Normalised detection, or ``None`` for a degenerate bounding box.
    """
    x1, y1, x2, y2 = (float(item[i]) for i in range(4))
    confidence = float(item[4])
    class_id = int(float(item[5]))
    class_name = CLASS_NAMES.get(class_id, f'class-{class_id}')
    if _looks_normalized([x1, y1, x2, y2]):
        x1 *= frame_width
        x2 *= frame_width
        y1 *= frame_height
        y2 *= frame_height
    bbox = _clip_bbox(
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
        frame_width,
        frame_height,
    )
    if bbox is None:
        return None
    track_id = int(item[6])
    return DetectionOverlay(
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        track_id=str(track_id) if track_id >= 0 else None,
        is_warning=(
            class_name in warning_classes
            or _is_warning_target(class_name, bbox, warning_targets)
        ),
    )


def _clip_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    """Clip a bounding box to frame bounds.

    Args:
        x1: First horizontal coordinate.
        y1: First vertical coordinate.
        x2: Second horizontal coordinate.
        y2: Second vertical coordinate.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Ordered clipped pixel box, or ``None`` when it has no area.
    """
    left = max(0, min(x1, x2, frame_width - 1))
    top = max(0, min(y1, y2, frame_height - 1))
    right = max(0, min(max(x1, x2), frame_width - 1))
    bottom = max(0, min(max(y1, y2), frame_height - 1))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _looks_normalized(values: list[float]) -> bool:
    """Determine whether all coordinates appear normalised to zero through one.

    Args:
        values: Coordinates to inspect.

    Returns:
        ``True`` when every value is within the inclusive unit interval.
    """
    return all(0.0 <= value <= 1.0 for value in values)


def _warning_targets(
    warnings: WarningPayload,
    frame_width: int,
    frame_height: int,
) -> set[tuple[int, int, int, int]]:
    """Return person boxes that should be highlighted as warning targets.

    Args:
        warnings: Decoded detector-warning payload.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Clipped explicit person boxes for active proximity warnings.
    """
    targets: set[tuple[int, int, int, int]] = set()
    proximity_keys = (
        ('warning_close_to_machinery', 'machinery'),
        ('warning_close_to_vehicle', 'vehicle'),
    )
    for key, _ in proximity_keys:
        warning = warnings.get(key)
        if warning is None or _warning_count(warning) <= 0:
            continue
        targets.update(
            _explicit_person_warning_bboxes(
                warning,
                frame_width,
                frame_height,
            ),
        )

    return targets


def _explicit_person_warning_bboxes(
    warning: WarningDetails,
    frame_width: int,
    frame_height: int,
) -> set[tuple[int, int, int, int]]:
    """Extract explicit person boxes from one warning payload.

    Args:
        warning: Warning details containing optional person boxes.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Set of clipped valid person boxes.
    """
    targets = set()
    for bbox_raw in warning.get('person_bboxes', []):
        bbox = _normalise_warning_bbox(bbox_raw, frame_width, frame_height)
        if bbox is not None:
            targets.add(bbox)
    return targets


def _normalise_warning_bbox(
    bbox_raw: WarningBoundingBox,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    """Normalise a warning box to clipped pixel coordinates.

    Args:
        bbox_raw: Detector box in normalised or pixel coordinates.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Ordered clipped pixel box, or ``None`` when it has no area.
    """
    x1, y1, x2, y2 = (float(bbox_raw[i]) for i in range(4))
    if _looks_normalized([x1, y1, x2, y2]):
        x1 *= frame_width
        x2 *= frame_width
        y1 *= frame_height
        y2 *= frame_height
    return _clip_bbox(
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
        frame_width,
        frame_height,
    )


def _is_warning_target(
    class_name: str,
    bbox: tuple[int, int, int, int],
    warning_targets: set[tuple[int, int, int, int]],
) -> bool:
    """Determine whether a detection matches an explicit warning target.

    Args:
        class_name: Canonical detection class name.
        bbox: Clipped pixel bounding box for the detection.
        warning_targets: Explicit warning person boxes.

    Returns:
        ``True`` when a person overlaps a warning target sufficiently.
    """
    if class_name != 'person':
        return False
    return any(_bbox_iou(bbox, target) >= 0.85 for target in warning_targets)


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Calculate intersection over union for two pixel boxes.

    Args:
        first: First left, top, right, bottom pixel box.
        second: Second left, top, right, bottom pixel box.

    Returns:
        Intersection-over-union score between zero and one.
    """
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _warning_classes(warnings: WarningPayload) -> set[str]:
    """Return detection classes that should receive warning presentation.

    Args:
        warnings: Decoded detector-warning payload.

    Returns:
        Canonical classes implicated by active warning types.
    """
    classes: set[str] = set()
    if 'warning_no_hardhat' in warnings:
        classes.add('no-hardhat')
    if 'warning_no_safety_vest' in warnings:
        classes.add('no-safety-vest')
    if 'warning_people_in_controlled_area' in warnings:
        classes.add('person')
    if 'warning_people_in_utility_pole_controlled_area' in warnings:
        classes.add('person')
    if 'detect_machinery_close_to_pole' in warnings:
        classes.update({'machinery', 'vehicle'})
    return classes


def has_warning(warnings: WarningPayload) -> bool:
    """Determine whether warning metadata contains an active warning.

    Args:
        warnings: Decoded detector-warning payload.

    Returns:
        ``True`` when any warning count is positive.
    """
    return any(_warning_count(value) > 0 for value in warnings.values())


def _warning_count(value: WarningDetails) -> int:
    """Extract the active warning count from warning details.

    Args:
        value: Typed warning details.

    Returns:
        Active warning count.
    """
    return value['count']


def _warning_summary_lines(
    warnings: WarningPayload,
    label_language: str,
    detections: tuple[DetectionOverlay, ...] = (),
    detection_warning_counts: dict[str, int] | None = None,
) -> list[str]:
    """Build translated warning-summary lines for an overlay.

    Args:
        warnings: Decoded detector-warning payload.
        label_language: Requested label language.
        detections: Optional normalised detections for inferred warnings.
        detection_warning_counts: Optional precomputed inferred warning counts.

    Returns:
        Bounded translated warning-summary lines.
    """
    language = normalise_label_language(label_language)
    labels = WARNING_LABELS.get(language, WARNING_LABELS['en'])
    lines: list[str] = []
    emitted_keys: set[str] = set()

    for key, value in warnings.items():
        count = _warning_count(value)
        if count <= 0:
            continue
        lines.append(
            _format_warning_summary_line(
                key,
                count,
                labels,
            ),
        )
        emitted_keys.add(key)
        if len(lines) >= _overlay_max_warning_summary_items:
            return lines

    inferred_counts = (
        detection_warning_counts
        if detection_warning_counts is not None
        else _warning_counts_from_detections(detections)
    )
    for key, count in inferred_counts.items():
        if key in emitted_keys:
            continue
        lines.append(
            _format_warning_summary_line(
                key,
                count,
                labels,
            ),
        )
        if len(lines) >= _overlay_max_warning_summary_items:
            break
    return lines


def _draw_warning_summary(
    frame: np.ndarray,
    warnings: WarningPayload,
    label_language: str,
    detections: tuple[DetectionOverlay, ...] = (),
    detection_warning_counts: dict[str, int] | None = None,
) -> None:
    """Draw the active warning summary in a frame corner.

    Args:
        frame: Mutable BGR image array to annotate.
        warnings: Decoded detector-warning payload.
        label_language: Requested label language.
        detections: Optional normalised detections for inferred warnings.
        detection_warning_counts: Optional precomputed inferred warning counts.
    """
    lines = _warning_summary_lines(
        warnings,
        label_language,
        detections=detections,
        detection_warning_counts=detection_warning_counts,
    )
    has_active_warning = bool(lines)
    if not has_active_warning and not _overlay_draw_warning_status:
        return
    if has_active_warning:
        lines = ['WARNING'] + lines
    else:
        lines = ['OK', 'No active warning']

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _font_scale(frame)
    thickness = max(1, _line_thickness(frame) - 1)
    padding_x = 10
    padding_y = 8
    gap = 5
    metrics = [
        _measure_label_text(line, frame, font, scale, thickness)
        for line in lines
    ]
    width = max(metric[0] for metric in metrics) + padding_x * 2
    text_block_height = sum(metric[1] + metric[2] for metric in metrics)
    height = text_block_height + gap * (len(lines) - 1) + padding_y * 2
    x1 = max(0, frame.shape[1] - width - 12)
    y1 = max(0, frame.shape[0] - height - 12)
    x2 = min(frame.shape[1] - 1, x1 + width)
    y2 = min(frame.shape[0] - 1, y1 + height)

    roi = frame[y1:y2 + 1, x1:x2 + 1]
    if not roi.size:
        return
    fill = np.empty_like(roi)
    fill[:, :] = (36, 36, 36)
    cv2.addWeighted(fill, 0.78, roi, 0.22, 0, roi)
    border_color = (0, 0, 255) if has_active_warning else (0, 180, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness=2)

    text_rows: list[tuple[str, int, int, int, int]] = []
    cursor_y = y1 + padding_y
    for line, (text_width, text_height, baseline) in zip(
        lines,
        metrics,
        strict=False,
    ):
        text_x = max(x1 + padding_x, x2 - padding_x - text_width)
        baseline_y = cursor_y + text_height
        text_rows.append((line, text_x, baseline_y, text_height, baseline))
        cursor_y += text_height + baseline + gap

    for line, text_x, baseline_y, _, _ in text_rows:
        _draw_label_text(
            frame,
            line,
            (text_x, baseline_y),
            font,
            scale,
            (255, 255, 255),
            max(1, thickness),
            (x1, y1, x2, y2),
        )


def _warning_counts_from_detections(
    detections: tuple[DetectionOverlay, ...],
) -> dict[str, int]:
    """Count normalised warning detections by warning key.

    Args:
        detections: Normalised detections to inspect.

    Returns:
        Count of detections for each mapped warning key.
    """
    counts: dict[str, int] = {}
    for detection in detections:
        _add_detection_warning_count(counts, detection)
    return counts


def _add_detection_warning_count(
    counts: dict[str, int],
    detection: DetectionOverlay,
) -> None:
    """Increment the warning count represented by one detection.

    Args:
        counts: Mutable inferred warning-count map.
        detection: Normalised detection to map to a warning key.
    """
    key = DETECTION_WARNING_KEYS.get(detection.class_name)
    if not key:
        return
    counts[key] = counts.get(key, 0) + 1


def _format_warning_summary_line(
    key: str,
    count: int,
    labels: dict[str, str],
) -> str:
    """Format one translated warning-summary line.

    Args:
        key: Canonical warning key.
        count: Active count for the warning.
        labels: Translation map for the selected language.

    Returns:
        Localised label with a count suffix when greater than one.
    """
    label = labels.get(key) or key
    suffix = f' x{count}' if count > 1 else ''
    return f'{label}{suffix}'


def _should_draw_label(detection: DetectionOverlay, label_count: int) -> bool:
    """Determine whether a detection label should be drawn.

    Args:
        detection: Normalised detection considered for label drawing.
        label_count: Number of labels already drawn on the frame.

    Returns:
        ``True`` when label configuration permits another label.
    """
    if not _overlay_draw_labels:
        return False
    if _overlay_label_warnings_only and not detection.is_warning:
        return False
    return _overlay_max_labels <= 0 or label_count < _overlay_max_labels


def _draw_detection(
    frame: np.ndarray,
    detection: DetectionOverlay,
    label_language: str,
    box_thickness: int,
    draw_label: bool = True,
) -> None:
    """Draw one detection box and optional label.

    Args:
        frame: Mutable BGR image array to annotate.
        detection: Normalised detection to draw.
        label_language: Canonical label language.
        box_thickness: Requested bounding-box line thickness.
        draw_label: Whether a label badge should be drawn.
    """
    if detection.class_name in {'safety-cone', 'cone'}:
        return

    x1, y1, x2, y2 = detection.bbox
    rgb = WARNING_RGB if detection.is_warning else _color_for_class(
        detection.class_name,
    )
    bgr = _rgb_to_bgr(rgb)
    thickness = max(1, int(box_thickness))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    if draw_label:
        _draw_label(frame, detection, rgb, label_language)


def _draw_label(
    frame: np.ndarray,
    detection: DetectionOverlay,
    rgb: tuple[int, int, int],
    label_language: str,
) -> None:
    """Draw a filled localised label badge for one detection.

    Args:
        frame: Mutable BGR image array to annotate.
        detection: Normalised detection to label.
        rgb: Badge colour in RGB order.
        label_language: Canonical label language.
    """
    x1, y1, x2, _ = detection.bbox
    label = _format_label(detection, label_language)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _font_scale(frame)
    thickness = max(1, _line_thickness(frame) - 1)
    padding_x = 6
    padding_y = 3
    text_width, text_height, baseline = _measure_label_text(
        label, frame, font, scale, thickness,
    )
    label_width = text_width + padding_x * 2
    label_height = text_height + baseline + padding_y * 2

    label_x1 = x1
    label_y1 = y1 - label_height
    if label_y1 < 0:
        label_y1 = y1
    label_x2 = min(frame.shape[1] - 1, label_x1 + label_width)
    if label_x2 >= frame.shape[1] - 1:
        label_x1 = max(0, frame.shape[1] - 1 - label_width)
        label_x2 = frame.shape[1] - 1
    label_y2 = min(frame.shape[0] - 1, label_y1 + label_height)

    label_roi = frame[label_y1:label_y2 + 1, label_x1:label_x2 + 1]
    if label_roi.size:
        cv2.rectangle(
            frame,
            (label_x1, label_y1),
            (label_x2, label_y2),
            _rgb_to_bgr(rgb),
            thickness=-1,
        )

    text_color = (255, 255, 255)
    text_origin = (
        label_x1 + padding_x,
        label_y1 + padding_y + text_height,
    )
    _draw_label_text(
        frame,
        label,
        text_origin,
        font,
        scale,
        text_color,
        thickness,
        (label_x1, label_y1, label_x2, label_y2),
    )


def _draw_polygons(
    frame: np.ndarray,
    polygon_specs: tuple[
        tuple[str, tuple[int, int, int], tuple[int, int, int], float],
        ...,
    ],
) -> None:
    """Draw polygons parsed from cached JSON strings.

    Args:
        frame: Mutable BGR image array to annotate.
        polygon_specs: JSON polygons with fill colour, border colour, and alpha.
    """
    for polygons_json, fill_rgb, stroke_rgb, fill_alpha in polygon_specs:
        polygons = _normalised_polygons_for_overlay(
            polygons_json,
            frame.shape[1],
            frame.shape[0],
        )
        if not polygons:
            continue
        _draw_polygon_rois(frame, polygons, fill_rgb, stroke_rgb, fill_alpha)


def _draw_polygon_data(
    frame: np.ndarray,
    polygon_specs: tuple[
        tuple[
            PolygonCollection,
            tuple[int, int, int],
            tuple[int, int, int],
            float,
        ],
        ...,
    ],
) -> None:
    """Draw polygons from already-decoded data.

    Args:
        frame: Mutable BGR image array to annotate.
        polygon_specs: Decoded polygons with fill colour, border colour, and alpha.
    """
    for polygon_data, fill_rgb, stroke_rgb, fill_alpha in polygon_specs:
        polygons = _normalised_polygons_from_data(
            polygon_data,
            frame.shape[1],
            frame.shape[0],
        )
        if not polygons:
            continue
        _draw_polygon_rois(frame, polygons, fill_rgb, stroke_rgb, fill_alpha)


def _draw_polygon_rois(
    frame: np.ndarray,
    polygons: tuple[np.ndarray, ...],
    fill_rgb: tuple[int, int, int],
    stroke_rgb: tuple[int, int, int],
    fill_alpha: float,
) -> None:
    """Draw polygon fills using bounded ROI copies.

    Args:
        frame: Mutable BGR image array to annotate.
        polygons: Clipped polygon points in pixel coordinates.
        fill_rgb: Polygon fill colour in RGB order.
        stroke_rgb: Polygon outline colour in RGB order.
        fill_alpha: Fill opacity between zero and one.
    """
    fill_bgr = _rgb_to_bgr(fill_rgb)
    stroke_bgr = _rgb_to_bgr(stroke_rgb)
    alpha = max(0.0, min(1.0, fill_alpha))
    for points in polygons:
        _blend_polygon_fill_roi(frame, points, fill_bgr, alpha)
        cv2.polylines(
            frame,
            [points],
            isClosed=True,
            color=stroke_bgr,
            thickness=3,
        )


def _blend_polygon_fill_roi(
    frame: np.ndarray,
    points: np.ndarray,
    fill_bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend a polygon fill into the smallest affected frame region.

    Args:
        frame: Mutable BGR image array to annotate.
        points: Clipped polygon points in pixel coordinates.
        fill_bgr: Fill colour in OpenCV BGR order.
        alpha: Fill opacity between zero and one.
    """
    if alpha <= 0:
        return
    x, y, width, height = cv2.boundingRect(points)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + width)
    y2 = min(frame.shape[0], y + height)
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    if not roi.size:
        return

    local_points = points.copy()
    local_points[:, 0] -= x1
    local_points[:, 1] -= y1
    fill = np.empty_like(roi)
    fill[:, :] = fill_bgr
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_points], 255)
    blended = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)
    cv2.copyTo(blended, mask, roi)


@lru_cache(maxsize=_overlay_parse_cache_size)
def _normalised_polygons_for_overlay(
    polygons_json: str,
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, ...]:
    """Return cached pixel polygons parsed from JSON.

    Args:
        polygons_json: JSON polygon collection.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.

    Returns:
        Valid clipped polygon arrays cached by geometry and frame size.
    """
    data = _parse_polygon_collection(polygons_json)

    polygons: list[np.ndarray] = []
    for polygon in data:
        points = _normalise_polygon(polygon, frame_width, frame_height)
        if points is not None:
            polygons.append(points)
    return tuple(polygons)


def _normalised_polygons_from_data(
    data: PolygonCollection,
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, ...]:
    """Normalise pixel polygons from decoded data.

    Args:
        data: Decoded polygon collection.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.

    Returns:
        Valid clipped polygon arrays.
    """
    polygons: list[np.ndarray] = []
    for polygon in data:
        points = _normalise_polygon(polygon, frame_width, frame_height)
        if points is not None:
            polygons.append(points)
    return tuple(polygons)


def _normalise_polygon(
    polygon: PolygonCoordinates,
    frame_width: int,
    frame_height: int,
) -> np.ndarray | None:
    """Normalise one polygon to clipped pixel coordinates.

    Args:
        polygon: Polygon coordinates in normalised or pixel form.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.

    Returns:
        Integer OpenCV polygon points, or ``None`` when no points are valid.
    """
    points = [(float(point[0]), float(point[1])) for point in polygon]

    if _points_look_normalized(points):
        points = [
            (x * frame_width, y * frame_height)
            for x, y in points
        ]

    clipped = [
        (
            max(0, min(int(round(x)), frame_width - 1)),
            max(0, min(int(round(y)), frame_height - 1)),
        )
        for x, y in points
    ]
    return np.array(clipped, dtype=np.int32)


def _points_look_normalized(points: list[tuple[float, float]]) -> bool:
    """Determine whether polygon points appear normalised to zero through one.

    Args:
        points: Polygon points to inspect.

    Returns:
        ``True`` when every point lies in the inclusive unit square.
    """
    return all(
        0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        for x, y in points
    )


def _format_label(
    detection: DetectionOverlay,
    label_language: str,
) -> str:
    """Format a localised detection label for display.

    Args:
        detection: Normalised detection to label.
        label_language: Requested label language.

    Returns:
        Localised display label for the detection class.
    """
    return _translate_class_name(
        detection.class_name,
        label_language,
    )


def _translate_class_name(class_name: str, label_language: str) -> str:
    """Translate a detection class name for the selected language.

    Args:
        class_name: Canonical or alias detector class name.
        label_language: Requested label language.

    Returns:
        Localised class label with English as a final fallback.
    """
    language = normalise_label_language(label_language)
    key = class_name.lower()
    labels = CLASS_LABELS.get(language, CLASS_LABELS['en'])
    return labels.get(key) or CLASS_LABELS['en'].get(key) or class_name


def _measure_label_text(
    label: str,
    frame: np.ndarray,
    font: int,
    scale: float,
    thickness: int,
) -> tuple[int, int, int]:
    """Measure label text using Pillow when OpenCV cannot render it.

    Args:
        label: Localised text to measure.
        frame: Target BGR image array.
        font: OpenCV font identifier for ASCII text.
        scale: OpenCV font scale for ASCII text.
        thickness: OpenCV text thickness.

    Returns:
        Text width, height, and baseline in pixels.
    """
    if _needs_pillow_text(label):
        rendered = _render_pillow_text_bitmap(
            label,
            _font_pixel_size(frame),
            (255, 255, 255),
        )
        if rendered is not None:
            return rendered.width, rendered.height, max(2, thickness)

    (width, height), baseline = cv2.getTextSize(
        label, font, scale, thickness,
    )
    return width, height, baseline


def _draw_label_text(
    frame: np.ndarray,
    label: str,
    text_origin: tuple[int, int],
    font: int,
    scale: float,
    text_color: tuple[int, int, int],
    thickness: int,
    text_area: tuple[int, int, int, int],
) -> None:
    """Draw label text using the appropriate text renderer.

    Args:
        frame: Mutable BGR image array to annotate.
        label: Localised text to draw.
        text_origin: Text baseline origin in pixels.
        font: OpenCV font identifier for ASCII text.
        scale: OpenCV font scale for ASCII text.
        text_color: Text colour in BGR order.
        thickness: OpenCV text thickness.
        text_area: Bounding area that clips the text bitmap.
    """
    if _needs_pillow_text(label):
        _draw_pillow_text(
            frame,
            label,
            text_origin,
            text_color,
            text_area,
        )
        return

    cv2.putText(
        frame,
        label,
        text_origin,
        font,
        scale,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        text_origin,
        font,
        scale,
        text_color,
        max(1, thickness),
        cv2.LINE_AA,
    )


def _needs_pillow_text(label: str) -> bool:
    """Determine whether OpenCV's Hershey font is unsafe for a label.

    Args:
        label: Localised text to inspect.

    Returns:
        ``True`` when non-ASCII glyph rendering requires Pillow.
    """
    return any(ord(char) > 127 for char in label)


def _font_pixel_size(frame: np.ndarray) -> int:
    """Calculate a readable Pillow font size for a frame.

    Args:
        frame: Target BGR image array.

    Returns:
        Bounded pixel font size scaled to the smaller frame dimension.
    """
    min_side = min(frame.shape[:2])
    return max(14, min(32, round(min_side / 32)))


@lru_cache(maxsize=32)
def _load_overlay_font(
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
    """Load a font that supports multilingual overlay labels.

    Args:
        size: Requested font size in pixels.

    Returns:
        First usable configured or system font, or ``None`` when unavailable.
    """
    configured = os.getenv('STREAMING_OVERLAY_FONT_PATH', '').strip()
    candidates = [
        configured,
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    # Prefer the explicit deployment font, then common Noto fonts with broad
    # CJK and Thai coverage before accepting a system fallback.
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_pillow_text(
    frame: np.ndarray,
    label: str,
    text_origin: tuple[int, int],
    text_color: tuple[int, int, int],
    text_area: tuple[int, int, int, int],
) -> None:
    """Draw non-ASCII text with cached Pillow bitmaps.

    Args:
        frame: Mutable BGR image array to annotate.
        label: Localised non-ASCII text to draw.
        text_origin: Requested text baseline origin in pixels.
        text_color: Text colour in BGR order.
        text_area: Bounding area that clips the rendered text.
    """
    rendered = _render_pillow_text_bitmap(
        label,
        _font_pixel_size(frame),
        text_color,
    )
    if rendered is None or rendered.width <= 0 or rendered.height <= 0:
        return

    x1, y1, x2, y2 = text_area
    text_x = max(x1, min(text_origin[0], max(x1, x2 - rendered.width)))
    text_y = max(
        y1,
        min(text_origin[1] - rendered.height, max(y1, y2 - rendered.height)),
    )
    _blend_bgra_roi(frame, rendered.bgra, text_x, text_y)


@lru_cache(maxsize=_overlay_text_bitmap_cache_size)
def _render_pillow_text_bitmap(
    label: str,
    font_size: int,
    text_color: tuple[int, int, int],
) -> _RenderedTextBitmap | None:
    """Render one text string once for subsequent ROI alpha blending.

    Args:
        label: Localised text to render.
        font_size: Requested font size in pixels.
        text_color: Text colour in BGR order.

    Returns:
        Cached BGRA text bitmap, or ``None`` when no font is available.
    """
    font = _load_overlay_font(font_size)
    if font is None:
        return None

    stroke_width = 2
    measure_image = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(measure_image)
    bbox = measure_draw.textbbox(
        (0, 0),
        label,
        font=font,
        stroke_width=stroke_width,
    )
    width = max(1, int(round(bbox[2] - bbox[0])))
    height = max(1, int(round(bbox[3] - bbox[1])))
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    rgb_color = (text_color[2], text_color[1], text_color[0])
    draw.text(
        (-bbox[0], -bbox[1]),
        label,
        font=font,
        fill=(*rgb_color, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )
    rgba = np.asarray(image, dtype=np.uint8)
    return _RenderedTextBitmap(
        bgra=rgba[:, :, [2, 1, 0, 3]].copy(),
        width=width,
        height=height,
    )


def _blend_bgra_roi(
    frame: np.ndarray,
    bgra: np.ndarray,
    x: int,
    y: int,
) -> None:
    """Alpha-blend cached text into only the affected frame rectangle.

    Args:
        frame: Mutable BGR image array to annotate.
        bgra: Cached four-channel text bitmap in OpenCV channel order.
        x: Requested destination left coordinate.
        y: Requested destination top coordinate.
    """
    frame_height, frame_width = frame.shape[:2]
    text_height, text_width = bgra.shape[:2]
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(frame_width, x + text_width)
    dst_y2 = min(frame_height, y + text_height)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return

    src_x1 = dst_x1 - x
    src_y1 = dst_y1 - y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    overlay = bgra[src_y1:src_y2, src_x1:src_x2]
    alpha = overlay[:, :, 3:4].astype(np.uint16)
    if not np.any(alpha):
        return
    inv_alpha = 255 - alpha
    roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
    blended = (
        overlay[:, :, :3].astype(np.uint16) * alpha
        + roi.astype(np.uint16) * inv_alpha
        + 127
    ) // 255
    roi[:, :] = blended.astype(np.uint8)


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    """Return a stable RGB colour for a detection class.

    Args:
        class_name: Canonical or alias detector class name.

    Returns:
        Configured semantic colour or deterministic vivid fallback colour.
    """
    key = class_name.lower()
    if key in CLASS_COLORS_RGB:
        return CLASS_COLORS_RGB[key]

    digest = hashlib.md5(key.encode('utf-8')).digest()
    return (
        80 + digest[0] % 176,
        80 + digest[1] % 176,
        80 + digest[2] % 176,
    )


def _line_thickness(frame: np.ndarray) -> int:
    """Calculate box line thickness for a frame size.

    Args:
        frame: Target BGR image array.

    Returns:
        Bounded line thickness in pixels.
    """
    min_side = min(frame.shape[:2])
    return max(2, min(4, round(min_side / 360)))


def _font_scale(frame: np.ndarray) -> float:
    """Calculate OpenCV font scale for a frame size.

    Args:
        frame: Target BGR image array.

    Returns:
        Bounded OpenCV font scale.
    """
    min_side = min(frame.shape[:2])
    return max(0.45, min(0.75, min_side / 1000))


def _is_bright(rgb: tuple[int, int, int]) -> bool:
    """Determine whether an RGB colour is visually bright.

    Args:
        rgb: Colour in red, green, blue order.

    Returns:
        ``True`` when luminance exceeds the legibility threshold.
    """
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance > 150


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an RGB colour tuple to OpenCV BGR order.

    Args:
        rgb: Colour in red, green, blue order.

    Returns:
        Same colour in blue, green, red order for OpenCV.
    """
    r, g, b = rgb
    return b, g, r

"""Localised labels and semantic colours for live overlay presentation."""
from __future__ import annotations

from typing import TYPE_CHECKING

from examples.shared.class_colors import published_color

if TYPE_CHECKING:
    from examples.streaming_web.overlay_models import DetectionOverlay

WARNING_RGB: tuple[int, int, int] = (244, 67, 54)

CLASS_LABELS: dict[str, dict[str, str]] = {
    'en': {
        'helmet': 'hardhat',
        'mask': 'mask',
        'no_helmet': 'no hardhat',
        'no_mask': 'no mask',
        'no_safety_vest': 'no safety vest',
        'person': 'person',
        'safety_cone': 'safety cone',
        'safety_vest': 'safety vest',
        'machinery': 'machinery',
        'utility_pole': 'utility pole',
        'vehicle': 'vehicle',
        'danger': 'danger',
        'unknown': 'unknown',
    },
    'zh-TW': {
        'helmet': '安全帽',
        'mask': '口罩',
        'no_helmet': '未戴安全帽',
        'no_mask': '未戴口罩',
        'no_safety_vest': '未穿安全背心',
        'person': '人員',
        'safety_cone': '交通錐',
        'safety_vest': '安全背心',
        'machinery': '機具',
        'utility_pole': '電桿',
        'vehicle': '車輛',
        'danger': '危險',
        'unknown': '未知',
    },
    'zh-CN': {
        'helmet': '安全帽',
        'mask': '口罩',
        'no_helmet': '未戴安全帽',
        'no_mask': '未戴口罩',
        'no_safety_vest': '未穿安全背心',
        'person': '人员',
        'safety_cone': '交通锥',
        'safety_vest': '安全背心',
        'machinery': '机具',
        'utility_pole': '电杆',
        'vehicle': '车辆',
        'danger': '危险',
        'unknown': '未知',
    },
    'ja': {
        'helmet': 'ヘルメット',
        'mask': 'マスク',
        'no_helmet': 'ヘルメットなし',
        'no_mask': 'マスクなし',
        'no_safety_vest': '安全ベストなし',
        'person': '作業員',
        'safety_cone': 'カラーコーン',
        'safety_vest': '安全ベスト',
        'machinery': '重機',
        'utility_pole': '電柱',
        'vehicle': '車両',
        'danger': '危険',
        'unknown': '不明',
    },
    'vi': {
        'helmet': 'mu bao ho',
        'mask': 'khau trang',
        'no_helmet': 'khong mu bao ho',
        'no_mask': 'khong khau trang',
        'no_safety_vest': 'khong ao bao ho',
        'person': 'nguoi',
        'safety_cone': 'coc an toan',
        'safety_vest': 'ao bao ho',
        'machinery': 'may moc',
        'utility_pole': 'cot dien',
        'vehicle': 'xe',
        'danger': 'nguy hiem',
        'unknown': 'khong ro',
    },
    'id': {
        'helmet': 'helm',
        'mask': 'masker',
        'no_helmet': 'tanpa helm',
        'no_mask': 'tanpa masker',
        'no_safety_vest': 'tanpa rompi',
        'person': 'orang',
        'safety_cone': 'kerucut',
        'safety_vest': 'rompi',
        'machinery': 'mesin',
        'utility_pole': 'tiang listrik',
        'vehicle': 'kendaraan',
        'danger': 'bahaya',
        'unknown': 'tidak dikenal',
    },
    'fr': {
        'helmet': 'casque',
        'mask': 'masque',
        'no_helmet': 'sans casque',
        'no_mask': 'sans masque',
        'no_safety_vest': 'sans gilet',
        'person': 'personne',
        'safety_cone': 'cone',
        'safety_vest': 'gilet',
        'machinery': 'machine',
        'utility_pole': 'poteau',
        'vehicle': 'vehicule',
        'danger': 'danger',
        'unknown': 'inconnu',
    },
    'th': {
        'helmet': 'หมวกนิรภัย',
        'mask': 'หน้ากาก',
        'no_helmet': 'ไม่สวมหมวก',
        'no_mask': 'ไม่สวมหน้ากาก',
        'no_safety_vest': 'ไม่สวมเสื้อสะท้อนแสง',
        'person': 'คนงาน',
        'safety_cone': 'กรวยจราจร',
        'safety_vest': 'เสื้อสะท้อนแสง',
        'machinery': 'เครื่องจักร',
        'utility_pole': 'เสาไฟ',
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
    'no_helmet': 'warning_no_hardhat',
    'no_mask': 'warning_no_mask',
    'no_safety_vest': 'warning_no_safety_vest',
}


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
    return detection.display_name or _translate_class_name(
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


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    """Return a stable RGB colour for a detection class.

    Args:
        class_name: Canonical or alias detector class name.

    Returns:
        Configured semantic colour or deterministic vivid fallback colour.
    """
    key = class_name.lower()
    red, green, blue = bytes.fromhex(published_color(key)[1:])
    return red, green, blue


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

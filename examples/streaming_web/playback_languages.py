from __future__ import annotations

import os

from examples.local_notification_server.lang_config import LANGUAGES
from examples.streaming_web.overlay_renderer import CLASS_LABELS
from examples.streaming_web.overlay_renderer import LANGUAGE_ALIASES
from examples.streaming_web.overlay_renderer import SUPPORTED_LABEL_LANGUAGES
from examples.streaming_web.overlay_renderer import WARNING_LABELS
from examples.streaming_web.schemas import OverlayLanguageInfo


def _allowed_overlay_languages() -> tuple[str, ...]:
    """Return the configured overlay languages enabled for playback.

    Returns:
        Ordered unique canonical language codes.

    Raises:
        ValueError: If configuration is empty or names unsupported languages.
    """
    configured = tuple(
        dict.fromkeys(
            language.strip()
            for language in os.getenv(
                'MEDIA_OVERLAY_ALLOWED_LANGUAGES',
                ','.join(SUPPORTED_LABEL_LANGUAGES),
            ).split(',')
            if language.strip()
        ),
    )
    if not configured or set(configured) - set(SUPPORTED_LABEL_LANGUAGES):
        raise ValueError(
            'MEDIA_OVERLAY_ALLOWED_LANGUAGES must contain supported codes',
        )
    return configured


OVERLAY_LANGUAGE_DETAILS: dict[str, dict[str, str | list[str]]] = {
    'zh-TW': {
        'name': 'Traditional Chinese',
        'native_name': '繁體中文',
        'aliases': ['zh', 'zh-Hant', 'zh_TW', 'zh-HK'],
    },
    'en': {
        'name': 'English',
        'native_name': 'English',
        'aliases': ['en-US', 'en-GB'],
    },
    'zh-CN': {
        'name': 'Simplified Chinese',
        'native_name': '简体中文',
        'aliases': ['zh-Hans', 'zh_CN', 'zh-SG'],
    },
    'ja': {
        'name': 'Japanese',
        'native_name': '日本語',
        'aliases': ['jp', 'ja-JP'],
    },
    'vi': {
        'name': 'Vietnamese',
        'native_name': 'Tiếng Việt',
        'aliases': ['vi-VN'],
    },
    'id': {
        'name': 'Indonesian',
        'native_name': 'Bahasa Indonesia',
        'aliases': ['id-ID'],
    },
    'fr': {
        'name': 'French',
        'native_name': 'Français',
        'aliases': ['fr-FR', 'fr-CA'],
    },
    'th': {
        'name': 'Thai',
        'native_name': 'ไทย',
        'aliases': ['th-TH'],
    },
}
OVERLAY_TO_NOTIFICATION_LANGUAGE: dict[str, str] = {
    'en': 'en-GB',
    'zh-TW': 'zh-TW',
    'zh-CN': 'zh-CN',
    'ja': 'ja-JP',
    'vi': 'vi-VN',
    'id': 'id-ID',
    'fr': 'fr-FR',
    'th': 'th-TH',
}


def _default_overlay_language() -> str:
    """Return the configured default overlay language.

    Returns:
        Canonical language code selected by default.

    Raises:
        ValueError: If the configured default is not allowed.
    """
    language = os.getenv('MEDIA_DEFAULT_OVERLAY_LANGUAGE', 'zh-TW')
    if language not in _allowed_overlay_languages():
        raise ValueError(
            'MEDIA_DEFAULT_OVERLAY_LANGUAGE must be an allowed language',
        )
    return language


def _language_alias_map() -> dict[str, str]:
    """Build normalised language aliases for API consumers.

    Returns:
        Mapping from recognised aliases to canonical overlay language codes.
    """
    aliases = dict(LANGUAGE_ALIASES)
    for code, details in OVERLAY_LANGUAGE_DETAILS.items():
        aliases[code] = code
        for alias in details['aliases']:
            aliases[alias] = code
            aliases[alias.lower()] = code
    return aliases


def _notification_language_code(code: str) -> str:
    """Map an overlay language to its notification language code.

    Args:
        code: Canonical overlay language code.

    Returns:
        Corresponding notification-template language code.
    """
    return OVERLAY_TO_NOTIFICATION_LANGUAGE[code]


def _overlay_language_options(
    allowed_languages: tuple[str, ...] | None = None,
) -> list[OverlayLanguageInfo]:
    """Build language-option metadata for the frontend.

    Args:
        allowed_languages: Optional prevalidated ordered language codes.

    Returns:
        Display, label, and notification details for each allowed language.
    """
    languages: list[OverlayLanguageInfo] = []
    codes = (
        _allowed_overlay_languages()
        if allowed_languages is None
        else allowed_languages
    )
    for code in codes:
        details = OVERLAY_LANGUAGE_DETAILS[code]
        notification_code = _notification_language_code(code)
        languages.append(
            OverlayLanguageInfo(
                code=code,
                notification_code=notification_code,
                display_name=str(details['name']),
                native_name=str(details['native_name']),
                is_default=code == _default_overlay_language(),
                class_labels=CLASS_LABELS[code],
                warning_labels=WARNING_LABELS[code],
                notification_templates=LANGUAGES[notification_code],
            ),
        )
    return languages

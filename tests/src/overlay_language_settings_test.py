from __future__ import annotations

from unittest.mock import patch

import pytest

from examples.streaming_web.playback_languages import (
    _allowed_overlay_languages,
)
from examples.streaming_web.playback_languages import _default_overlay_language
from src.settings.overlay_language import get_overlay_language_policy
from src.stream_overlay_frames import (
    _allowed_overlay_languages as worker_languages,
)


@pytest.mark.parametrize(
    'allowed,expected',
    [
        ('zh,zh_TW,en-US', ('zh-TW', 'en')),
        ('fr-FR,fr-CA', ('fr',)),
    ],
)
def test_api_and_worker_share_canonical_language_policy(allowed, expected):
    with patch.dict(
        'os.environ',
        {
            'MEDIA_OVERLAY_ALLOWED_LANGUAGES': allowed,
            'MEDIA_DEFAULT_OVERLAY_LANGUAGE': expected[0],
        },
    ):
        policy = get_overlay_language_policy()
        assert worker_languages() == _allowed_overlay_languages() == expected
        assert _default_overlay_language() == expected[0]
        assert get_overlay_language_policy() is policy
    assert policy.allowed == expected


@pytest.mark.parametrize('allowed', ['', 'not-a-language', 'en,invalid'])
def test_worker_and_api_reject_the_same_invalid_configuration(allowed):
    with patch.dict(
        'os.environ',
        {'MEDIA_OVERLAY_ALLOWED_LANGUAGES': allowed},
    ):
        for load in (worker_languages, _allowed_overlay_languages):
            with pytest.raises(ValueError):
                load()

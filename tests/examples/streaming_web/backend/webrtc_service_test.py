from __future__ import annotations

import unittest
from unittest.mock import patch

from examples.streaming_web.backend.webrtc_service import (
    get_public_ice_servers,
)


class WebRtcServiceTest(unittest.TestCase):
    """Tests for public WebRTC ICE server configuration."""

    @patch.dict(
        'os.environ',
        {
            'STREAMING_WEBRTC_STUN_URLS': '',
            'STREAMING_WEBRTC_TURN_URLS': '',
        },
        clear=False,
    )
    def test_get_public_ice_servers_uses_default_stun(self) -> None:
        """Exercise this test."""
        self.assertEqual(
            get_public_ice_servers(),
            [{'urls': ['stun:stun.l.google.com:19302']}],
        )

    @patch.dict(
        'os.environ',
        {
            'STREAMING_WEBRTC_STUN_URLS': 'stun:a, stun:b',
            'STREAMING_WEBRTC_TURN_URLS': 'turn:relay',
            'STREAMING_WEBRTC_TURN_USERNAME': 'user',
            'STREAMING_WEBRTC_TURN_CREDENTIAL': 'secret',
            'STREAMING_WEBRTC_TURN_SHARED_SECRET': '',
        },
        clear=False,
    )
    def test_get_public_ice_servers_includes_static_turn(self) -> None:
        """Exercise this test."""
        self.assertEqual(
            get_public_ice_servers(),
            [
                {'urls': ['stun:a', 'stun:b']},
                {
                    'urls': ['turn:relay'],
                    'username': 'user',
                    'credential': 'secret',
                },
            ],
        )


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.streaming_web.webrtc_service import _build_turn_rest_credential
from examples.streaming_web.webrtc_service import get_public_ice_servers


class WebRtcServiceTest(unittest.TestCase):
    """Tests for public WebRTC ICE server configuration."""

    @patch('examples.streaming_web.webrtc_service.time.time')
    def test_build_turn_rest_credential_matches_coturn_format(
        self,
        mock_time: MagicMock,
    ) -> None:
        """Generate the HMAC-SHA1 credential required by coturn."""
        mock_time.return_value = 1_700_000_000

        self.assertEqual(
            _build_turn_rest_credential(
                shared_secret='turn-shared-secret',
                user_id='alice',
                ttl_seconds=600,
            ),
            ('1700000600:alice', 'LHzsQVfbI581fwl4uDb0pmdZnnM='),
        )

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

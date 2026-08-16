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

    @patch.dict('os.environ', {}, clear=True)
    def test_get_public_ice_servers_requires_stun_configuration(self) -> None:
        """STUN endpoints must be configured explicitly."""
        with self.assertRaises(KeyError):
            get_public_ice_servers('alice')

    @patch.dict(
        'os.environ',
        {
            'STREAMING_WEBRTC_STUN_URLS': 'stun:a, stun:b',
        },
        clear=True,
    )
    def test_get_public_ice_servers_uses_configured_stun(self) -> None:
        """Return the configured STUN endpoints without TURN."""
        self.assertEqual(
            get_public_ice_servers('alice'),
            [{'urls': ['stun:a', 'stun:b']}],
        )

    @patch.dict(
        'os.environ',
        {
            'STREAMING_WEBRTC_STUN_URLS': 'stun:relay',
            'STREAMING_WEBRTC_TURN_URLS': 'turn:relay',
            'STREAMING_WEBRTC_TURN_SHARED_SECRET': 'turn-secret',
            'STREAMING_WEBRTC_TURN_TTL_SECONDS': '600',
        },
        clear=True,
    )
    @patch(
        'examples.streaming_web.webrtc_service.time.time',
        return_value=1_700_000_000,
    )
    def test_get_public_ice_servers_uses_turn_rest_credentials(
        self,
        mock_time: MagicMock,
    ) -> None:
        """TURN REST credentials use the explicitly configured TTL."""
        servers = get_public_ice_servers('alice')

        self.assertEqual(servers[0], {'urls': ['stun:relay']})
        self.assertEqual(servers[1]['username'], '1700000600:alice')
        self.assertEqual(
            servers[1]['credential'],
            'XHKohCPbA/hq/kAmxP3/ALxaTSI=',
        )
        mock_time.assert_called_once()

    @patch.dict(
        'os.environ',
        {
            'STREAMING_WEBRTC_STUN_URLS': 'stun:relay',
            'STREAMING_WEBRTC_TURN_URLS': 'turn:relay',
            'STREAMING_WEBRTC_TURN_SHARED_SECRET': 'turn-secret',
            'STREAMING_WEBRTC_TURN_TTL_SECONDS': 'not-an-integer',
        },
        clear=True,
    )
    def test_get_public_ice_servers_rejects_invalid_turn_ttl(self) -> None:
        """Invalid TURN TTL configuration must fail rather than fall back."""
        with self.assertRaises(ValueError):
            get_public_ice_servers('alice')


if __name__ == '__main__':
    unittest.main()

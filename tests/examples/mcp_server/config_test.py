from __future__ import annotations

import os
import unittest

from examples.mcp_server import config as cfg


class TransportConfigTests(unittest.TestCase):
    """Unit tests for get_transport_config builder function."""

    def setUp(self) -> None:
        """Ensure environment starts clean for each test."""
        self._original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith('MCP_'):
                os.environ.pop(key, None)

    def tearDown(self) -> None:
        """Restore environment variables after tests."""
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_default_transport_config_values(self) -> None:
        """When no variables set, defaults should match DEFAULT_ENV."""
        cfg_obj = cfg.get_transport_config()
        self.assertEqual(cfg_obj['transport'], 'streamable-http')
        self.assertEqual(cfg_obj['host'], '127.0.0.1')
        self.assertEqual(cfg_obj['port'], 8092)
        self.assertEqual(cfg_obj['path'], '/mcp')
        self.assertEqual(cfg_obj['sse_path'], '/sse')
        self.assertFalse(cfg_obj['debug'])

    def test_custom_transport_config_from_env(self) -> None:
        """Custom environment variables should override defaults."""
        os.environ.update(
            {
                'MCP_TRANSPORT': 'sse',
                'MCP_HOST': '127.0.0.1',
                'MCP_PORT': '9000',
                'MCP_PATH': '/custom',
                'MCP_SSE_PATH': '/events',
                'MCP_DEBUG': 'true',
            },
        )
        cfg_obj = cfg.get_transport_config()
        self.assertEqual(cfg_obj['transport'], 'sse')
        self.assertEqual(cfg_obj['host'], '127.0.0.1')
        self.assertEqual(cfg_obj['port'], 9000)
        self.assertEqual(cfg_obj['path'], '/custom')
        self.assertEqual(cfg_obj['sse_path'], '/events')
        self.assertTrue(cfg_obj['debug'])

    def test_invalid_transport_is_rejected(self):
        os.environ['MCP_TRANSPORT'] = 'invalid'
        with self.assertRaises(ValueError):
            cfg.get_transport_config()

    def test_debug_false_for_falsey_values(self) -> None:
        """False-like debug strings should evaluate as False."""
        for val in ('0', 'false', 'no'):
            with self.subTest(val=val):
                os.environ['MCP_DEBUG'] = val
                result = cfg.get_transport_config()
                self.assertFalse(
                    result['debug'],
                    f"{val} should yield debug=False",
                )


if __name__ == '__main__':
    unittest.main()

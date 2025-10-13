from __future__ import annotations

import os
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from examples.mcp_server import config as cfg


class EnvVarHelpersTests(unittest.TestCase):
    """Unit tests for individual environment helper functions."""

    def setUp(self) -> None:
        """Clear environment variables before each test for isolation."""
        self._original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith('MCP_') or key.startswith('TEST_'):
                os.environ.pop(key, None)

    def tearDown(self) -> None:
        """Restore original environment variables after each test."""
        os.environ.clear()
        os.environ.update(self._original_env)

    # ------------------------------------------------------------------
    # get_env_var
    # ------------------------------------------------------------------

    def test_get_env_var_returns_value_when_set(self) -> None:
        """get_env_var should return the variable value if defined."""
        os.environ['TEST_KEY'] = 'abc'
        value = cfg.get_env_var('TEST_KEY', 'default')
        self.assertEqual(value, 'abc')

    def test_get_env_var_returns_default_when_unset(self) -> None:
        """get_env_var should fall back to default or DEFAULT_ENV."""
        value = cfg.get_env_var('NON_EXIST', 'xyz')
        self.assertEqual(value, 'xyz')

    # ------------------------------------------------------------------
    # get_env_bool
    # ------------------------------------------------------------------

    def test_get_env_bool_recognises_truthy_values(self) -> None:
        """get_env_bool should interpret 1/true/yes/on as True."""
        truthy = ('1', 'true', 'yes', 'on')
        for val in truthy:
            with self.subTest(val=val):
                os.environ['TEST_BOOL'] = val
                self.assertTrue(cfg.get_env_bool('TEST_BOOL'))

    def test_get_env_bool_defaults_false(self) -> None:
        """get_env_bool should return False if variable is missing."""
        self.assertFalse(cfg.get_env_bool('MISSING_BOOL'))

    def test_get_env_bool_respects_default_flag(self) -> None:
        """get_env_bool should use default=True if not set."""
        self.assertTrue(cfg.get_env_bool('UNSET_BOOL', default=True))

    # ------------------------------------------------------------------
    # get_env_int
    # ------------------------------------------------------------------

    def test_get_env_int_parses_valid_integer(self) -> None:
        """get_env_int should correctly parse integers."""
        os.environ['TEST_INT'] = '42'
        self.assertEqual(cfg.get_env_int('TEST_INT'), 42)

    def test_get_env_int_returns_default_on_invalid(self) -> None:
        """get_env_int should return default if value cannot be parsed."""
        os.environ['TEST_INT'] = 'notanint'
        self.assertEqual(cfg.get_env_int('TEST_INT', 10), 10)

    def test_get_env_int_parses_negative_integer(self) -> None:
        """get_env_int should handle negative integer strings."""
        os.environ['TEST_INT'] = '-7'
        self.assertEqual(cfg.get_env_int('TEST_INT'), -7)

    # ------------------------------------------------------------------
    # get_env_float
    # ------------------------------------------------------------------

    def test_get_env_float_parses_valid_float(self) -> None:
        """get_env_float should correctly parse float strings."""
        os.environ['TEST_FLOAT'] = '3.1415'
        self.assertAlmostEqual(cfg.get_env_float('TEST_FLOAT'), 3.1415)

    def test_get_env_float_returns_default_on_invalid(self) -> None:
        """get_env_float should fall back to default for invalid input."""
        os.environ['TEST_FLOAT'] = 'abc'
        self.assertEqual(cfg.get_env_float('TEST_FLOAT', 2.0), 2.0)

    # ------------------------------------------------------------------
    # get_env_list
    # ------------------------------------------------------------------

    def test_get_env_list_splits_and_trims_values(self) -> None:
        """get_env_list should split values by separator and trim
        whitespace.
        """
        os.environ['TEST_LIST'] = ' a , b ,c '
        result = cfg.get_env_list('TEST_LIST')
        self.assertEqual(result, ['a', 'b', 'c'])

    def test_get_env_list_returns_default_when_unset(self) -> None:
        """get_env_list should return default list when variable unset."""
        result = cfg.get_env_list('NO_LIST', default=['x'])
        self.assertEqual(result, ['x'])

    def test_get_env_list_returns_empty_when_unset_no_default(self) -> None:
        """get_env_list should return empty list when unset and no default."""
        self.assertEqual(cfg.get_env_list('MISSING_LIST'), [])


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
        self.assertEqual(cfg_obj['host'], '0.0.0.0')
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

    @patch('builtins.print')
    def test_invalid_transport_falls_back_to_stdio(
        self,
        mock_print: Mock,
    ) -> None:
        """Invalid MCP_TRANSPORT should fall back to 'stdio' with warning."""
        os.environ['MCP_TRANSPORT'] = 'invalid'
        result = cfg.get_transport_config()
        self.assertEqual(result['transport'], 'stdio')
        mock_print.assert_called_once()
        self.assertIn(
            'Invalid transport',
            mock_print.call_args[0][0],
        )

    def test_debug_false_for_falsey_values(self) -> None:
        """False-like debug strings should evaluate as False."""
        for val in ('0', 'false', 'no'):
            with self.subTest(val=val):
                os.environ['MCP_DEBUG'] = val
                result = cfg.get_transport_config()
                self.assertFalse(
                    result['debug'], f"{val} should yield debug=False",
                )


if __name__ == '__main__':
    unittest.main()

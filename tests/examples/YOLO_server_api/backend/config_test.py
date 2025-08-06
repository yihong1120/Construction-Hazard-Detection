from __future__ import annotations

import inspect
import unittest

from examples.YOLO_server_api.backend import config


class TestConfig(unittest.TestCase):
    """
    Unit tests for the config module and USE_TENSORRT variable.
    """

    def test_use_tensorrt_exists(self) -> None:
        """Test that the USE_TENSORRT variable exists in config.

        Ensures the config module contains the USE_TENSORRT attribute.
        """
        # Check existence of USE_TENSORRT
        self.assertTrue(hasattr(config, 'USE_TENSORRT'))

    def test_use_tensorrt_type(self) -> None:
        """Test that USE_TENSORRT is a boolean value.

        Ensures the type of USE_TENSORRT is bool.
        """
        # Check type of USE_TENSORRT
        self.assertIsInstance(config.USE_TENSORRT, bool)

    def test_use_tensorrt_default(self) -> None:
        """Test that the default value of USE_TENSORRT is True.

        Ensures the default configuration uses TensorRT.
        """
        # Check default value
        self.assertTrue(config.USE_TENSORRT)

    def test_use_tensorrt_can_be_false(self) -> None:
        """Test that USE_TENSORRT can be set to False.

        Ensures the variable is mutable and can be disabled for testing.
        """
        # Save original value and restore after test
        original_value: bool = config.USE_TENSORRT
        try:
            config.USE_TENSORRT = False
            self.assertFalse(config.USE_TENSORRT)
        finally:
            config.USE_TENSORRT = original_value

    def test_use_tensorrt_docstring(self) -> None:
        """Test that USE_TENSORRT has a comment in the source code.

        Ensures the source code contains documentation for USE_TENSORRT.
        """
        # Check for comment and variable in source
        source: str = inspect.getsource(config)
        self.assertIn('USE_TENSORRT', source)
        self.assertIn('根據需求設置', source)


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.YOLO_server_api.backend.config \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/config_test.py
"""

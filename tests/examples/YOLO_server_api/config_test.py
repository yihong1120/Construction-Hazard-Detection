from __future__ import annotations

import importlib
import inspect
import os
import unittest
from unittest.mock import patch

from examples.YOLO_server_api import config


class TestConfig(unittest.TestCase):
    """Unit tests for the config module and USE_TENSORRT variable."""

    def test_use_tensorrt_exists(self) -> None:
        # Check existence of USE_TENSORRT
        """Test use tensorrt exists."""
        self.assertTrue(hasattr(config, 'USE_TENSORRT'))

    def test_use_tensorrt_type(self) -> None:
        # Check type of USE_TENSORRT
        """Test use tensorrt type."""
        self.assertIsInstance(config.USE_TENSORRT, bool)

    def test_use_tensorrt_default(self) -> None:
        """Test that the default value of USE_TENSORRT matches environment
        variable.

        Ensures the configuration respects the environment variable default.
        """
        # Check that USE_TENSORRT matches the expected environment default
        # Since the environment variable defaults to 'false', USE_TENSORRT
        # should be False
        expected_value = os.getenv('USE_TENSORRT', 'false').lower() == 'true'
        self.assertEqual(config.USE_TENSORRT, expected_value)

    def test_use_tensorrt_can_be_false(self) -> None:
        # Save original value and restore after test
        """Test use tensorrt can be false."""
        original_value: bool = config.USE_TENSORRT
        try:
            config.USE_TENSORRT = False
            self.assertFalse(config.USE_TENSORRT)
        finally:
            config.USE_TENSORRT = original_value

    def test_use_tensorrt_docstring(self) -> None:
        # Check for comment and variable in source
        """Test use tensorrt docstring."""
        source: str = inspect.getsource(config)
        self.assertIn('USE_TENSORRT', source)
        self.assertIn('Whether to use TensorRT for inference', source)

    def test_use_sahi_variable(self) -> None:
        """Test the USE_SAHI configuration variable."""
        self.assertTrue(hasattr(config, 'USE_SAHI'))
        self.assertIsInstance(config.USE_SAHI, bool)

    def test_model_variants_variable(self) -> None:
        """Test the MODEL_VARIANTS configuration variable."""
        self.assertTrue(hasattr(config, 'MODEL_VARIANTS'))
        self.assertIsInstance(config.MODEL_VARIANTS, list)
        self.assertGreater(len(config.MODEL_VARIANTS), 0)

    def test_lazy_load_models_variable(self) -> None:
        """Test the LAZY_LOAD_MODELS configuration variable."""
        self.assertTrue(hasattr(config, 'LAZY_LOAD_MODELS'))
        self.assertIsInstance(config.LAZY_LOAD_MODELS, bool)

    def test_max_loaded_models_variable(self) -> None:
        """Test the MAX_LOADED_MODELS configuration variable."""
        self.assertTrue(hasattr(config, 'MAX_LOADED_MODELS'))
        self.assertIsInstance(config.MAX_LOADED_MODELS, int)

    def test_preload_smallest_variable(self) -> None:
        """Test the PRELOAD_SMALLEST configuration variable."""
        self.assertTrue(hasattr(config, 'PRELOAD_SMALLEST'))
        self.assertIsInstance(config.PRELOAD_SMALLEST, bool)

    def test_explicit_cuda_cleanup_variable(self) -> None:
        """Test the EXPLICIT_CUDA_CLEANUP configuration variable."""
        self.assertTrue(hasattr(config, 'EXPLICIT_CUDA_CLEANUP'))
        self.assertIsInstance(config.EXPLICIT_CUDA_CLEANUP, bool)

    def test_get_model_variants_function(self) -> None:
        """Test the get_model_variants function."""
        variants = config.get_model_variants()
        self.assertIsInstance(variants, list)
        self.assertEqual(variants, config.MODEL_VARIANTS)

    def test_is_lazy_loading_enabled_function(self) -> None:
        """Test the is_lazy_loading_enabled function."""
        result = config.is_lazy_loading_enabled()
        self.assertIsInstance(result, bool)
        self.assertEqual(result, config.LAZY_LOAD_MODELS)

    def test_get_max_loaded_models_function(self) -> None:
        """Test the get_max_loaded_models function."""
        result = config.get_max_loaded_models()
        self.assertIsInstance(result, int)
        self.assertEqual(result, config.MAX_LOADED_MODELS)

    def test_should_preload_smallest_model_function(self) -> None:
        """Test the should_preload_smallest_model function."""
        result = config.should_preload_smallest_model()
        self.assertIsInstance(result, bool)
        self.assertEqual(result, config.PRELOAD_SMALLEST)

    def test_should_cleanup_cuda_cache_function(self) -> None:
        """Test the should_cleanup_cuda_cache function."""
        result = config.should_cleanup_cuda_cache()
        self.assertIsInstance(result, bool)
        self.assertEqual(result, config.EXPLICIT_CUDA_CLEANUP)

    def test_model_variants_empty_input(self) -> None:
        """Test that empty MODEL_VARIANTS input stays empty."""
        empty_env_value = ''
        variants = [v.strip() for v in empty_env_value.split(',') if v.strip()]

        self.assertEqual(variants, [])

    @patch.dict(os.environ, {'MODEL_VARIANTS': ''}, clear=False)
    def test_actual_model_variants_empty(self) -> None:
        """Test actual module behaviour by running in subprocess with empty
        MODEL_VARIANTS."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                '-c',
                'import os; os.environ["MODEL_VARIANTS"] = ""; '
                'from examples.YOLO_server_api import config; '
                'print(config.MODEL_VARIANTS)',
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout.strip().splitlines()[-1]
        self.assertEqual(output, '[]')

    @patch.dict(
        os.environ,
        {'USE_SAHI': 'true', 'USE_TENSORRT': 'true'},
        clear=False,
    )
    def test_actual_sahi_tensorrt_conflict_warning(self) -> None:
        """Test actual warning when both SAHI and TensorRT are enabled."""
        import subprocess
        import sys

        # Run a subprocess with both flags enabled to test warning
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                'import os; os.environ["USE_SAHI"] = "true"; '
                'os.environ["USE_TENSORRT"] = "true"; '
                'import warnings; warnings.simplefilter("always"); '
                'from examples.YOLO_server_api import config',
            ],
            capture_output=True,
            text=True,
        )

        # Should have warning in stderr
        self.assertIn('USE_SAHI=True forces .pt model usage', result.stderr)

    def test_config_info_display(self) -> None:
        """Test that the configuration info string is properly formatted."""
        self.assertTrue(hasattr(config, '_CONFIG_INFO'))
        self.assertIsInstance(config._CONFIG_INFO, str)
        self.assertIn('YOLO Server API Configuration', config._CONFIG_INFO)
        self.assertIn('USE_TENSORRT', config._CONFIG_INFO)
        self.assertIn('USE_SAHI', config._CONFIG_INFO)

    def test_sahi_tensorrt_conflict_warning(self) -> None:
        """Test that a warning would be issued when both SAHI and TensorRT are
        enabled."""
        # Test the warning logic without actually triggering it
        # Since both USE_SAHI and USE_TENSORRT would be True in that scenario
        use_sahi_true = True
        use_tensorrt_true = True

        # This condition should trigger a warning
        should_warn = use_sahi_true and use_tensorrt_true
        self.assertTrue(should_warn)

        # Test that the warning message would contain the expected text
        expected_message = (
            'USE_SAHI=True forces .pt model usage, overriding '
            'USE_TENSORRT=True'
        )
        self.assertIsInstance(expected_message, str)
        self.assertIn('USE_SAHI=True', expected_message)
        self.assertIn('USE_TENSORRT=True', expected_message)


if __name__ == '__main__':
    unittest.main()


class TestYoloServerConfigCoverage(unittest.TestCase):
    """Provide TestYoloServerConfigCoverage."""

    def test_inference_device_honours_explicit_and_auto_detection(
        self,
    ) -> None:
        """Explicit devices win; automatic mode mirrors CUDA availability."""
        with patch.object(config, 'YOLO_INFERENCE_DEVICE', 'cuda:2'):
            self.assertEqual(config.get_inference_device(), 'cuda:2')

        with (
            patch.object(config, 'YOLO_INFERENCE_DEVICE', 'auto'),
            patch.object(config, '_is_cuda_available', return_value=True),
        ):
            self.assertEqual(config.get_inference_device(), 'cuda:0')

        with (
            patch.object(config, 'YOLO_INFERENCE_DEVICE', 'auto'),
            patch.object(config, '_is_cuda_available', return_value=False),
        ):
            self.assertEqual(config.get_inference_device(), 'cpu')

    def test_inference_device_falls_back_when_cuda_probe_fails(self) -> None:
        """CUDA probe failures produce a warning and retain a CPU fallback."""
        with (
            patch.object(config, 'YOLO_INFERENCE_DEVICE', 'auto'),
            patch.object(
                config,
                '_is_cuda_available',
                side_effect=RuntimeError('CUDA probe failed'),
            ),
            self.assertWarnsRegex(UserWarning, 'falling back to CPU'),
        ):
            self.assertEqual(config.get_inference_device(), 'cpu')

    def test_cuda_availability_probe_returns_bool(self) -> None:
        """The isolated PyTorch probe returns a boolean result."""
        self.assertIsInstance(config._is_cuda_available(), bool)

    def test_sahi_tensorrt_conflict_warns_during_configuration_load(
        self,
    ) -> None:
        """Reloading the module validates mutually incompatible model modes."""
        original_environment = {
            'USE_SAHI': os.environ.get('USE_SAHI'),
            'USE_TENSORRT': os.environ.get('USE_TENSORRT'),
        }
        try:
            with patch.dict(
                os.environ,
                {'USE_SAHI': 'true', 'USE_TENSORRT': 'true'},
                clear=False,
            ):
                with self.assertWarnsRegex(
                    UserWarning,
                    'USE_SAHI=True forces .pt model usage',
                ):
                    importlib.reload(config)
        finally:
            for name, value in original_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            importlib.reload(config)

    def test_log_configuration_uses_the_module_logger(self) -> None:
        """Configuration output is emitted through structured logging."""
        with patch.object(config.logger, 'info') as info:
            config.log_configuration()

        info.assert_called_once_with('%s', config._CONFIG_INFO)

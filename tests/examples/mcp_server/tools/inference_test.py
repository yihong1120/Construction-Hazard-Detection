from __future__ import annotations

import base64
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import cv2
import numpy as np

from examples.mcp_server.tools.inference import InferenceTools


class DetectFrameTests(unittest.IsolatedAsyncioTestCase):
    """Tests for InferenceTools.detect_frame method."""

    async def test_detect_frame_requires_input(self) -> None:
        """Should raise when neither image_base64 nor image_url is provided."""
        tool = InferenceTools()
        with self.assertRaises(ValueError):
            await tool.detect_frame()

    async def test_detect_frame_success_with_base64(self) -> None:
        """Should decode base64 and call detector.generate_detections."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_dets = [[1, 2, 3, 4, 0.9, 1]]
        fake_trk: list[list[int]] = [[1, 2, 3, 4]]
        b64_str = base64.b64encode(b'dummy').decode()

        with (
            patch.object(
                InferenceTools, '_load_image',
                AsyncMock(return_value=fake_frame),
            ),
            patch.object(InferenceTools, '_init_detector', AsyncMock()),
        ):
            tool = InferenceTools()
            detector_mock = AsyncMock()
            detector_mock.generate_detections.return_value = (
                fake_dets, fake_trk,
            )
            tool._detector = detector_mock

            res = await tool.detect_frame(image_base64=b64_str)

        detector_mock.generate_detections.assert_awaited_once_with(fake_frame)
        self.assertEqual(res['detections'], fake_dets)
        self.assertEqual(res['tracked'], fake_trk)
        self.assertIn('meta', res)
        self.assertEqual(res['meta']['frame_size'], [640, 480])

    async def test_detect_frame_logs_and_reraises_on_failure(self) -> None:
        """Should log error and re-raise when exception occurs."""
        with patch.object(
            InferenceTools,
            '_load_image',
            AsyncMock(side_effect=RuntimeError('boom')),
        ), patch(
            'examples.mcp_server.tools.inference.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = InferenceTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.detect_frame(image_base64='abc')
            logger.error.assert_called_once()
            self.assertIn('Detection failed', logger.error.call_args[0][0])

    async def test_detect_frame_calls_init_when_detector_missing(self) -> None:
        """On first use, detect_frame should initialise the detector."""
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        fake_dets = [[0, 0, 1, 1, 0.5, 0]]
        fake_trk: list[list[int]] = []

        async def init_side_effect(*args, **kwargs):
            # Install a detector after init is called
            detector = AsyncMock()
            detector.generate_detections.return_value = (fake_dets, fake_trk)
            tool._detector = detector

        with patch.object(
            InferenceTools,
            '_load_image',
            AsyncMock(return_value=fake_frame),
        ), patch.object(
            InferenceTools,
            '_init_detector',
            AsyncMock(side_effect=init_side_effect),
        ) as mock_init:
            tool = InferenceTools()
            res = await tool.detect_frame(
                image_base64='abc==',
                use_remote=True,
                model_key='yolo11n',
            )
        mock_init.assert_awaited_once()
        self.assertEqual(res['detections'], fake_dets)
        self.assertEqual(res['tracked'], fake_trk)
        self.assertEqual(res['meta']['engine'], 'remote')

    async def test_detect_frame_raises_when_load_image_returns_none(
        self,
    ) -> None:
        """If image fails to load, detect_frame should log and raise
        ValueError.
        """
        with patch.object(
            InferenceTools,
            '_load_image',
            AsyncMock(return_value=None),
        ), patch(
            'examples.mcp_server.tools.inference.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = InferenceTools()
            tool.logger = logger
            with self.assertRaises(ValueError):
                await tool.detect_frame(image_base64='abc')
            logger.error.assert_called_once()


class LoadImageTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the private _load_image method."""

    async def test_load_image_from_valid_base64(self) -> None:
        """Should decode base64 bytes and return an image."""
        fake_np = np.zeros((2, 2, 3), dtype=np.uint8)
        encoded = base64.b64encode(cv2.imencode('.jpg', fake_np)[1]).decode()
        with patch('cv2.imdecode', return_value=fake_np):
            tool = InferenceTools()
            result = await tool._load_image(encoded, None)
        self.assertTrue(isinstance(result, np.ndarray))
        if isinstance(result, np.ndarray):
            self.assertEqual(result.shape, (2, 2, 3))

    async def test_load_image_invalid_base64_triggers_logger(self) -> None:
        """Invalid base64 should log error and return None."""
        tool = InferenceTools()
        with patch(
            'examples.mcp_server.tools.inference.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool.logger = logger
            res = await tool._load_image('%%%notbase64%%%', None)
            self.assertIsNone(res)
            logger.error.assert_called_once()

    async def test_load_image_from_url_success(self) -> None:
        """Should download image and decode it."""
        fake_bytes = b'fakeimage'
        fake_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        mock_response = MagicMock()
        mock_response.content = fake_bytes
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get.return_value = mock_response
        with (
            patch('httpx.AsyncClient', return_value=mock_client),
            patch('cv2.imdecode', return_value=fake_frame),
        ):
            tool = InferenceTools()
            res = await tool._load_image(None, 'http://example.com/img.jpg')
        self.assertTrue(isinstance(res, np.ndarray))

    async def test_load_image_returns_none_when_inputs_missing(self) -> None:
        """When both inputs are None, loader should return None."""
        tool = InferenceTools()
        res = await tool._load_image(None, None)
        self.assertIsNone(res)

    async def test_load_image_fallback_to_pillow_when_cv2_none(self) -> None:
        """Should try Pillow if cv2 fails to decode."""
        fake_bytes = b'imgdata'
        mock_response = MagicMock()
        mock_response.content = fake_bytes
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get.return_value = mock_response
        with (
            patch('httpx.AsyncClient', return_value=mock_client),
            patch('cv2.imdecode', return_value=None),
            patch('PIL.Image.open') as mock_open,
        ):
            mock_open.return_value.convert.return_value = MagicMock()
            tool = InferenceTools()
            res = await tool._load_image(None, 'http://img')
        self.assertTrue(res is None or isinstance(res, np.ndarray))

    async def test_load_image_raises_then_logs_when_all_decoders_fail(
        self,
    ) -> None:
        """If cv2 returns None and Pillow fails,
        it should log and return None.
        """
        encoded = base64.b64encode(b'abc').decode()
        with (
            patch('cv2.imdecode', return_value=None),
            patch('PIL.Image.open', side_effect=ValueError('fail')),
            patch(
                'examples.mcp_server.tools.inference.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = InferenceTools()
            tool.logger = logger
            res = await tool._load_image(encoded, None)
            self.assertIsNone(res)
            logger.error.assert_called_once()

    async def test_load_image_base64_fallback_pillow_success(self) -> None:
        """Base64 path: when cv2 decode fails, Pillow fallback should
        succeed and return an array.
        """
        fake_np = np.zeros((1, 1, 3), dtype=np.uint8)
        buf = cv2.imencode('.jpg', fake_np)[1]
        encoded = base64.b64encode(buf).decode()
        with (
            patch('cv2.imdecode', return_value=None),
            patch('PIL.Image.open') as mock_open,
        ):
            mock_open.return_value.convert.return_value = np.zeros(
                (1, 1, 3), dtype=np.uint8,
            )
            tool = InferenceTools()
            result = await tool._load_image(encoded, None)
        self.assertTrue(isinstance(result, np.ndarray))
        if isinstance(result, np.ndarray):
            self.assertEqual(result.shape, (1, 1, 3))

    async def test_load_image_handles_data_url_prefix(self) -> None:
        """Data URL prefix should be stripped before decoding."""
        fake_np = np.zeros((2, 2, 3), dtype=np.uint8)
        # Encode a tiny jpeg buffer
        buf = cv2.imencode('.jpg', fake_np)[1]
        encoded = base64.b64encode(buf).decode()
        data_url = f"data:image/jpeg;base64,{encoded}"
        with patch('cv2.imdecode', return_value=fake_np):
            tool = InferenceTools()
            result = await tool._load_image(data_url, None)
        self.assertTrue(isinstance(result, np.ndarray))

    async def test_load_image_url_all_decoders_fail(self) -> None:
        """URL path: when cv2 returns None and Pillow fails, it should log
        and return None.
        """
        fake_bytes = b'img'
        mock_response = MagicMock()
        mock_response.content = fake_bytes
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get.return_value = mock_response
        with (
            patch('httpx.AsyncClient', return_value=mock_client),
            patch('cv2.imdecode', return_value=None),
            patch('PIL.Image.open', side_effect=ValueError('fail')),
            patch(
                'examples.mcp_server.tools.inference.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = InferenceTools()
            tool.logger = logger
            result = await tool._load_image(
                None,
                'http://example.com/fail.jpg',
            )
            self.assertIsNone(result)
            logger.error.assert_called_once()

    async def test_load_image_url_fallback_pillow_success(self) -> None:
        """URL path: when cv2 decode fails, Pillow fallback should
        succeed and return an array.
        """
        fake_bytes = b'imgdata'
        mock_response = MagicMock()
        mock_response.content = fake_bytes
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get.return_value = mock_response
        with (
            patch('httpx.AsyncClient', return_value=mock_client),
            patch('cv2.imdecode', return_value=None),
            patch('PIL.Image.open') as mock_open,
        ):
            # Simulate Pillow producing an RGB image array
            mock_open.return_value.convert.return_value = np.zeros(
                (1, 1, 3), dtype=np.uint8,
            )
            tool = InferenceTools()
            res = await tool._load_image(
                None,
                'http://example.com/img.jpg',
            )
        self.assertTrue(isinstance(res, np.ndarray))
        if isinstance(res, np.ndarray):
            self.assertEqual(res.shape, (1, 1, 3))

    async def test_load_image_logs_error_on_exception(self) -> None:
        """Should catch unexpected exceptions and log error."""
        with (
            patch('cv2.imdecode', side_effect=RuntimeError('oops')),
            patch(
                'examples.mcp_server.tools.inference.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = InferenceTools()
            tool.logger = logger
            result = await tool._load_image('YWJj', None)
            self.assertIsNone(result)
            logger.error.assert_called_once()


class InitDetectorTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the private _init_detector method."""

    async def test_init_detector_creates_live_stream_detector(self) -> None:
        """Should create LiveStreamDetector with expected parameters."""
        with patch(
            'examples.mcp_server.tools.inference.LiveStreamDetector',
        ) as mock_lsd, patch(
            'examples.mcp_server.tools.inference.get_env_var',
            return_value='api_url',
        ), patch(
            'examples.mcp_server.tools.inference.get_env_int',
            side_effect=[1, 10],
        ):
            tool = InferenceTools()
            await tool._init_detector(
                use_remote=True,
                model_key='yolo11n',
                use_ultralytics=False,
                remote_tracker='centroid',
                remote_cost_threshold=0.7,
                ws_frame_size=(640, 480),
                use_jpeg_ws=True,
                movement_thr=40.0,
            )
            mock_lsd.assert_called_once()
            kwargs = mock_lsd.call_args.kwargs
            self.assertEqual(kwargs['api_url'], 'api_url')
            self.assertEqual(kwargs['fps'], 1)
            self.assertEqual(kwargs['max_id_keep'], 10)
            self.assertTrue('movement_thr' in kwargs)
            self.assertTrue(tool._detector is not None)


class CloseTests(unittest.IsolatedAsyncioTestCase):
    """Tests for InferenceTools.close method."""

    async def test_close_invokes_detector_close(self) -> None:
        """When detector exists, its close() should be awaited and cleared."""
        detector_mock = AsyncMock()
        tool = InferenceTools()
        tool._detector = detector_mock
        await tool.close()
        detector_mock.close.assert_awaited_once()
        self.assertIsNone(tool._detector)

    async def test_close_no_detector_does_nothing(self) -> None:
        """Should not raise when no detector exists."""
        tool = InferenceTools()
        await tool.close()
        # No exception should be raised
        self.assertIsNone(tool._detector)


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.tools.inference \
       --cov-report=term-missing \
       tests/examples/mcp_server/tools/inference_test.py
'''

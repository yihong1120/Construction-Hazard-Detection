from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.mcp_server.tools.streaming import StreamingTools


class StartStopDetectionTests(unittest.IsolatedAsyncioTestCase):
    """Tests for start_detection_stream and stop_detection_stream."""

    async def test_start_detection_stream_with_custom_id(self):
        """Should return unsupported message and register stream."""
        with patch(
            'examples.mcp_server.tools.streaming.LiveStreamDetector',
        ) as mock_detector:
            tool = StreamingTools()
            result = await tool.start_detection_stream(
                'rtsp://cam',
                'stream_1',
            )
        mock_detector.assert_called_once()
        self.assertFalse(result['success'])
        self.assertIn('unsupported', result['status'])
        self.assertIn('stream_1', tool._active_streams)

    async def test_start_detection_stream_autogen_id(self):
        """Should auto-generate a stream_id."""
        with patch(
            'examples.mcp_server.tools.streaming.LiveStreamDetector',
        ):
            tool = StreamingTools()
            result = await tool.start_detection_stream('file.mp4', None)
        self.assertIn('stream_', result['stream_id'])
        self.assertIn(result['stream_id'], tool._active_streams)

    async def test_start_detection_stream_exception(self):
        """Should log and raise on exception."""
        with (
            patch(
                'examples.mcp_server.tools.streaming.LiveStreamDetector',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.streaming.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = StreamingTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.start_detection_stream('x')
            logger.error.assert_called_once()

    async def test_stop_detection_stream_existing(self):
        """Should update existing stream and return unsupported message."""
        with patch(
            'examples.mcp_server.tools.streaming.LiveStreamDetector',
        ):
            tool = StreamingTools()
            tool._active_streams['s1'] = {'status': 'active'}
            result = await tool.stop_detection_stream('s1')
        self.assertFalse(result['success'])
        self.assertIn('unsupported', result['status'])

    async def test_stop_detection_stream_non_existing(self):
        """Should handle non-existing stream gracefully."""
        with patch(
            'examples.mcp_server.tools.streaming.LiveStreamDetector',
        ):
            tool = StreamingTools()
            result = await tool.stop_detection_stream('nope')
        self.assertIn('unsupported', result['status'])

    async def test_stop_detection_stream_exception(self):
        """Should log and raise on exception."""
        with (
            patch(
                'examples.mcp_server.tools.streaming.LiveStreamDetector',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.streaming.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = StreamingTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.stop_detection_stream('id')
            logger.error.assert_called_once()


class StreamStatusTests(unittest.IsolatedAsyncioTestCase):
    """Tests for get_stream_status."""

    async def test_get_specific_existing_stream(self):
        """Should return info for existing stream."""
        tool = StreamingTools()
        tool._active_streams['abc'] = {'status': 'active'}
        res = await tool.get_stream_status('abc')
        self.assertTrue(res['success'])
        self.assertIn('stream_info', res)

    async def test_get_specific_missing_stream(self):
        """Should return not found message."""
        tool = StreamingTools()
        res = await tool.get_stream_status('missing')
        self.assertFalse(res['success'])
        self.assertIn('not found', res['message'])

    async def test_get_all_streams_status(self):
        """Should return aggregated stats."""
        tool = StreamingTools()
        tool._active_streams = {
            'a': {'status': 'active'},
            'b': {'status': 'unsupported'},
        }
        res = await tool.get_stream_status()
        self.assertTrue(res['success'])
        self.assertEqual(res['active_streams'], 1)
        self.assertEqual(res['total_streams'], 2)

    async def test_get_stream_status_exception(self):
        """Should log and raise on exception."""
        with patch(
            'examples.mcp_server.tools.streaming.logging.getLogger',
        ) as mock_logger:
            logger = mock_logger.return_value
            tool = StreamingTools()
            tool.logger = logger
            with patch.object(
                tool,
                '_active_streams',
                new_callable=MagicMock,
                side_effect=RuntimeError('boom'),
            ):
                with self.assertRaises(RuntimeError):
                    await tool.get_stream_status()
            logger.error.assert_called_once()


class CaptureFrameTests(unittest.IsolatedAsyncioTestCase):
    """Tests for capture_frame."""

    async def test_capture_frame_base64_success(self):
        """Should capture frame and encode base64."""
        fake_cap = MagicMock()
        fake_cap.read.return_value = (True, 'frame')
        with (
            patch('examples.mcp_server.tools.streaming.StreamCapture'),
            patch(
                'examples.mcp_server.tools.streaming.cv2.VideoCapture',
                return_value=fake_cap,
            ),
            patch(
                'examples.mcp_server.tools.streaming.cv2.imencode',
                return_value=(True, b'jpegdata'),
            ),
            patch(
                'examples.mcp_server.tools.streaming.base64.b64encode',
                return_value=b'encoded',
            ),
        ):
            tool = StreamingTools()
            res = await tool.capture_frame('video.mp4')
        self.assertTrue(res['success'])
        self.assertEqual(res['format'], 'base64')

    async def test_capture_frame_bytes_format(self):
        """Should return bytes when requested."""
        fake_cap = MagicMock()
        fake_cap.read.return_value = (True, 'frame')
        with (
            patch('examples.mcp_server.tools.streaming.StreamCapture'),
            patch(
                'examples.mcp_server.tools.streaming.cv2.VideoCapture',
                return_value=fake_cap,
            ),
            patch(
                'examples.mcp_server.tools.streaming.cv2.imencode',
                return_value=(True, b'bytes'),
            ),
        ):
            tool = StreamingTools()
            res = await tool.capture_frame('cam', frame_format='bytes')
        self.assertTrue(res['success'])
        self.assertEqual(res['format'], 'bytes')

    async def test_capture_frame_array_format(self):
        """Should return array when requested."""
        fake_cap = MagicMock()
        fake_cap.read.return_value = (True, [[1, 2], [3, 4]])
        with (
            patch('examples.mcp_server.tools.streaming.StreamCapture'),
            patch(
                'examples.mcp_server.tools.streaming.cv2.VideoCapture',
                return_value=fake_cap,
            ),
        ):
            tool = StreamingTools()
            res = await tool.capture_frame('cam', frame_format='array')
        self.assertTrue(res['success'])
        self.assertIsInstance(res['frame_data'], list)

    async def test_capture_frame_fail(self):
        """Should handle failed read gracefully."""
        fake_cap = MagicMock()
        fake_cap.read.return_value = (False, None)
        with (
            patch('examples.mcp_server.tools.streaming.StreamCapture'),
            patch(
                'examples.mcp_server.tools.streaming.cv2.VideoCapture',
                return_value=fake_cap,
            ),
        ):
            tool = StreamingTools()
            res = await tool.capture_frame('x')
        self.assertFalse(res['success'])

    async def test_capture_frame_exception(self):
        """Should log and raise on exception."""
        with (
            patch(
                'examples.mcp_server.tools.streaming.StreamCapture',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.streaming.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = StreamingTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.capture_frame('url')
            logger.error.assert_called_once()


class ViewerTests(unittest.IsolatedAsyncioTestCase):
    """Tests for start_stream_viewer and stop_stream_viewer."""

    async def test_start_stream_viewer_success(self):
        """Should start viewer and return URL."""
        fake_viewer = AsyncMock()
        fake_viewer.start_viewer.return_value = (True, 'http://localhost:8081')
        with patch(
            'examples.mcp_server.tools.streaming.StreamViewer',
            return_value=fake_viewer,
        ):
            tool = StreamingTools()
            res = await tool.start_stream_viewer('rtsp://cam')
        self.assertTrue(res['success'])
        self.assertIn('localhost', res['viewer_url'])

    async def test_start_stream_viewer_fail(self):
        """Should handle failed viewer start."""
        fake_viewer = AsyncMock()
        fake_viewer.start_viewer.return_value = (False, 'url')
        with patch(
            'examples.mcp_server.tools.streaming.StreamViewer',
            return_value=fake_viewer,
        ):
            tool = StreamingTools()
            res = await tool.start_stream_viewer('cam')
        self.assertFalse(res['success'])

    async def test_start_stream_viewer_exception(self):
        """Should log and raise on exception."""
        with (
            patch(
                'examples.mcp_server.tools.streaming.StreamViewer',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.streaming.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = StreamingTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.start_stream_viewer('x')
            logger.error.assert_called_once()

    async def test_stop_stream_viewer_success(self):
        """Should stop viewer successfully."""
        fake_viewer = AsyncMock()
        fake_viewer.stop_viewer.return_value = True
        with patch(
            'examples.mcp_server.tools.streaming.StreamViewer',
            return_value=fake_viewer,
        ):
            tool = StreamingTools()
            res = await tool.stop_stream_viewer(8081)
        self.assertTrue(res['success'])

    async def test_stop_stream_viewer_fail(self):
        """Should return failure message if stop_viewer returns False."""
        fake_viewer = AsyncMock()
        fake_viewer.stop_viewer.return_value = False
        with patch(
            'examples.mcp_server.tools.streaming.StreamViewer',
            return_value=fake_viewer,
        ):
            tool = StreamingTools()
            res = await tool.stop_stream_viewer(9090)
        self.assertFalse(res['success'])

    async def test_stop_stream_viewer_exception(self):
        """Should log and raise on exception."""
        with (
            patch(
                'examples.mcp_server.tools.streaming.StreamViewer',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.streaming.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = StreamingTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.stop_stream_viewer()
            logger.error.assert_called_once()


class EnsureMethodsTests(unittest.IsolatedAsyncioTestCase):
    """Tests for _ensure_* methods."""

    async def test_ensure_live_detector_initialises_once(self):
        """Should initialise detector only once."""
        with patch(
            'examples.mcp_server.tools.streaming.LiveStreamDetector',
        ) as mock_live:
            tool = StreamingTools()
            await tool._ensure_live_detector()
            await tool._ensure_live_detector()
            mock_live.assert_called_once()

    async def test_ensure_live_detector_skip_when_exists(self):
        """Should skip creation if already set."""
        tool = StreamingTools()
        tool._live_detector = object()
        await tool._ensure_live_detector()

    async def test_ensure_stream_capture(self):
        """Should initialise capture with URL and skip if None."""
        with patch(
            'examples.mcp_server.tools.streaming.StreamCapture',
        ) as mock_cap:
            tool = StreamingTools()
            await tool._ensure_stream_capture('url')
            await tool._ensure_stream_capture(None)
            mock_cap.assert_called_once()

    async def test_ensure_stream_capture_skip_when_exists(self):
        """Should skip creation if already exists."""
        tool = StreamingTools()
        tool._stream_capture = object()
        await tool._ensure_stream_capture(None)

    async def test_ensure_stream_capture_none_when_uninitialised(self):
        """When uninitialised and URL None,
        should early return without creating.
        """
        tool = StreamingTools()
        await tool._ensure_stream_capture(None)
        self.assertIsNone(tool._stream_capture)

    async def test_ensure_stream_viewer(self):
        """Should initialise viewer with URL and skip if None."""
        with patch(
            'examples.mcp_server.tools.streaming.StreamViewer',
        ) as mock_view:
            tool = StreamingTools()
            await tool._ensure_stream_viewer('url')
            await tool._ensure_stream_viewer(None)
            mock_view.assert_called_once()

    async def test_ensure_stream_viewer_skip_when_exists(self):
        """Should skip creation if already exists."""
        tool = StreamingTools()
        tool._stream_viewer = object()
        await tool._ensure_stream_viewer(None)

    async def test_ensure_stream_viewer_none_when_uninitialised(self):
        """When uninitialised and URL None,
        should early return without creating.
        """
        tool = StreamingTools()
        await tool._ensure_stream_viewer(None)
        self.assertIsNone(tool._stream_viewer)


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.tools.streaming\
    --cov-report=term-missing\
    tests/examples/mcp_server/tools/streaming_test.py
'''

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from src.stream_viewer import StreamViewer


class TestStreamViewer(unittest.TestCase):

    @patch('src.stream_viewer.cv2.VideoCapture')
    def test_initialisation(self, mock_video_capture):
        # Initialise StreamViewer with a test URL
        stream_url = 'https://example.com/stream'
        viewer = StreamViewer(stream_url)

        # Check if the URL is set correctly
        self.assertEqual(viewer.stream_url, stream_url)

        # Check if the window name is set correctly
        self.assertEqual(viewer.window_name, 'Stream Viewer')

        # Check if VideoCapture was called with the correct URL
        mock_video_capture.assert_called_once_with(stream_url)

    @patch('src.stream_viewer.cv2.destroyAllWindows')
    @patch('src.stream_viewer.cv2.VideoCapture')
    @patch('src.stream_viewer.cv2.waitKey')
    @patch('src.stream_viewer.cv2.imshow')
    def test_display_stream(
        self,
        mock_imshow,
        mock_wait_key,
        mock_video_capture,
        mock_destroyAllWindows,
    ):
        # Mock VideoCapture instance
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance

        # Simulate read() returning True with a dummy frame
        mock_cap_instance.read.side_effect = [
            (True, MagicMock()), (True, MagicMock()), (False, None),
        ]

        # Simulate waitKey() returning 'q' to break the loop
        mock_wait_key.side_effect = [ord('a'), ord('b'), ord('q')]

        # Initialise StreamViewer and call display_stream
        viewer = StreamViewer('https://example.com/stream')
        viewer.display_stream()

        # Check if imshow was called correctly
        self.assertEqual(mock_imshow.call_count, 2)

        # Check if waitKey was called at least twice
        self.assertGreaterEqual(mock_wait_key.call_count, 2)

        # Check if read was called at least twice
        self.assertGreaterEqual(mock_cap_instance.read.call_count, 2)

        # Check if destroyAllWindows was called
        mock_destroyAllWindows.assert_called_once()

    @patch('src.stream_viewer.cv2.VideoCapture')
    @patch('src.stream_viewer.cv2.destroyAllWindows')
    def test_release_resources(
        self, mock_destroy_all_windows, mock_video_capture,
    ):
        # Mock VideoCapture instance
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance

        # Initialise StreamViewer and call release_resources
        viewer = StreamViewer('https://example.com/stream')
        viewer.release_resources()

        # Check if release was called on VideoCapture instance
        mock_cap_instance.release.assert_called_once()

        # Check if destroyAllWindows was called
        mock_destroy_all_windows.assert_called_once()


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from flask_socketio import SocketIO

from examples.streaming_web.sockets import register_sockets
from examples.streaming_web.sockets import update_images


class TestSockets(unittest.TestCase):
    def setUp(self):
        self.socketio = MagicMock(spec=SocketIO)
        self.r = MagicMock()

    @patch('examples.streaming_web.sockets.emit')
    def test_handle_connect(self, mock_emit):
        """
        Test that the handle_connect function emits a connection message
        and starts the background task to update images.
        """
        # Mock start_background_task method
        self.socketio.start_background_task = MagicMock()

        # Register the sockets and simulate a 'connect' event
        register_sockets(self.socketio, self.r)
        connect_handler = self.socketio.on.call_args_list[0][0][1]
        print(f"connect_handler: {connect_handler}")
        connect_handler()

        # Check if the emit and start_background_task were called as expected
        mock_emit.assert_called_once_with('message', {'data': 'Connected'})
        self.socketio.start_background_task.assert_called_once_with(
            update_images, self.socketio, self.r,
        )

    @patch('builtins.print')
    def test_handle_disconnect(self, mock_print):
        """
        Test that the handle_disconnect function prints a disconnect message.
        """
        # Register the sockets and simulate a 'disconnect' event
        register_sockets(self.socketio, self.r)
        disconnect_handler = self.socketio.on.call_args_list[1][0][1]
        disconnect_handler()

        # Check if the print statement was called as expected
        mock_print.assert_called_once_with('Client disconnected')

    @patch('builtins.print')
    def test_handle_error(self, mock_print):
        """
        Test that the handle_error function prints an error message.
        """
        # Register the sockets and simulate an 'error' event
        register_sockets(self.socketio, self.r)
        error_handler = self.socketio.on.call_args_list[2][0][1]
        print(f"error_handler: {error_handler}")
        error_handler(Exception('Test error'))

        # Check if the print statement was called as expected
        mock_print.assert_called_once_with('Error: Test error')

    @patch('examples.streaming_web.sockets.get_labels')
    @patch('examples.streaming_web.sockets.get_image_data')
    @patch('examples.streaming_web.sockets.SocketIO.emit')
    @patch('examples.streaming_web.sockets.SocketIO.sleep')
    def test_update_images(
        self,
        mock_sleep,
        mock_emit,
        mock_get_image_data,
        mock_get_labels,
    ):
        """
        Test that the update_images function emits updates with image data.
        """
        mock_get_labels.return_value = ['label1', 'label2']
        mock_get_image_data.side_effect = [
            [('image_data1', 'image_name1')],
            [('image_data2', 'image_name2')],
        ]

        def stop_loop(*args, **kwargs):
            raise Exception('Stop loop')

        # Simulate breaking the loop after the first iteration
        mock_sleep.side_effect = stop_loop

        with patch('builtins.print') as _:
            try:
                update_images(self.socketio, self.r)
            except Exception:
                pass

        # Ensure that get_labels is called once
        mock_get_labels.assert_called_once_with(self.r)

        # Ensure that emit is called twice (once for each label)
        self.assertEqual(mock_emit.call_count, 2)

        # Check the arguments passed to emit for the first call
        mock_emit.assert_any_call(
            'update',
            {
                'label': 'label1',
                'images': ['image_data1'],
                'image_names': ['image_name1'],
            },
        )

        # Check the arguments passed to emit for the second call
        mock_emit.assert_any_call(
            'update',
            {
                'label': 'label2',
                'images': ['image_data2'],
                'image_names': ['image_name2'],
            },
        )


if __name__ == '__main__':
    unittest.main()

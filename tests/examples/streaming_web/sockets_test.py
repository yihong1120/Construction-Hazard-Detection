import unittest
from unittest.mock import MagicMock, patch
from flask_socketio import SocketIO
from examples.streaming_web.sockets import register_sockets, update_images


class TestSockets(unittest.TestCase):
    def setUp(self):
        self.socketio = MagicMock(spec=SocketIO)
        self.r = MagicMock()
        register_sockets(self.socketio, self.r)

    @patch('examples.streaming_web.sockets.emit')
    def test_handle_connect(self, mock_emit):
        """
        Test that the handle_connect function emits a connection message
        and starts the background task to update images.
        """
        # Simulate the 'connect' event
        connect_handler = self.socketio.on.call_args_list[0][0][1]
        connect_handler()

        mock_emit.assert_called_once_with('message', {'data': 'Connected'})
        self.socketio.start_background_task.assert_called_once_with(
            update_images, self.socketio, self.r
        )

    def test_handle_disconnect(self):
        """
        Test that the handle_disconnect function prints a disconnect message.
        """
        with patch('builtins.print') as mock_print:
            # Simulate the 'disconnect' event
            disconnect_handler = self.socketio.on.call_args_list[1][0][1]
            disconnect_handler()
            mock_print.assert_called_once_with('Client disconnected')

    def test_handle_error(self):
        """
        Test that the handle_error function prints an error message.
        """
        with patch('builtins.print') as mock_print:
            # Simulate the 'error' event
            error_handler = self.socketio.on.call_args_list[2][0][1]
            error_handler(Exception('Test error'))
            mock_print.assert_called_once_with('Error: Test error')

    @patch('examples.streaming_web.sockets.get_labels')
    @patch('examples.streaming_web.sockets.get_image_data')
    @patch('examples.streaming_web.sockets.SocketIO.emit')
    @patch('examples.streaming_web.sockets.SocketIO.sleep')
    def test_update_images(self, mock_sleep, mock_emit, mock_get_image_data, mock_get_labels):
        """
        Test that the update_images function emits updates with image data.
        """
        mock_get_labels.return_value = ['label1', 'label2']
        mock_get_image_data.side_effect = [
            [('image_data1', 'image_name1')],
            [('image_data2', 'image_name2')],
        ]

        def stop_loop(*args, **kwargs):
            raise Exception("Stop loop")

        mock_sleep.side_effect = stop_loop  # Simulate breaking the loop after the first iteration

        with patch('builtins.print') as mock_print:
            try:
                update_images(self.socketio, self.r)
            except Exception:
                pass

        mock_get_labels.assert_called_once_with(self.r)
        mock_get_image_data.assert_any_call(self.r, 'label1')
        mock_get_image_data.assert_any_call(self.r, 'label2')
        mock_emit.assert_any_call(
            'update',
            {
                'label': 'label1',
                'images': ['image_data1'],
                'image_names': ['image_name1'],
            },
        )
        mock_emit.assert_any_call(
            'update',
            {
                'label': 'label2',
                'images': ['image_data2'],
                'image_names': ['image_name2'],
            },
        )
        mock_print.assert_called_once_with('Error updating images: Stop loop')


if __name__ == '__main__':
    unittest.main()

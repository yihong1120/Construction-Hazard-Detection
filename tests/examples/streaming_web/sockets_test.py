from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import socketio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.streaming_web.sockets import register_sockets
from examples.streaming_web.sockets import update_images


class TestSockets(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the streaming_web sockets.
    """

    def setUp(self) -> None:
        """
        Set up a SocketIO instance and mocks for Redis.
        """
        self.app = FastAPI()
        self.sio = socketio.AsyncServer(async_mode='asgi')
        self.sio_app = socketio.ASGIApp(
            self.sio, self.app, socketio_path='/socket.io',
        )
        self.client = TestClient(self.sio_app)
        self.redis_mock = MagicMock()

        # Register sockets to the SocketIO instance with mock Redis
        register_sockets(self.sio, self.redis_mock)

    @patch('examples.streaming_web.sockets.socketio.AsyncServer.emit')
    def test_handle_connect(self, mock_emit: MagicMock) -> None:
        """
        Test the 'connect' event handler to ensure a message is emitted.
        """
        # Trigger the connect event
        with self.client.websocket_connect('/socket.io') as websocket:
            try:
                websocket.receive_json()
            except Exception as e:
                print(f"WebSocket connection failed with error: {str(e)}")

        # Check if the message was emitted with correct data
        mock_emit.assert_called_once_with(
            'message', {'data': 'Connected'}, room=unittest.mock.ANY,
        )

    @patch('builtins.print')
    def test_handle_disconnect(self, mock_print: MagicMock) -> None:
        """
        Test the 'disconnect' event handler to
        ensure the correct message is printed.
        """
        # Simulate a client disconnection
        with self.client.websocket_connect('/socket.io') as websocket:
            try:
                websocket.close()
            except Exception as e:
                print(f"WebSocket disconnection failed with error: {str(e)}")

        # Check if 'Client disconnected' was printed
        mock_print.assert_called_once_with('Client disconnected')

    @patch('builtins.print')
    def test_handle_error(self, mock_print: MagicMock) -> None:
        """
        Test the 'error' event handler to ensure the error message is printed.
        """
        error_message = 'Mocked error'

        # Trigger the error event by simulating
        # an error during socket communication
        with self.client.websocket_connect('/socket.io') as websocket:
            try:
                self.sio.emit('error', {'data': error_message}, room=websocket)
            except Exception as e:
                print(f"Error event failed with error: {str(e)}")

        # Check if the error message was printed
        mock_print.assert_called_once_with(
            f"Error: {{'data': '{error_message}'}}",
        )

    @patch('examples.streaming_web.sockets.get_labels')
    @patch('examples.streaming_web.sockets.get_image_data')
    @patch('examples.streaming_web.sockets.socketio.AsyncServer.emit')
    @patch('examples.streaming_web.sockets.socketio.AsyncServer.sleep')
    async def test_update_images(
        self, mock_sleep: MagicMock, mock_emit: MagicMock,
        mock_get_image_data: MagicMock, mock_get_labels: MagicMock,
    ) -> None:
        """
        Test the 'update_images' function to
        ensure labels and images are emitted correctly.
        """
        # Mock the behavior of get_labels and get_image_data
        mock_get_labels.return_value = ['label1', 'label2']
        mock_get_image_data.side_effect = [
            [(b'image_data1', 'image1.png')],
            [(b'image_data2', 'image2.png')],
        ]

        # Limit loop iterations for test
        with patch(
            'examples.streaming_web.sockets.update_images',
            side_effect=StopIteration,
        ):
            try:
                await update_images(self.sio, self.redis_mock)
            except StopIteration:
                pass

        # Ensure sleep is called with 10 seconds
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_called_with(10)

        # Check that emit was called correctly for each label
        mock_emit.assert_any_call(
            'update',
            {
                'label': 'label1',
                'images': [b'image_data1'],
                'image_names': ['image1.png'],
            },
        )
        mock_emit.assert_any_call(
            'update',
            {
                'label': 'label2',
                'images': [b'image_data2'],
                'image_names': ['image2.png'],
            },
        )

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        del self.sio
        del self.redis_mock
        del self.app
        del self.client


if __name__ == '__main__':
    unittest.main()

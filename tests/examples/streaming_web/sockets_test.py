from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import socketio

from examples.streaming_web.sockets import register_sockets
from examples.streaming_web.sockets import update_images


class TestSockets(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the streaming_web sockets.
    """

    def setUp(self) -> None:
        """
        Sets up the test environment by initialising Socket.IO server
        and mocking RedisManager.
        """
        self.sio = socketio.AsyncServer()
        self.redis_manager = AsyncMock()
        register_sockets(self.sio, self.redis_manager)

        # Extracting the decorated event handlers from sio
        self.connect_handler = self.sio.handlers['/']['connect']
        self.disconnect_handler = self.sio.handlers['/']['disconnect']
        self.error_handler = self.sio.handlers['/']['error']

    async def test_connect(self) -> None:
        """
        Tests if a client can successfully connect to the server
        and receive a welcome message.
        """
        sid = 'test_sid'
        environ: dict[str, Any] = {}

        with patch.object(
            self.sio, 'emit', new_callable=AsyncMock,
        ) as mock_emit, patch.object(
            self.sio, 'start_background_task',
        ) as mock_background_task:
            await self.connect_handler(sid, environ)

            # Emit is awaited with the correct arguments
            mock_emit.assert_awaited_with(
                'message', {'data': 'Connected'}, room=sid,
            )
            # Ensure the background task is started
            mock_background_task.assert_called_once_with(
                update_images, self.sio, self.redis_manager,
            )

    async def test_disconnect(self) -> None:
        """
        Tests if a client disconnect event is logged successfully.
        """
        sid = 'test_sid'

        # Use Mock instead of AsyncMock to avoid warnings
        with patch('builtins.print', new_callable=Mock) as mock_print:
            await self.disconnect_handler(sid)

            # Check if print was called with the disconnect message
            mock_print.assert_called_once_with('Client disconnected')

    async def test_error(self) -> None:
        """
        Tests if errors are logged correctly during events.
        """
        sid = 'test_sid'
        error = Exception('Test error')

        # Use Mock instead of AsyncMock to avoid warnings
        with patch('builtins.print', new_callable=Mock) as mock_print:
            await self.error_handler(sid, error)

            # Check if print was called with the error message
            mock_print.assert_called_once_with(f"Error: {str(error)}")

    async def test_update_images(self) -> None:
        """
        Tests the update_images background task to ensure
        that images are emitted correctly to all clients.
        """
        with (
            patch.object(
                self.sio, 'sleep', new_callable=AsyncMock,
            ) as mock_sleep,
            patch.object(
                self.sio, 'emit', new_callable=AsyncMock,
            ) as mock_emit,
        ):
            mock_sleep.side_effect = [None, None, Exception('Stop Loop')]

            self.redis_manager.get_labels.return_value = ['label1', 'label2']
            self.redis_manager.fetch_latest_frames.side_effect = (
                lambda label: [
                    (f"image_data_{label}_1", f"image_name_{label}_1"),
                    (f"image_data_{label}_2", f"image_name_{label}_2"),
                ]
            )

            try:
                await update_images(self.sio, self.redis_manager)
            except Exception as e:
                print(f"Error updating images: {str(e)}")

            self.redis_manager.get_labels.assert_awaited()

            # Verifying that emit is called with the correct data for
            # each label
            mock_emit.assert_any_await(
                'update',
                {
                    'label': 'label1',
                    'images': [
                        'image_data_label1_1',
                        'image_data_label1_2',
                    ],
                    'image_names': [
                        'image_name_label1_1',
                        'image_name_label1_2',
                    ],
                },
            )
            mock_emit.assert_any_await(
                'update',
                {
                    'label': 'label2',
                    'images': [
                        'image_data_label2_1',
                        'image_data_label2_2',
                    ],
                    'image_names': [
                        'image_name_label2_1',
                        'image_name_label2_2',
                    ],
                },
            )


if __name__ == '__main__':
    unittest.main()

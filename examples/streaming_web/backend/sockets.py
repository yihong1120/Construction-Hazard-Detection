from __future__ import annotations

from typing import Any

import socketio


def register_sockets(sio: socketio.AsyncServer, redis_manager: Any) -> None:
    """
    Registers Socket.IO event handlers
    for connection, disconnection, and errors.

    Args:
        sio (socketio.AsyncServer): The Socket.IO server instance.
        redis_manager (Any): The RedisManager instance for Redis operations.
    """

    @sio.event
    async def connect(sid: str, environ: dict) -> None:
        """
        Handles new client connections by emitting a welcome message
        and starting the background task for image updates.

        Args:
            sid (str): The session ID of the connected client.
            environ (dict): The environment dictionary for the connection.
        """
        await sio.emit('message', {'data': 'Connected'}, room=sid)
        sio.start_background_task(update_images, sio, redis_manager)

    @sio.event
    async def disconnect(sid: str) -> None:
        """
        Handles client disconnections by logging the event.

        Args:
            sid (str): The session ID of the disconnected client.
        """
        print('Client disconnected')

    @sio.event
    async def error(sid: str, e: Exception) -> None:
        """
        Handles errors by logging the exception details.

        Args:
            sid (str): The session ID of the client experiencing the error.
            e (Exception): The exception raised during the event.
        """
        print(f"Error: {str(e)}")


async def update_images(sio: socketio.AsyncServer, redis_manager: Any) -> None:
    """
    Background task to update images for all labels at a set interval. Emits
    updated images to connected clients for each label.

    Args:
        sio (socketio.AsyncServer): The Socket.IO server instance.
        redis_manager (Any): The RedisManager instance to interact with Redis.
    """
    while True:
        try:
            # Waits for 1 seconds before updating images
            await sio.sleep(1)

            # Fetches all labels from Redis
            labels = await redis_manager.get_labels()

            for label in labels:
                # Fetches image data for each label
                image_data = await redis_manager.fetch_latest_frames(label)

                # Emits updated image data to all clients
                await sio.emit(
                    'update',
                    {
                        'label': label,
                        'images': [img for img, _ in image_data],
                        'image_names': [name for _, name in image_data],
                    },
                )
        except Exception as e:
            # Logs any exceptions that occur during the update process
            print(f"Error updating images: {str(e)}")
            break

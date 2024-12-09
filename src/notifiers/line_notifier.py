from __future__ import annotations

import os
from io import BytesIO
from typing import TypedDict

import aiohttp
import numpy as np
from dotenv import load_dotenv
from PIL import Image


class InputData(TypedDict):
    """
    A type definition for input data to the notifier.

    Attributes:
        message (str): The notification message.
        image (np.ndarray | None): An optional image in NumPy array format.
    """
    message: str
    image: np.ndarray | None


class ResultData(TypedDict):
    """
    A type definition for the result of a notification request.

    Attributes:
        response_code (int): The HTTP response status code.
    """
    response_code: int


class LineNotifier:
    """
    A class for managing notifications sent via the LINE Notify API.

    This class facilitates sending messages and optional images to LINE Notify,
    handling token retrieval and image preparation.
    """

    def __init__(self) -> None:
        """
        Initialises the LineNotifier instance.
        """
        load_dotenv()

    async def send_notification(
        self,
        message: str,
        image: np.ndarray | bytes | None = None,
        line_token: str | None = None,
    ) -> int:
        """
        Sends a notification via LINE Notify.

        Args:
            message (str): The message to be sent.
            image (np.ndarray | bytes | None): An optional image, provided as
                a NumPy array or bytes (e.g., encoded image data).
            line_token (str | None): The LINE Notify token. If not provided,
                attempts to load from environment variables.

        Returns:
            int: The HTTP response status code from LINE Notify.

        Raises:
            ValueError: If the LINE Notify token is not provided or not found
                in environment variables.
        """
        if not line_token:
            line_token = os.getenv('LINE_NOTIFY_TOKEN')
        if not line_token:
            raise ValueError(
                'LINE_NOTIFY_TOKEN not provided '
                'or found in environment variables.',
            )

        # Prepare the payload and headers for the request.
        payload = {'message': message}
        headers = {'Authorization': f"Bearer {line_token}"}

        # Use FormData to handle file attachment and form submission.
        form = aiohttp.FormData()
        # Add message to form data.
        for key, value in payload.items():
            form.add_field(key, value)

        if image is not None:
            # Prepare the image for upload.
            image_buffer = self._prepare_image_file(image)
            form.add_field(
                'imageFile', image_buffer,
                filename='image.png', content_type='image/png',
            )

        # Send the request asynchronously using aiohttp.
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://notify-api.line.me/api/notify',
                headers=headers,
                data=form,
            ) as response:
                return response.status

    def _prepare_image_file(self, image: np.ndarray | bytes) -> BytesIO:
        """Prepares an image file for sending.

        Converts a NumPy array or raw bytes into a PNG image suitable for
        upload.

        Args:
            image (np.ndarray | bytes): The image to prepare.

        Returns:
            BytesIO: A binary stream containing the PNG image.
        """
        if isinstance(image, bytes):
            # If the image is in bytes, decode it into a NumPy array.
            image = np.array(Image.open(BytesIO(image)))
        # Convert the NumPy array into a PIL Image.
        image_pil = Image.fromarray(image)
        # Save the image into a binary buffer in PNG format.
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)  # Reset the buffer position for reading.
        return buffer


async def main() -> None:
    """
    Demonstrates usage of the LineNotifier class.

    Sends a test message with an optional dummy image to LINE Notify.
    """
    notifier = LineNotifier()
    message = 'Hello, LINE Notify!'
    # Create a dummy image (100x100 black square) for testing.
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response_code = await notifier.send_notification(
        message, image=image, line_token='YOUR_LINE_TOKEN',
    )
    print(f"Response code: {response_code}")


if __name__ == '__main__':
    import asyncio
    # Run the example usage function in an asynchronous event loop.
    asyncio.run(main())

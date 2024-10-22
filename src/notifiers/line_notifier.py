from __future__ import annotations

import os
from io import BytesIO
from typing import TypedDict

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image


class InputData(TypedDict):
    message: str
    image: np.ndarray | None


class ResultData(TypedDict):
    response_code: int


class LineNotifier:
    """
    A class for managing notifications sent via the LINE Notify API.
    """

    def __init__(self):
        """
        Initialises the LineNotifier instance.
        """
        load_dotenv()

    def send_notification(
        self,
        message: str,
        image: np.ndarray | bytes | None = None,
        line_token: str | None = None,
    ) -> int:
        """
        Sends a notification via LINE Notify, optionally including an image.

        Args:
            message (str): The message to send.
            image (Optional[np.ndarray | bytes]): Image sent with the message.
                Defaults to None.
            line_token (Optional[str]): The LINE Notify token to use.

        Returns:
            int: The status code of the response.
        """
        # Get the LINE Notify token
        if not line_token:
            line_token = os.getenv('LINE_NOTIFY_TOKEN')
        if not line_token:
            raise ValueError(
                'LINE_NOTIFY_TOKEN not provided or in environment variables.',
            )

        payload = {'message': message}
        headers = {'Authorization': f"Bearer {line_token}"}

        # Prepare image if provided
        files = self._prepare_image_file(image) if image is not None else None

        # Send the request
        response = requests.post(
            'https://notify-api.line.me/api/notify',
            headers=headers,
            params=payload,
            files=files,
        )

        return response.status_code

    def _prepare_image_file(self, image: np.ndarray | bytes) -> dict:
        """
        Prepares the image file for the request.

        Args:
            image (np.ndarray | bytes): The image to send.

        Returns:
            dict: The files dictionary for the request.
        """
        if isinstance(image, bytes):
            image = np.array(Image.open(BytesIO(image)))
        image_pil = Image.fromarray(image)
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        return {'imageFile': ('image.png', buffer, 'image/png')}


# Example usage
def main():
    notifier = LineNotifier()
    message = 'Hello, LINE Notify!'
    # Create a dummy image for testing
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response_codes = notifier.send_notification(
        message, image=image, line_token='YOUR_LINE_TOKEN',
    )
    print(f"Response codes: {response_codes}")


if __name__ == '__main__':
    main()

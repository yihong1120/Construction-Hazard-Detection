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

    def __init__(self, line_token: str | None = None):
        """
        Initialises the LineNotifier instance.

        Args:
            line_token (Optional[str]): The LINE Notify token.
            If None, the token is attempted to be retrieved from
            environment variables. Defaults to None.
        """
        load_dotenv()
        self.line_token = line_token or os.getenv('LINE_NOTIFY_TOKEN')
        if not self.line_token:
            raise ValueError(
                'LINE_NOTIFY_TOKEN not provided or in environment variables.',
            )

    def send_notification(
        self,
        message: str,
        image: np.ndarray | None = None,
    ) -> int:
        """
        Sends a notification via LINE Notify, optionally including an image.

        Args:
            message (str): The message to send.
            label (Optional[str]): The label of the image_name.
            image (Optional[np.ndarray]): The image to send with the message.
                Defaults to None.

        Returns:
            response.status_code (int): The status code of the response.
        """
        headers = {'Authorization': f"Bearer {self.line_token}"}
        payload = {'message': message}
        files = {}

        if image is not None:
            if isinstance(image, bytes):
                # Convert bytes to NumPy array
                image = np.array(Image.open(BytesIO(image)))
            image_pil = Image.fromarray(image)
            buffer = BytesIO()
            image_pil.save(buffer, format='PNG')
            buffer.seek(0)
            files = {'imageFile': ('image.png', buffer, 'image/png')}
            response = requests.post(
                'https://notify-api.line.me/api/notify',
                headers=headers,
                params=payload,
                files=files,
            )
        else:
            response = requests.post(
                'https://notify-api.line.me/api/notify',
                headers=headers,
                params=payload,
            )

        return response.status_code


# Example usage
def main():
    notifier = LineNotifier(line_token='your_line_notify_token_here')
    message = 'Hello, LINE Notify!'
    # Create a dummy image for testing
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response_code = notifier.send_notification(message, image=image)
    print(f"Response code: {response_code}")


if __name__ == '__main__':
    main()

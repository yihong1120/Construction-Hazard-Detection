from __future__ import annotations

import os
from io import BytesIO
from typing import TypedDict, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from typing import TypedDict


class NotificationData(TypedDict):
    message: str
    image: Optional[np.ndarray]


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

    def send_notification(self, data: NotificationData) -> int:
        """
        Sends a notification via LINE Notify, optionally including an image.

        Args:
            data (NotificationData): The notification data including message
                and optional image.

        Returns:
            response.status_code (int): The status code of the response.
        """
        headers = {'Authorization': f"Bearer {self.line_token}"}
        payload = {'message': data['message']}
        files = {}

        if data['image'] is not None:
            if isinstance(data['image'], bytes):
                # Convert bytes to NumPy array
                image = np.array(Image.open(BytesIO(data['image'])))
            else:
                image = data['image']
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
    data: NotificationData = {
        'message': message,
        'image': image,
    }
    response_code = notifier.send_notification(data)
    print(f"Response code: {response_code}")


if __name__ == '__main__':
    main()

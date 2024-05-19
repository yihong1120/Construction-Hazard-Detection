import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import requests
from dotenv import load_dotenv

class LineNotifier:
    """
    A class for managing notifications sent via the LINE Notify API.
    """

    def __init__(self, line_token: Optional[str] = None):
        """
        Initialises the LineNotifier instance.

        Args:
            line_token (Optional[str]): The LINE Notify token. If None, the token is attempted to be retrieved from environment variables. Defaults to None.
        """
        load_dotenv()
        self.line_token = line_token or os.getenv('LINE_NOTIFY_TOKEN')
        if not self.line_token:
            raise ValueError("LINE_NOTIFY_TOKEN is not provided and is absent from the environment variables.")

    def send_notification(self, message: str, label: Optional[str] = None, image: Optional[np.ndarray] = None) -> int:
        """
        Sends a notification message through LINE Notify, with the option to include an image.

        Args:
            message (str): The message to send.
            label (Optional[str]): The label of the image_name.
            image (Optional[np.ndarray]): The image to send with the message. Defaults to None.

        Returns:
            response.status_code (int): The status code of the response.
        """
        headers = {'Authorization': f'Bearer {self.line_token}'}
        payload = {'message': message}
        files = {}

        if image is not None:
            files = {'imageFile': ('image.png', image, 'image/png')}
            response = requests.post('https://notify-api.line.me/api/notify', headers=headers, params=payload, files=files)
        else:
            response = requests.post('https://notify-api.line.me/api/notify', headers=headers, params=payload)

        return response.status_code

# Example usage
def main():
    notifier = LineNotifier(line_token='your_line_notify_token_here')
    message = "Hello, LINE Notify!"
    response_code = notifier.send_notification(message)
    print(f"Response code: {response_code}")

if __name__ == '__main__':
    main()

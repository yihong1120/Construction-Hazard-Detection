from __future__ import annotations

import os
from io import BytesIO

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image


class MessengerNotifier:
    """
    A class to handle sending notifications through Facebook Messenger
    """

    def __init__(self, page_access_token: str | None = None):
        """
        Initialises the MessengerNotifier with a Facebook page access token.

        Args:
            page_access_token (str, optional): The token for the Facebook page.
                Defaults to environment variable 'FACEBOOK_PAGE_ACCESS_TOKEN'.

        Raises:
            ValueError: If 'FACEBOOK_PAGE_ACCESS_TOKEN' is missing.
        """
        load_dotenv()
        self.page_access_token = page_access_token or os.getenv(
            'FACEBOOK_PAGE_ACCESS_TOKEN',
        )
        if not self.page_access_token:
            raise ValueError('FACEBOOK_PAGE_ACCESS_TOKEN missing.')

    def send_notification(
        self,
        recipient_id: str,
        message: str,
        image: np.ndarray | None = None,
    ) -> int:
        """
        Sends a notification to a specified recipient via Facebook Messenger.

        Args:
            recipient_id (str): The recipient's ID.
            message (str): The text message to send.
            image (np.ndarray): Optional image as a NumPy array (RGB format).

        Returns:
            int: The HTTP status code of the response.

        Notes:
            - If image is provided, it sends a message with image attachment.
            - Otherwise, sends a text message.
        """
        headers = {'Authorization': f"Bearer {self.page_access_token}"}
        url = (
            f"https://graph.facebook.com/v11.0/me/messages?"
            f"access_token={self.page_access_token}"
        )

        if image is not None:
            # Prepare image data
            image_pil = Image.fromarray(image)
            buffer = BytesIO()
            image_pil.save(buffer, format='PNG')
            buffer.seek(0)
            files = {'filedata': ('image.png', buffer, 'image/png')}

            # Send message with image attachment
            response = requests.post(
                url=url,
                headers=headers,
                files=files,
                data={
                    'recipient': f'{{"id":"{recipient_id}"}}',
                    'message': '{"attachment":{"type":"image","payload":{}}}',
                },
            )
        else:
            # Send plain text message
            payload = {
                'message': {'text': message},
                'recipient': {'id': recipient_id},
            }
            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
            )

        return response.status_code


# Example usage
def main():
    notifier = MessengerNotifier()
    recipient_id = 'your_recipient_id_here'
    message = 'Hello, Messenger!'
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # Example image (black)
    response_code = notifier.send_notification(
        recipient_id, message, image=image,
    )
    print(f"Response code: {response_code}")


if __name__ == '__main__':
    main()

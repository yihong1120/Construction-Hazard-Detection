import os
import requests
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from io import BytesIO


class MessengerNotifier:
    def __init__(self, page_access_token: str | None = None):
        load_dotenv()
        self.page_access_token = page_access_token or os.getenv('FACEBOOK_PAGE_ACCESS_TOKEN')
        if not self.page_access_token:
            raise ValueError('FACEBOOK_PAGE_ACCESS_TOKEN not provided or in environment variables.')

    def send_notification(self, recipient_id: str, message: str, image: np.ndarray | None = None) -> int:
        headers = {'Authorization': f"Bearer {self.page_access_token}"}
        payload = {'message': {'text': message}, 'recipient': {'id': recipient_id}}

        if image is not None:
            image_pil = Image.fromarray(image)
            buffer = BytesIO()
            image_pil.save(buffer, format='PNG')
            buffer.seek(0)
            files = {'filedata': ('image.png', buffer, 'image/png')}
            response = requests.post(
                f"https://graph.facebook.com/v11.0/me/messages?access_token={self.page_access_token}",
                headers=headers,
                files=files,
                data={'recipient': f'{{"id":"{recipient_id}"}}', 'message': '{"attachment":{"type":"image","payload":{}}}'}
            )
        else:
            response = requests.post(
                f"https://graph.facebook.com/v11.0/me/messages?access_token={self.page_access_token}",
                headers=headers,
                json=payload
            )
        return response.status_code


# Example usage
def main():
    notifier = MessengerNotifier()
    recipient_id = 'your_recipient_id_here'
    message = 'Hello, Messenger!'
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response_code = notifier.send_notification(recipient_id, message, image=image)
    print(f"Response code: {response_code}")


if __name__ == '__main__':
    main()

from __future__ import annotations

import logging
import os
from io import BytesIO

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image


class WeChatNotifier:
    """
    A class to handle sending notifications through WeChat Work.
    """

    def __init__(
        self, corp_id: str | None = None,
        corp_secret: str | None = None,
        agent_id: int | None = None,
    ):
        """
        Initialises the WeChatNotifier with authentication details.

        Args:
            corp_id (str, optional): The corporation ID.
                Defaults to environment variable 'WECHAT_CORP_ID'.
            corp_secret (str, optional): The corporation secret.
                Defaults to environment variable 'WECHAT_CORP_SECRET'.
            agent_id (int, optional): The agent ID.
                Defaults to env variable 'WECHAT_AGENT_ID' or 0 if not set.
        """
        load_dotenv()
        self.corp_id = corp_id or os.getenv('WECHAT_CORP_ID')
        self.corp_secret = corp_secret or os.getenv('WECHAT_CORP_SECRET')
        self.agent_id = agent_id or int(os.getenv('WECHAT_AGENT_ID') or 0)
        self.access_token = self.get_access_token()

    def get_access_token(self) -> str:
        """
        Retrieves the access token from WeChat Work API.

        Returns:
            str: The access token string.
        """
        url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?"
            f"corpid={self.corp_id}&corpsecret={self.corp_secret}"
        )
        response = requests.get(url)
        return response.json().get('access_token')

    def send_notification(
        self,
        user_id: str,
        message: str,
        image: np.ndarray | None = None,
    ) -> dict:
        """
        Sends a notification to a specified user in WeChat Work.

        Args:
            user_id (str): The user ID to send the notification to.
            message (str): The text message to send.
            image (np.ndarray): Optional image as a NumPy array (RGB format).

        Returns:
            dict: The response JSON from the WeChat API.
        """
        url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/message/send?"
            f"access_token={self.access_token}"
        )
        payload = {
            'touser': user_id,
            'msgtype': 'text',
            'agentid': self.agent_id,
            'text': {
                'content': message,
            },
            'safe': 0,
        }

        if image is not None:
            media_id = self.upload_media(image)
            payload = {
                'touser': user_id,
                'msgtype': 'image',
                'agentid': self.agent_id,
                'image': {
                    'media_id': media_id,
                },
                'safe': 0,
            }

        response = requests.post(url, json=payload)
        return response.json()

    def upload_media(self, image: np.ndarray) -> str:
        """
        Uploads an image media to WeChat Work.

        Args:
            image (np.ndarray): The image as a NumPy array (RGB format).

        Returns:
            str: The media ID of the uploaded image.
        """
        image_pil = Image.fromarray(image)
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/media/upload?"
            f"access_token={self.access_token}&type=image"
        )
        files = {'media': ('image.png', buffer, 'image/png')}
        response = requests.post(url, files=files)
        return response.json().get('media_id')


# Example usage
def main():
    notifier = WeChatNotifier()
    user_id = 'your_user_id_here'
    message = 'Hello, WeChat!'
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # Example image (black)
    _ = notifier.send_notification(user_id, message, image=image)

    # Log a static, non-sensitive message only
    logging.info('WeChat send_notification completed')


if __name__ == '__main__':
    # Configure a basic logger only for direct execution
    logging.basicConfig(level=logging.INFO)
    main()

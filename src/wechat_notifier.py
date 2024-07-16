import os
import requests
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from io import BytesIO


class WeChatNotifier:
    def __init__(self, corp_id: str | None = None, corp_secret: str | None = None, agent_id: int | None = None):
        load_dotenv()
        self.corp_id = corp_id or os.getenv('WECHAT_CORP_ID')
        self.corp_secret = corp_secret or os.getenv('WECHAT_CORP_SECRET')
        self.agent_id = agent_id or int(os.getenv('WECHAT_AGENT_ID'))
        self.access_token = self.get_access_token()

    def get_access_token(self) -> str:
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"
        response = requests.get(url)
        return response.json().get('access_token')

    def send_notification(self, user_id: str, message: str, image: np.ndarray | None = None) -> str:
        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={self.access_token}"
        payload = {
            "touser": user_id,
            "msgtype": "text",
            "agentid": self.agent_id,
            "text": {
                "content": message
            },
            "safe": 0
        }

        if image is not None:
            media_id = self.upload_media(image)
            payload = {
                "touser": user_id,
                "msgtype": "image",
                "agentid": self.agent_id,
                "image": {
                    "media_id": media_id
                },
                "safe": 0
            }

        response = requests.post(url, json=payload)
        return response.json()

    def upload_media(self, image: np.ndarray) -> str:
        image_pil = Image.fromarray(image)
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        url = f"https://qyapi.weixin.qq.com/cgi-bin/media/upload?access_token={self.access_token}&type=image"
        files = {'media': ('image.png', buffer, 'image/png')}
        response = requests.post(url, files=files)
        return response.json().get('media_id')


# Example usage
def main():
    notifier = WeChatNotifier()
    user_id = 'your_user_id_here'
    message = 'Hello, WeChat!'
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response = notifier.send_notification(user_id, message, image=image)
    print(response)


if __name__ == '__main__':
    main()

from __future__ import annotations

import os
from io import BytesIO
from typing import TypedDict

import aiohttp
import numpy as np
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

    async def send_notification(
        self,
        message: str,
        image: np.ndarray | bytes | None = None,
        line_token: str | None = None,
    ) -> int:
        if not line_token:
            line_token = os.getenv('LINE_NOTIFY_TOKEN')
        if not line_token:
            raise ValueError(
                'LINE_NOTIFY_TOKEN not provided or in environment variables.',
            )

        payload = {'message': message}
        headers = {'Authorization': f"Bearer {line_token}"}

        # 使用 FormData 來處理附檔
        form = aiohttp.FormData()
        # 將文字參數加入 form（LINE Notify 的參數需用 form-data 的方式提交）
        for k, v in payload.items():
            form.add_field(k, v)

        if image is not None:
            image_buffer = self._prepare_image_file(image)
            # 使用 add_field 指定檔案名稱及 content_type
            form.add_field(
                'imageFile', image_buffer,
                filename='image.png', content_type='image/png',
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://notify-api.line.me/api/notify',
                headers=headers,
                data=form,
            ) as response:
                return response.status

    def _prepare_image_file(self, image: np.ndarray | bytes) -> BytesIO:
        if isinstance(image, bytes):
            image = np.array(Image.open(BytesIO(image)))
        image_pil = Image.fromarray(image)
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

# Example usage


async def main():
    notifier = LineNotifier()
    message = 'Hello, LINE Notify!'
    # Create a dummy image for testing
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response_code = await notifier.send_notification(
        message, image=image, line_token='YOUR_LINE_TOKEN',
    )
    print(f"Response code: {response_code}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

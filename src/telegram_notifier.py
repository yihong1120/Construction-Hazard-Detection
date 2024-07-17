from __future__ import annotations

import asyncio
import os
from io import BytesIO

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from telegram import Bot


class TelegramNotifier:
    def __init__(self, bot_token: str | None = None):
        load_dotenv()
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.bot = Bot(token=self.bot_token)

    async def send_notification(
        self, chat_id: str,
        message: str,
        image: np.ndarray | None = None,
    ) -> str:
        if image is not None:
            image_pil = Image.fromarray(image)
            buffer = BytesIO()
            image_pil.save(buffer, format='PNG')
            buffer.seek(0)
            await self.bot.send_photo(
                chat_id=chat_id,
                photo=buffer,
                caption=message,
            )
        else:
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
            )
        return 200


# Example usage
async def main():
    notifier = TelegramNotifier()
    chat_id = 'your_chat_id_here'
    message = 'Hello, Telegram!'
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response = await notifier.send_notification(chat_id, message, image=image)
    print(response)


if __name__ == '__main__':
    asyncio.run(main())

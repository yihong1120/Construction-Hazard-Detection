import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from io import BytesIO
from telegram import Bot


class TelegramNotifier:
    def __init__(self, bot_token: str | None = None):
        load_dotenv()
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.bot = Bot(token=self.bot_token)

    def send_notification(self, chat_id: str, message: str, image: np.ndarray | None = None) -> str:
        if image is not None:
            image_pil = Image.fromarray(image)
            buffer = BytesIO()
            image_pil.save(buffer, format='PNG')
            buffer.seek(0)
            self.bot.send_photo(chat_id=chat_id, photo=buffer, caption=message)
        else:
            self.bot.send_message(chat_id=chat_id, text=message)
        return "Message sent"


# Example usage
def main():
    notifier = TelegramNotifier()
    chat_id = 'your_chat_id_here'
    message = 'Hello, Telegram!'
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response = notifier.send_notification(chat_id, message, image=image)
    print(response)


if __name__ == '__main__':
    main()

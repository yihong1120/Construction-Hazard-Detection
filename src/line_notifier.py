import requests
from datetime import datetime
from dotenv import load_dotenv
import os

class LineNotifier:
    """
    A class to handle notifications sent via LINE Notify API.
    """

    def __init__(self):
        """
        Initialises the LineNotifier instance by loading environment variables and setting the LINE Notify token.
        """
        load_dotenv()
        self.line_token = os.getenv('LINE_NOTIFY_TOKEN')

        if self.line_token is None:
            raise ValueError("LINE_NOTIFY_TOKEN is not set in the environment variables.")

    def send_notification(self, message: str) -> int:
        """
        Sends a notification message through LINE Notify.

        Args:
            message (str): The message to be sent.

        Returns:
            int: The HTTP status code returned by the LINE Notify API.
        """
        headers = {
            'Authorization': f'Bearer {self.line_token}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        payload = {'message': message}
        response = requests.post(
            'https://notify-api.line.me/api/notify',
            headers=headers,
            data=payload
        )
        return response.status_code

# If you want to use this script directly
if __name__ == '__main__':
    notifier = LineNotifier()  # Create an instance of LineNotifier
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f'[{current_time}] Warning: Safety protocol breached!'
    status = notifier.send_notification(message)
    print(f'Notification sent, status code: {status}')
import requests
from datetime import datetime
from dotenv import load_dotenv
import os

class LineNotifier:
    """
    A class to handle notifications sent via LINE Notify API.
    """

    def __init__(self, line_token: str = None):
        """
        Initialises the LineNotifier instance.

        Args:
            line_token (str, optional): The LINE Notify token. Defaults to None.
        """
        # If a token is provided, use it
        if line_token:
            self.line_token = line_token
        else:
            # Try to get the token from OS environment variables
            self.line_token = os.getenv('LINE_NOTIFY_TOKEN')

            # If the token is still None, load from .env file
            if not self.line_token:
                load_dotenv()
                self.line_token = os.getenv('LINE_NOTIFY_TOKEN')

            if not self.line_token:
                raise ValueError("LINE_NOTIFY_TOKEN is not provided and not set in the environment variables.")

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
    # Optionally pass the LINE Notify token as an argument
    token = input("Enter your LINE Notify token (press Enter to use ENV variable): ").strip()
    token = token if token else None  # Convert empty string to None

    notifier = LineNotifier(token)  # Create an instance of LineNotifier
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f'[{current_time}] Test message!'
    status = notifier.send_notification(message)
    print(f'Notification sent, status code: {status}')

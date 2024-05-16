import os
from datetime import datetime
from pathlib import Path
from typing import Optional
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

    def send_notification(self, message: str, label: Optional[str] = None, image_name: Optional[str] = None) -> int:
        """
        Sends a notification message through LINE Notify, with the option to include an image.

        Args:
            message (str): The message to send.
            label (Optional[str]): The label of the image_name.
            image_name (Optional[str]): The name of the image file to send with the notification. Defaults to None.

        Returns:
            int: The status code of the HTTP request.
        """
        headers = {'Authorization': f'Bearer {self.line_token}'}
        payload = {'message': message}
        files = None

        if image_name and label:
            image_path = Path('detected_frames') / label / image_name
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                return 404  # Not Found

            with image_path.open('rb') as image_file:
                files = {'imageFile': image_file}
                response = requests.post('https://notify-api.line.me/api/notify', headers=headers, data=payload, files=files)
        else:
            response = requests.post('https://notify-api.line.me/api/notify', headers=headers, data=payload)
        
        return response.status_code

# Example usage
def main():
    notifier = LineNotifier()  # Instantiate LineNotifier using an environment variable or .env file
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f'[{current_time}] Alert! Check the detection result.'

    # Send a notification along with an image
    label = 'some_label'  # Specify the label if needed
    image_name = 'detected_frame.jpg'  # Specify the image name, assuming it's in 'detected_frames' directory
    status = notifier.send_notification(message, label, image_name)
    print(f'Notification dispatched, status code: {status}')

if __name__ == '__main__':
    main()

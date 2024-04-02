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
        self.line_token = line_token or os.getenv('LINE_NOTIFY_TOKEN')
        if not self.line_token:
            load_dotenv()
            self.line_token = os.getenv('LINE_NOTIFY_TOKEN')
            if not self.line_token:
                raise ValueError("LINE_NOTIFY_TOKEN is not provided and is absent from the environment variables.")

    def send_notification(self, message: str, image_name: Optional[str] = None) -> int:
        """
        Sends a notification message through LINE Notify, with the option to include an image.

        Args:
            message (str): The message to be sent.
            image_name (Optional[str]): The name of the image to be sent. The image should be located in the 'demo_data' directory. Defaults to None.

        Returns:
            int: The HTTP status code returned by the LINE Notify API.
        """
        # Prepare the request
        headers = {
            'Authorization': f'Bearer {self.line_token}'
        }
        payload = {'message': message}
        image_path = Path('detected_frames') / image_name if image_name else None
        files = {'imageFile': open(image_path, 'rb')} if image_path else None
        
        # Send the notification
        response = requests.post(
            'https://notify-api.line.me/api/notify',
            headers=headers,
            data=payload,
            files=files
        )
        # Close the file after sending if it was opened
        if files:
            files['imageFile'].close()
        return response.status_code

# Example usage
def main():
    notifier = LineNotifier()  # Instantiate LineNotifier using an environment variable or .env file
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f'[{current_time}] Alert! Check the detection result.'
    
    # Send a notification along with an image
    image_name = 'detected_frame.jpg'  # Specify the image name, assuming it's in 'demo_data' directory
    status = notifier.send_notification(message, image_name)
    print(f'Notification dispatched, status code: {status}')

if __name__ == '__main__':
    main()
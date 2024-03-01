import requests
from datetime import datetime
from dotenv import load_dotenv
import os

class LineNotifier:
    """
    A class to manage notifications sent via the LINE Notify API.
    """

    def __init__(self, line_token: str = None):
        """
        Initialises the LineNotifier instance.

        Args:
            line_token (str, optional): The LINE Notify token. Defaults to None.
        """
        # Utilise the provided token if available
        if line_token:
            self.line_token = line_token
        else:
            # Attempt to retrieve the token from OS environment variables
            self.line_token = os.getenv('LINE_NOTIFY_TOKEN')

            # If the token remains None, attempt to load it from a .env file
            if not self.line_token:
                load_dotenv()
                self.line_token = os.getenv('LINE_NOTIFY_TOKEN')

            # Raise an error if the token is still not found
            if not self.line_token:
                raise ValueError("LINE_NOTIFY_TOKEN is not provided and is absent from the environment variables.")

    def send_notification(self, message: str, image_path: str = None) -> int:
        """
        Sends a notification message through LINE Notify, with the option to include an image.

        Args:
            message (str): The message to be sent.
            image_path (str, optional): The file path for the image to be sent. Defaults to None.

        Returns:
            int: The HTTP status code returned by the LINE Notify API.
        """
        headers = {
            'Authorization': f'Bearer {self.line_token}'
        }
        payload = {'message': message}
        # Attach the image if an image path is provided
        files = {'imageFile': open(image_path, 'rb')} if image_path else None
        response = requests.post(
            'https://notify-api.line.me/api/notify',
            headers=headers,
            data=payload,
            files=files
        )
        # Ensure the file is closed after sending
        if files:
            files['imageFile'].close()
        return response.status_code

# Example usage (no direct user input)
def main():
    notifier = LineNotifier()  # Instantiate LineNotifier using an environment variable
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f'[{current_time}] Alert! Check the detection result.'
    
    # Send a notification along with an image
    image_path = 'demo_data/prediction_visual.png'  # Set the default image path
    status = notifier.send_notification(message, image_path)
    print(f'Notification dispatched, status code: {status}')

if __name__ == '__main__':
    main()

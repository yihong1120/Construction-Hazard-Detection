import os
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from dotenv import load_dotenv
from typing import TypedDict
import numpy as np
import cv2
from datetime import datetime, timedelta

# JSON file to store image records
IMAGE_RECORD_FILE = 'config/image_records.json'
CHECK_INTERVAL_DAYS = 1  # Check for old images once a day

class InputData(TypedDict):
    message: str
    image: bytes | None


class ResultData(TypedDict):
    response_code: int


class LineMessenger:
    """
    A class for managing notifications sent via the LINE Messaging API and Cloudinary image upload.
    """

    def __init__(self, channel_access_token: str | None = None, recipient_id: str | None = None):
        """
        Initialises the LineMessenger instance.

        Args:
            channel_access_token (Optional[str]): The LINE Messaging API channel access token.
        """
        load_dotenv()
        self.channel_access_token = channel_access_token or os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
        self.recipient_id = recipient_id

        if not self.channel_access_token:
            raise ValueError('LINE_CHANNEL_ACCESS_TOKEN not provided or in environment variables.')
        
        if not self.recipient_id:
            raise ValueError('Recipient ID not provided.')

        # Configure Cloudinary
        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET')
        )

        # Load upload records
        self.image_records = self.load_image_records()

    def load_image_records(self) -> dict:
        """
        Loads image records from the JSON file.
        """
        try:
            if os.path.exists(IMAGE_RECORD_FILE):
                with open(IMAGE_RECORD_FILE, 'r') as file:
                    return json.load(file)
        except Exception as e:
            print(f"Failed to load image records: {e}")
        return {}

    def save_image_records(self) -> None:
        """
        Saves image records to the JSON file.
        """
        try:
            with open(IMAGE_RECORD_FILE, 'w') as file:
                json.dump(self.image_records, file)
        except Exception as e:
            print(f"Failed to save image records: {e}")

    def push_message(self, message: str, image_bytes: bytes | None = None) -> int:
        """
        Sends a message via LINE Messaging API, optionally including an image.

        Args:
            recipient_id (str): The recipient ID.
            message (str): The message to send.
            image_bytes (Optional[bytes]): The image bytes to upload to Cloudinary. Defaults to None.

        Returns:
            response.status_code (int): The status code of the response.
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.channel_access_token}"
        }

        message_data = {
            "to": self.recipient_id,
            "messages": [
                {
                    "type": "text",
                    "text": message
                }
            ]
        }

        public_id = None

        if image_bytes is not None:
            # Upload image to Cloudinary
            image_url, public_id = self.upload_image_to_cloudinary(image_bytes)
            if not image_url:
                raise ValueError("Failed to upload image to Cloudinary")

            print(f"Image URL: {image_url}")
            message_data["messages"].append({
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": image_url
            })

        # Send message via LINE API
        response = requests.post(
            'https://api.line.me/v2/bot/message/push',
            headers=headers,
            json=message_data,
        )

        # Print response for debugging
        if response.status_code == 200:
            print(f"Message sent successfully.")
        else:
            print(f"Error: {response.status_code}, {response.text}")

        if public_id:
            # Record image upload time
            self.record_image_upload(public_id)
            # Delete images older than 7 days, checked once a day
            self.delete_old_images_with_interval()

        return response.status_code

    def upload_image_to_cloudinary(self, image_data: bytes) -> tuple[str, str]:
        """
        Uploads an image to Cloudinary and returns the URL and public_id.

        Args:
            image_data (bytes): The image data to upload.

        Returns:
            tuple: The URL of the uploaded image and its public_id.
        """
        try:
            response = cloudinary.uploader.upload(image_data, resource_type="image")
            return response['secure_url'], response['public_id']
        except Exception as e:
            print(f"Failed to upload image to Cloudinary: {e}")
            return "", ""

    def record_image_upload(self, public_id: str) -> None:
        """
        Records the image upload time in the JSON file.
        """
        self.image_records[public_id] = datetime.now().isoformat()
        self.save_image_records()

    def delete_old_images_with_interval(self) -> None:
        """
        Deletes images that have been uploaded for more than 7 days, and only checks every day.
        """
        last_checked = self.image_records.get('last_checked')
        if last_checked is None or datetime.now() - datetime.fromisoformat(last_checked) > timedelta(days=CHECK_INTERVAL_DAYS):
            # Record check time
            self.image_records['last_checked'] = datetime.now().isoformat()
            self.delete_old_images()
            self.save_image_records()

    def delete_old_images(self) -> None:
        """
        Deletes images that have been uploaded for more than 7 days.
        """
        now = datetime.now()
        expired_public_ids = [
            public_id for public_id, upload_time in self.image_records.items()
            if public_id != 'last_checked' and now - datetime.fromisoformat(upload_time) > timedelta(days=7)
        ]

        for public_id in expired_public_ids:
            self.delete_image_from_cloudinary(public_id)
            del self.image_records[public_id]  # Remove expired image from records

        # Save records after deletion
        self.save_image_records()

    def delete_image_from_cloudinary(self, public_id: str) -> None:
        """
        Deletes an image from Cloudinary using its public_id.

        Args:
            public_id (str): The public_id of the image to delete.
        """
        try:
            response = cloudinary.uploader.destroy(public_id)
            if response.get('result') == 'ok':
                print(f"Image with public_id {public_id} successfully deleted from Cloudinary.")
            else:
                print(f"Failed to delete image with public_id {public_id}. Response: {response}")
        except Exception as e:
            print(f"Error deleting image from Cloudinary: {e}")


# Example usage
def main():
    channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', 'YOUR_LINE_CHANNEL')
    recipient_id = os.getenv('LINE_RECIPIENT_ID', 'YOUR_RECIPIENT_ID')

    messenger = LineMessenger(
        channel_access_token=channel_access_token, 
        recipient_id=recipient_id
    )
    
    message = 'Hello, LINE Messaging API!'

    # Create a black image for testing
    height, width = 480, 640
    frame_with_detections = np.zeros((height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.png', frame_with_detections)
    frame_bytes = buffer.tobytes()

    response_code = messenger.push_message(message, image_bytes=frame_bytes)
    print(f"Response code: {response_code}")


if __name__ == '__main__':
    main()

from __future__ import annotations

import logging

import requests


class BroadcastNotifier:
    """
    A class to connect to a broadcast system and send messages.
    """

    def __init__(self, broadcast_url: str):
        """
        Initialises the BroadcastNotifier with the broadcast system's URL.

        Args:
            broadcast_url (str): The URL of the broadcast API or service.
        """
        self.broadcast_url = broadcast_url
        self.logger = logging.getLogger(__name__)

    def broadcast_message(self, message: str) -> bool:
        """
        Sends a message to the broadcast system.

        Args:
            message (str): The message to broadcast.

        Returns:
            bool: True if the message was successfully sent, False otherwise.
        """
        try:
            # Example of sending a POST request to the broadcast system's API
            response = requests.post(
                self.broadcast_url, json={'message': message},
            )

            if response.status_code == 200:
                self.logger.info(f"Message broadcast successfully: {message}")
                return True
            else:
                self.logger.error(
                    f"Failed to broadcast message: {message}. "
                    f"Status code: {response.status_code}",
                )
                return False

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error broadcasting message: {e}")
            return False


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialise the BroadcastNotifier with the broadcast system URL
    notifier = BroadcastNotifier('http://localhost:8080/broadcast')

    # Send a test message
    status = notifier.broadcast_message('Test broadcast message')
    print(f"Broadcast status: {status}")


# Example usage:
if __name__ == '__main__':
    main()

from __future__ import annotations

import logging

import requests


class BroadcastNotifier:
    """Send text messages to an HTTP broadcast endpoint."""

    def __init__(self, broadcast_url: str) -> None:
        """Initialise the broadcast notifier.

        Args:
            broadcast_url: URL of the broadcast API or service.
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


def main() -> None:
    """Send a sample broadcast message for direct script execution."""
    logging.basicConfig(level=logging.INFO)

    notifier = BroadcastNotifier('http://localhost:8080/broadcast')

    status = notifier.broadcast_message('Test broadcast message')
    print(f"Broadcast status: {status}")


# Example usage:
if __name__ == '__main__':
    main()

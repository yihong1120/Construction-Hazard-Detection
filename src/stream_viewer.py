from __future__ import annotations

import cv2
from typing import TypedDict

class StreamConfig(TypedDict):
    stream_url: str
    window_name: str

class StreamViewer:
    """
    A class to handle the viewing of video streams (RTSP, HTTP, etc.).

    Attributes:
        stream_url (str): The URL of the video stream.
        window_name (str): The name of the window where the stream will be displayed.
    """

    def __init__(self, config: StreamConfig):
        """
        Initialises the StreamViewer instance with a stream URL and a window name.

        Args:
            config (StreamConfig): The configuration for the stream viewer.
        """
        self.stream_url = config['stream_url']
        self.window_name = config['window_name']
        self.cap = cv2.VideoCapture(self.stream_url)

    def display_stream(self):
        """
        Displays the video stream in a window.

        Continuously captures frames from the video stream and displays them.
        The loop breaks when 'q' is pressed or if the stream cannot be retrieved.
        """
        while True:
            # Capture the next frame from the stream.
            ret, frame = self.cap.read()

            # If the frame was successfully retrieved.
            if ret:
                # Display the video frame.
                cv2.imshow(self.window_name, frame)

                # Break the loop if 'q' is pressed.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print('Failed to retrieve frame.')
                break

        # Release the video capture object and close all OpenCV windows.
        self.release_resources()

    def release_resources(self):
        """
        Releases resources used by the StreamViewer.
        """
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Define the stream configuration
    config: StreamConfig = {
        'stream_url': 'https://cctv4.kctmc.nat.gov.tw/50204bfc/',
        'window_name': 'Stream Viewer'
    }
    
    viewer = StreamViewer(config)
    viewer.display_stream()

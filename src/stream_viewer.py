from __future__ import annotations

import cv2


class StreamViewer:
    """
    A class to handle the viewing of video streams (RTSP, HTTP, etc.).
    """

    def __init__(self, stream_url: str, window_name: str = 'Stream Viewer'):
        """
        Initialises the StreamViewer instance with a stream URL
        and a window name.

        Args:
            stream_url (str): The URL of the video stream.
            window_name (str): The name of the window where the stream
                               will be displayed.
        """
        self.stream_url = stream_url
        self.window_name = window_name
        self.cap = cv2.VideoCapture(self.stream_url)

    def display_stream(self):
        """
        Displays the video stream in a window.

        Continuously captures frames from the video stream and displays them.
        The loop breaks when 'q' is pressed or if the stream cannot be
        retrieved.
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


def main():
    """
    Main function to run the StreamViewer.
    """
    # Replace 'vide0_url' with your stream URL.
    video_url = (
        'https://cctv4.kctmc.nat.gov.tw/50204bfc/'
    )
    viewer = StreamViewer(video_url)
    viewer.display_stream()


if __name__ == '__main__':
    main()

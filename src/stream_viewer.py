from __future__ import annotations

import argparse
import os

import cv2


class StreamViewer:
    """Display RTSP, HTTP, or local video streams with OpenCV."""

    def __init__(
        self,
        stream_url: str,
        window_name: str = 'Stream Viewer',
    ) -> None:
        """Initialise the stream viewer.

        Args:
            stream_url: URL or path of the video stream.
            window_name: OpenCV window name used for display.
        """
        self.stream_url = stream_url
        self.window_name = window_name

        # TCP transport avoids packet loss issues on many RTSP cameras.
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        self.cap = cv2.VideoCapture(self.stream_url)

    def display_stream(self) -> None:
        """Display frames until the stream ends or the user presses ``q``."""
        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print('Failed to retrieve frame.')
                break

        self.release_resources()

    def release_resources(self) -> None:
        """Release OpenCV capture and window resources."""
        self.cap.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and run the stream viewer.

    Args:
        argv: Optional command-line argument list. ``None`` reads from
            ``sys.argv`` via ``argparse``.
    """
    parser = argparse.ArgumentParser(description='View a video stream.')
    parser.add_argument('stream_url', help='RTSP, HTTP, or local video URL.')
    parser.add_argument(
        '--window-name',
        default='Stream Viewer',
        help='OpenCV window name.',
    )
    args = parser.parse_args(argv)

    viewer = StreamViewer(args.stream_url, args.window_name)
    viewer.display_stream()


if __name__ == '__main__':
    main()

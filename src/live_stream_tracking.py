import argparse
import cv2
from ultralytics import YOLO
import youtube_dl
import datetime
from typing import Generator, Tuple

def get_live_video_url(youtube_url: str) -> str:
    """
    Retrieves the live video stream URL from a YouTube video.

    Args:
        youtube_url (str): The full URL to the YouTube video.

    Returns:
        str: The live stream URL of the YouTube video.
    """
    ydl_opts = {
        'format': 'bestaudio/best',  # Choose the best quality
        'quiet': True,  # Suppress output
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get("url")
        return video_url

def generate_detections(cap: cv2.VideoCapture, model: YOLO) -> Generator[Tuple, None, None]:
    """
    Yields detection results and the current timestamp from a video capture object frame by frame.

    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.
        model (YOLO): Loaded YOLO model object.

    Yields:
        Generator[Tuple]: Tuple of detection ids, detection data, the frame, and the current timestamp for each video frame.
    """
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # Exit if there is an issue with video capture

        # Get the current timestamp
        now = datetime.datetime.now()
        timestamp = now.timestamp()  # This converts the datetime to a Unix timestamp

        # Run YOLOv8 tracking on the frame, maintaining tracks between frames
        results = model.track(frame, persist=True)

        # Yield the detection results, the frame, and the current timestamp
        if results[0].boxes is not None:  # Check if there are detections
            yield results[0].boxes.id, results[0].boxes.data, frame, timestamp
        else:
            yield [], [], frame, timestamp  # Yield empty lists if no detections

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='YouTube video URL', required=True)
    parser.add_argument('--model', type=str, default='../models/yolov8n.pt', help='Path to the YOLOv8 model')
    args = parser.parse_args()

    # Load the YOLOv8 model
    model = YOLO(args.model)

    # Get YouTube live video stream URL
    live_stream_url = get_live_video_url(args.url)

    # Open the YouTube live video stream
    cap = cv2.VideoCapture(live_stream_url)

    # Use the generator function to get detections
    for ids, datas, frame, timestamp in generate_detections(cap, model):
        print("Timestamp:", timestamp)
        print("IDs:", ids)
        print("Data (xyxy format):")
        for data in datas:
            print(data)

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

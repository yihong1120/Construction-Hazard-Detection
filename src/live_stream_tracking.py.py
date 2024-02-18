import cv2
from ultralytics import YOLO
import youtube_dl

# Load the YOLOv8 model
model = YOLO('models/yolov8n.pt')

# YouTube video URL
youtube_url = 'https://www.youtube.com/watch?v=ZxL5Hm3mIBk'  # Replace this with your YouTube live video URL

# Function to get live video stream URL
def get_live_video_url(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',  # Choose quality
        'quiet': True,  # No printouts
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get("url")
        return video_url

# Get YouTube live video stream URL
live_stream_url = get_live_video_url(youtube_url)

# Open the YouTube live video stream
cap = cv2.VideoCapture(live_stream_url)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached or there is a problem
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

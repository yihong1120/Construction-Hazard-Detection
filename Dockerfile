# Use an official PyTorch image with CUDA 11.8 as the base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV MODEL_PATH=models/best_yolov8n.pt
ENV IMAGE_PATH=demo_data/prediction_visual.png
ENV LINE_NOTIFY_TOKEN=Your/Line/Notify/Token
ENV VIDEO_URL=Your/Video/URL

# Run demo.py when the container launches
CMD ["python", "src/demo.py"]

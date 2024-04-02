# Use an official PyTorch image with CUDA 11.8 as the base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Create a user to run the app (for better security)
RUN useradd -ms /bin/bash appuser

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
# Note: Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
# Use --chown to change the ownership of the copied files to the non-root user
COPY --chown=appuser:appuser . /app

# Switch to non-root user for better security
USER appuser

# Make port 80 available to the world outside this container
EXPOSE 80

# No need to define MODEL_PATH, IMAGE_PATH, LINE_NOTIFY_TOKEN, and VIDEO_URL as environment variables here
# as they will be dynamically read from configuration.json by the application

# Run main.py when the container launches
CMD ["python", "main.py"]
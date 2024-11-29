# Use an official PyTorch image with CUDA 12.1 as the base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Create a user to run the app (for better security)
RUN useradd -ms /bin/bash appuser

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set the timezone to Asia/Taipei
ENV TZ=Asia/Taipei

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY --chown=appuser:appuser . /app

# Create the 'logs' directory and set ownership if it does not exist
RUN [ ! -d /app/logs ] && mkdir /app/logs && chown appuser:appuser /app/logs || echo "/app/logs already exists"

# Switch to non-root user for better security
USER appuser

# Set ENTRYPOINT to allow dynamic arguments for the configuration file
ENTRYPOINT ["python3", "main.py"]

# Default CMD provides a placeholder configuration file
CMD ["--config", "/app/config/configuration.yaml"]

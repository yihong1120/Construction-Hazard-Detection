# Use an official PyTorch image with CUDA 12.1 as the base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including libGL
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libmariadb-dev-compat \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Create a non-root user for better security
RUN useradd -ms /bin/bash appuser

# Change the ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose port 5000 for the YOLO Server API
EXPOSE 5000

# Define the default command to run the app
CMD ["uvicorn", "examples.YOLO_server_api.app:sio_app", "--host", "0.0.0.0", "--port", "5000"]

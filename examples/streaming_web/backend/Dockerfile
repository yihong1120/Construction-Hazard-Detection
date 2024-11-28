# Use an official Python image with the required version as the base image
FROM python:3.12.7-slim

# Set the working directory within the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    build-essential \
    redis-tools \
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

# Expose the application port
EXPOSE 8000

# Define environment variables for Redis configuration (can be overridden at runtime)
# ENV PYTHONPATH=/app
ENV REDIS_HOST=127.0.0.1
ENV REDIS_PORT=6379
ENV REDIS_PASSWORD='passcode'

# Define the default command to run the application
CMD ["uvicorn", "examples.streaming_web.backend.app:sio_app", "--host", "0.0.0.0", "--port", "8000"]

# Use a lightweight Python image as the base
FROM python:3.12.7-slim

# Set the working directory
WORKDIR /app

# Install minimal system dependencies and NVIDIA CUDA runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl libgl1-mesa-glx libglib2.0-0 libffi-dev libssl-dev libpq-dev \
    build-essential gcc python3-dev && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvrtc-12-1 cuda-cudart-12-1 && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only requirements file to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip

# Remove development tools to reduce image size
RUN apt-get purge -y build-essential gcc python3-dev && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create a non-root user for security
RUN useradd -ms /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set default working directory for the user
WORKDIR /app

# Default command (can be overridden by docker-compose or other tools)
CMD ["python3"]

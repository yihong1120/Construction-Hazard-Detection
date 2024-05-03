# Use an official PyTorch image with CUDA 11.8 as the base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Create a user to run the app (for better security)
RUN useradd -ms /bin/bash appuser

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
# Note: Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
# Use --chown to change the ownership of the copied files to the non-root user
COPY --chown=appuser:appuser . /app

# Switch to non-root user for better security
USER appuser

# Expose the ports used by the Gunicorn servers
EXPOSE 8000 8001

# The CMD directive is used to run the script.
# Since we are using docker-compose to override this command, it can be left blank or set as a placeholder.
CMD ["echo", "Please use docker-compose to run this container."]
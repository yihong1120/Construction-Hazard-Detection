# Use the shared base image
FROM base:latest

# Set the working directory within the container
WORKDIR /app

# H.264 HTTP streaming endpoints transcode Redis image frames with ffmpeg.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
COPY . .

# Expose the application port
EXPOSE 8000

# Define environment variables for Redis configuration (can be overridden at runtime)
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_PASSWORD='password'

# Define the default command to run the application
CMD ["uvicorn", "examples.streaming_web.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

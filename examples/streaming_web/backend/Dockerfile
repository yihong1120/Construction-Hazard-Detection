# Use the shared base image
FROM base:latest

# Set the working directory within the container
WORKDIR /app

# Copy the application code into the container
COPY . .

# Expose the application port
EXPOSE 8000

# Define environment variables for Redis configuration (can be overridden at runtime)
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_PASSWORD='password'

# Define the default command to run the application
CMD ["uvicorn", "examples.streaming_web.backend.app:sio_app", "--host", "0.0.0.0", "--port", "8000"]

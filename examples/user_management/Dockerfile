# Use the shared base image
FROM yihong123/base:latest

# Set working directory
WORKDIR /app

# Copy the application code into the container
COPY . .

# Expose the port for the service
EXPOSE 8001

# Start the user management application
CMD ["uvicorn", "examples.user_management.app:app", "--host", "0.0.0.0", "--port", "8001"]

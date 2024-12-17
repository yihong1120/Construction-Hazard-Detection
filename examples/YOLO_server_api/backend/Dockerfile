# Use the shared base image
FROM yihong123/base:latest

# Set working directory
WORKDIR /app

# Copy YOLO Server API source code
COPY examples/YOLO_server_api /app

# Copy models/pt files
COPY models/pt /app/models/pt

# Expose the port for the YOLO server API
EXPOSE 6000

# Start the YOLO server API
CMD ["uvicorn", "examples.YOLO_server_api.app:app", "--host", "0.0.0.0", "--port", "6000"]

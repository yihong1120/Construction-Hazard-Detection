# Use the official Python slim image as the base image
FROM python:3.12.7-slim

# Set the working directory within the container
WORKDIR /app

# Install build dependencies and required libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libmariadb-dev-compat \
    libmariadb-dev \
    default-mysql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file into the container first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Create a non-root user for better security
RUN useradd -ms /bin/bash appuser

# Change ownership of the application directory to the newly created user
RUN chown -R appuser:appuser /app

# Switch to the non-root user for improved security
USER appuser

# Expose port 8001 for the user management service
EXPOSE 8001

# Define the default command to run the application using Uvicorn
# Run the FastAPI application on all available interfaces at port 8001
CMD ["uvicorn", "examples.user_management.app:app", "--host", "0.0.0.0", "--port", "8001"]

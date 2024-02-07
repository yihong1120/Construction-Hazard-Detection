# Use the official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt ./

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Set environment variables
# Replace 'your_line_notify_token' with the actual token or pass it via
# the `docker run` command using the `-e` option.
ENV LINE_TOKEN=your_line_notify_token

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the application when the container starts
CMD ["python", "./src/demo.py"]
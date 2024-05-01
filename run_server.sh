#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Ensure the script is running in the directory where the script is located.
cd "$(dirname "$0")"

# Install required Python packages from the requirements file.
echo "Installing required Python packages..."
pip install -r requirements.txt

# Start the Gunicorn server for the Model API on port 8000.
echo "Starting the Model Server on port 8000..."
gunicorn -w 1 -b 0.0.0.0:8000 "examples.Model-Server.app:app" &

# Wait for the model server to start properly
sleep 10

# Run the main Python script with configuration.
echo "Running the main application..."
python main.py --config config/configuration.json &

# Wait for the main application to initialize properly
sleep 10

# Start the Gunicorn server for the Stream Web app on port 8001.
echo "Starting Gunicorn server for the Stream Web on port 8001..."
gunicorn -w 1 -b 127.0.0.1:8001 "examples.Stream-Web.app:app"

echo "All processes have started successfully."

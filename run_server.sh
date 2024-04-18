#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Ensure the script is running in the directory where the script is located.
cd "$(dirname "$0")"

# Install required Python packages from the requirements file.
echo "Installing required Python packages..."
pip install -r requirements.txt

# Run the model API Python script.
echo "Starting the Model Server..."
python examples/Model-Server/model_api.py &

# Run the main Python script with configuration.
echo "Running the main application..."
python main.py --config config/configuration.json &

# Start the Gunicorn server for the Flask application.
echo "Starting Gunicorn server..."
gunicorn -w 4 -b 127.0.0.1:8000 "examples.Stream-Web.app:app"

echo "All processes have started successfully."

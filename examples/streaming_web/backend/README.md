
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web example

This section provides an example implementation of a Streaming Web application, designed to facilitate real-time camera feeds and updates. This guide provides information on how to use, configure, and understand the features of this application.

## Usage

1. **Run the server:**
    ```sh
    python app.py
    ```

    or

    ```sh
    uvicorn examples.streaming_web.backend.app:sio_app --host 127.0.0.1 --port 8000
    ```

2. **Open your web browser and navigate to:**
    ```sh
    http://localhost:8000
    ```

## Features

- **Real-Time Streaming**: Display real-time camera feeds with automatic updates every 5 seconds.
- **WebSocket Integration**: Utilises WebSocket for efficient real-time communication.
- **Dynamic Content Loading**: Automatically updates camera images without refreshing the page.
- **Responsive Design**: Adapts to various screen sizes for a seamless user experience.
- **Customisable Layout**: Adjust layout and styles using CSS.

## Configuration

The application can be configured through the following files:

- **app.py**: Main application file that starts the server and defines the routes.
- **routes.py**: Defines the web routes and their respective handlers.
- **sockets.py**: Manages WebSocket connections and events.
- **utils.py**: Utility functions for the application.
- **index.js**: Handles dynamic image updates for the main page.
- **camera.js**: Manages the camera image updates.
- **label.js**: Handles WebSocket communication and updates based on labels.
- **styles.css**: Contains the styles for the web application, ensuring a responsive and accessible design.

## File Overview

### app.py
The main entry point of the application that starts the server and sets up routes.

### routes.py
Defines the various web routes and their respective request handlers.

### sockets.py
Manages WebSocket connections, handling events such as connection, reconnection, and updates.

### utils.py
Contains utility functions used across the application for various tasks.

### index.js
Handles the periodic update of camera images on the main page using jQuery.

### camera.js
Updates the camera images on the page by refreshing the image source every 5 seconds.

### label.js
Manages WebSocket connections, handling updates based on the current label displayed on the page.

### styles.css
Defines the styling for the application, including responsive design, forced color adjustments for accessibility, and smooth image transitions.

Ensure to review and adjust the configuration settings in the respective files to suit your specific requirements.

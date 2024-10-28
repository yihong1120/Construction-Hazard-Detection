
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web example

This section provides an example implementation of a Streaming Web application, designed to facilitate real-time camera feeds and updates. This guide provides information on how to use, configure, and understand the features of this application.

## Usage

1. **Run the server:**
    ```sh
    python app.py
    ```

    Alternatively, use Gunicorn to start the application with an asynchronous worker:
    ```sh
    gunicorn -w 1 -k eventlet -b 127.0.0.1:8000 "examples.streaming_web.app:app"
    ```

2. **Access the application:**
   Open your web browser and navigate to:
    ```sh
    http://localhost:8000
    ```

## Features

- **Real-Time Streaming**: Displays real-time camera feeds with automatic updates every 5 seconds.
- **WebSocket Integration**: Utilises WebSocket for efficient real-time communication.
- **Dynamic Content Loading**: Automatically updates camera images without page refresh.
- **Responsive Design**: Adapts to various screen sizes for a seamless user experience.
- **Customisable Layout**: Modify layout and styles using CSS for a tailored appearance.

## Configuration and File Overview

The application can be customised and configured via the following key files:

- **app.py**: Main application file that starts the server and defines the routes.
- **routes.py**: Defines web routes and their respective handlers.
- **sockets.py**: Manages WebSocket connections and events.
- **utils.py**: Contains utility functions for the application.
- **index.js**: Handles dynamic image updates on the main page.
- **camera.js**: Manages the camera image updates.
- **label.js**: Handles WebSocket communication and label-based updates.
- **styles.css**: Contains the styles for the web application, ensuring responsive and accessible design.

Ensure to review and adjust configuration settings in these files as necessary for your environment.

## Nginx Configuration Example

To use Nginx as a reverse proxy for this FastAPI application, you may refer to the following key configuration parts. For a complete example configuration file, see `nginx_config_example.conf` in the `config/` directory.

1. **HTTP Redirect to HTTPS**: Redirect all HTTP requests to HTTPS for secure communication.
    ```nginx
    server {
        listen 80;
        server_name yourdomain.com;
        location / {
            return 301 https://$server_name$request_uri;
        }
    }
    ```

2. **HTTPS Configuration**: Enables SSL certificates and proxies static files and WebSocket requests.
    ```nginx
    server {
        listen 443 ssl;
        server_name yourdomain.com;

        # SSL certificate paths
        ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

        # Static files
        location /upload/ {
            alias /home/youruser/Documents/Construction-Hazard-Detection/static/uploads/;
            autoindex on;
            allow all;
        }

        # WebSocket configuration
        location /ws/ {
            proxy_pass http://127.0.0.1:8000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_buffering off;
            # Additional headers to forward client information
        }

        # General HTTP proxy
        location / {
            proxy_pass http://127.0.0.1:8000;
            # Forward headers for client and SSL status
        }
    }
    ```

3. **SSL Certificate Setup**

   To secure the server with SSL, a free SSL certificate from Let's Encrypt can be used. Here are the recommended steps:

   - **Install Certbot**: Use Certbot to handle automatic SSL certificate installation and renewal.
   - **Obtain SSL Certificates**: Run Certbot with your domain name to create SSL certificates:
     ```sh
     sudo certbot --nginx -d yourdomain.com
     ```
   - **Set Up Automatic Renewal**: Certbot handles automatic renewal; however, you can add a Cron job to check periodically:
     ```sh
     0 12 * * * /usr/bin/certbot renew --quiet
     ```

This setup ensures secure, automatic SSL management for the Nginx server.

## Additional Notes

For further customisation, refer to the `examples/streaming_web` folder and adjust files as per project needs. The code is modular, allowing you to update or replace components for scalability and maintenance.

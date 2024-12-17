
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Server API with User and Models Management

This project provides a complete implementation of a YOLO Server API integrated with a User Management System and a Models Management System. The server facilitates object detection using the YOLO model while providing robust user and model management capabilities.

---

## Features

### YOLO Server API
- **Object Detection**: Perform object detection on uploaded images using YOLO models.
- **Caching**: Improve performance by caching detection results.
- **Error Handling**: Robust error handling for seamless user experience.
- **Authentication**: Secures the API with role-based JWT authentication.

### User Management System
- **User Creation**: Add new user accounts with assigned roles.
- **User Deletion**: Remove users from the system.
- **User Updates**: Update user credentials like usernames and passwords.
- **Active Status Management**: Enable or disable user accounts.
- **Role Management**: Supports role-based permissions (`admin`, `model_manage`, `user`, `guest`).

### Models Management System
- **Model Uploads**: Upload and update YOLO models dynamically.
- **Model Retrieval**: Retrieve the latest version of a YOLO model.
- **Versioning**: Track model updates with timestamp comparisons.
- **Validation**: Ensure uploaded models meet system requirements.

---

## Quick Start

### 1. Install Dependencies
Ensure you have Python installed, and then install the required dependencies:
```sh
pip install -r requirements.txt
```

### 2. Configure the Application
Update the configuration in `config.py` or `.env`:
- Set the database connection URI (e.g., SQLite or PostgreSQL).
- Configure the JWT secret key and other environment variables.

### 3. Run the Server
You can start the server with:
```sh
uvicorn examples.YOLO_server_api.backend.app:sio_app --host 127.0.0.1 --port 8000
```

### 4. Send Requests
Use tools like `curl`, Postman, or your browser to interact with the API.

---

## Endpoints Overview

### YOLO Server API Endpoints

- **`POST /api/detect`**: Perform object detection on an uploaded image.
  - **Parameters**:
    - `file` (image): The image file for detection.
    - `model` (str): The YOLO model to use (default: `yolo11n`).
  - **Example**:
    ```sh
    curl -X POST -F "file=@path/to/image.jpg" -F "model=yolo11n" http://localhost:8000/detect
    ```

---

### User Management Endpoints

- **`POST /api/add_user`**: Add a new user.
  - **Parameters**:
    - `username` (str): The username.
    - `password` (str): The password.
    - `role` (str): Role (`admin`, `model_manage`, `user`, `guest`).
  - **Example**:
    ```sh
    curl -X POST -H "Content-Type: application/json" \
         -d '{"username":"admin","password":"securepassword","role":"admin"}' \
         http://localhost:8000/add_user
    ```

- **`DELETE /api/delete_user/<username>`**: Delete a user.
  - **Parameters**:
    - `username` (str): The username to delete.
  - **Example**:
    ```sh
    curl -X DELETE http://localhost:8000/delete_user/admin
    ```

- **`PUT /api/update_username`**: Update a user's username.
  - **Parameters**:
    - `old_username` (str): Existing username.
    - `new_username` (str): New username.
  - **Example**:
    ```sh
    curl -X PUT -H "Content-Type: application/json" \
         -d '{"old_username":"admin","new_username":"superadmin"}' \
         http://localhost:8000/update_username
    ```

- **`PUT /api/update_password`**: Update a user's password.
  - **Parameters**:
    - `username` (str): The username.
    - `new_password` (str): The new password.
  - **Example**:
    ```sh
    curl -X PUT -H "Content-Type: application/json" \
         -d '{"username":"admin","new_password":"newsecurepassword"}' \
         http://localhost:8000/update_password
    ```

- **`PUT /api/set_user_active_status`**: Set a user's active status.
  - **Parameters**:
    - `username` (str): The username.
    - `is_active` (bool): Active status (`true` or `false`).
  - **Example**:
    ```sh
    curl -X PUT -H "Content-Type: application/json" \
         -d '{"username":"admin","is_active":false}' \
         http://localhost:8000/set_user_active_status
    ```

---

### Models Management Endpoints

- **`POST /api/model_file_update`**: Update a YOLO model file.
  - **Parameters**:
    - `model` (str): Model name.
    - `file` (file): Model file to upload.
  - **Example**:
    ```sh
    curl -X POST -F "model=yolo11n" -F "file=@path/to/model.pt" http://localhost:8000/model_file_update
    ```

- **`POST /api/get_new_model`**: Retrieve an updated YOLO model file if available.
  - **Parameters**:
    - `model` (str): Model name.
    - `last_update_time` (ISO 8601 string): Last update time.
  - **Example**:
    ```sh
    curl -X POST -H "Content-Type: application/json" \
         -d '{"model":"yolo11n", "last_update_time":"2023-01-01T00:00:00"}' \
         http://localhost:8000/get_new_model
    ```

---

## Configuration Options

### YOLO Server Settings
Defined in `config.py`:
- `MODEL_PATH`: Path to YOLO model files.
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detection results.
- `CACHE_ENABLED`: Enable/disable caching of detection results.
- `AUTH_ENABLED`: Enable/disable JWT authentication.

### Database Configuration
Set the `SQLALCHEMY_DATABASE_URI` in `app.py` to match your database:
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
```

---

## File Structure

- **`app.py`**: Main server application.
- **`routers.py`**: Defines endpoints for YOLO Server, User Management, and Model Management.
- **`auth.py`**: Handles JWT-based authentication.
- **`cache.py`**: Implements caching for improved performance.
- **`detection.py`**: Performs object detection with YOLO models.
- **`user_operation.py`**: Handles user management operations.
- **`model_files.py`**: Handles model file updates and retrievals.
- **`config.py`**: Central configuration file for the application.

---

## Notes

- **Security**: Ensure that sensitive credentials (e.g., JWT secret, database passwords) are securely stored using environment variables.
- **Testing**: Use unit tests provided in `tests/` to validate your implementation.


🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Authentication and Authorisation Example

This directory contains an example implementation of user authentication and authorisation using FastAPI, Redis, SQLAlchemy (async engine), and JSON Web Tokens (JWT). It demonstrates:

- How to integrate FastAPI with an asynchronous SQLAlchemy database setup.
- How to use Redis for session-like caching (storing JWT jti lists and refresh tokens).
- How to incorporate FastAPI JWT Bearer tokens for secure endpoints.
- How to manage user operations (e.g., add, delete, update username/password, activate/deactivate users).
- How to automatically rotate your JWT secret key on a schedule (using APScheduler).

This example is primarily for demonstration and learning purposes. Please adapt it to your production environment with caution.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Environment Variables](#environment-variables)
5. [Running the Example](#running-the-example)
6. [JWT Secret Key Rotation](#jwt-secret-key-rotation)
7. [Rate Limiting](#rate-limiting)
8. [Authentication and User Management API](#authentication-and-user-management-api)
   - [Authentication Endpoints](#authentication-endpoints)
   - [User Management Endpoints](#user-management-endpoints)
9. [Code Explanation](#code-explanation)

## Project Structure

```
examples/auth/
├── __init__.py
├── auth.py
├── cache.py
├── config.py
├── database.py
├── jwt_config.py
├── jwt_scheduler.py
├── lifespan.py
├── models.py
├── redis_pool.py
├── routers.py
├── security.py
├── user_operation.py
├── README.md                  <- You are here
```

**Key files overview**:

- **`auth.py`**
  Defines core authentication logic, including creating access/refresh tokens and verifying refresh tokens.

- **`cache.py`**
  Provides helper functions for storing and retrieving user data (including jti lists and refresh tokens) in Redis.

- **`config.py`**
  Contains a `Settings` class that reads environment variables for JWT secret key and database configuration.

- **`database.py`**
  Sets up the asynchronous SQLAlchemy engine and session, and provides a base declarative model class.

- **`jwt_config.py`**
  Creates `jwt_access` and `jwt_refresh` instances using `fastapi_jwt.JwtAccessBearer`.

- **`jwt_scheduler.py`**
  Schedules a job to rotate the JWT secret key every 30 days via APScheduler.

- **`lifespan.py`**
  Defines a `global_lifespan` context manager that initialises and cleans up resources when the FastAPI app starts and stops (scheduler, Redis, database tables creation).

- **`models.py`**
  Defines SQLAlchemy ORM models (`User`, `Site`, `Violation`) and demonstrates relationships (e.g., many-to-many, one-to-many).

- **`redis_pool.py`**
  Provides a class for creating and managing a single Redis connection pool for both HTTP routes and WebSocket routes.

- **`routers.py`**
  Contains the FastAPI routes for authentication (login, logout, refresh) and user management (add, delete, update).

- **`security.py`**
  Contains a function to update the JWT secret key in the FastAPI application state.

- **`user_operation.py`**
  Provides functions for user CRUD operations (add, delete, update username/password, set active/inactive).

## Requirements

All required Python packages are listed in the `requirements.txt` file. The main dependencies include:

- **FastAPI**
- **fastapi-jwt** (for handling JWT-based security)
- **redis** (asynchronous client)
- **SQLAlchemy** (async engine for database interactions)
- **asyncmy** (async MySQL driver, or adapt to the RDBMS of your choice)
- **werkzeug** (for password hashing)
- **apscheduler** (for scheduled tasks)
- **python-dotenv** (for loading environment variables)

## Installation

1. **Clone the repository** (or copy the `examples/auth` folder into your project).

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install the dependencies** from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your `.env`** file (see [Environment Variables](#environment-variables)).

## Environment Variables

Create a `.env` file in the `examples/auth` directory (or your project’s root) with the following variables (adjust the values as needed):

```bash
# JWT secret key
JWT_SECRET_KEY="your_super_secret_key"

# SQLAlchemy database connection URI
# e.g. MySQL: mysql+asyncmy://<user>:<password>@<host>/<database_name>
DATABASE_URL="mysql+asyncmy://user:password@localhost/dbname"

# Redis connection details
REDIS_HOST="127.0.0.1"
REDIS_PORT="6379"
REDIS_PASSWORD=""
```

These variables are read by `config.py` at runtime via `pydantic_settings.BaseSettings`. You can also provide these variables via your system’s environment if you prefer.

## Running the Example

Below is an example `main.py` that ties everything together:

```python
# main.py
from fastapi import FastAPI
from examples.auth.lifespan import global_lifespan
from examples.auth.routers import auth_router, user_management_router

app = FastAPI(lifespan=global_lifespan)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(user_management_router, prefix="/users", tags=["User Management"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

1. **Run the application**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. **Verify** that the tables are automatically created in your database (due to the startup logic in `lifespan.py`).
3. **Access the docs** at: <http://127.0.0.1:8000/docs> or <http://127.0.0.1:8000/redoc>.

## JWT Secret Key Rotation

This example demonstrates automatic secret key rotation every 30 days using APScheduler. The relevant code is in `jwt_scheduler.py` and `security.py`.

- **Scheduler**: In `lifespan.py`, the `start_jwt_scheduler(app)` function is called to run a background task that rotates the secret key.
- **Secret Key Update**: The `update_secret_key(app)` function generates a new secret key and stores it in `app.state.jwt_secret_key`.

> **Note**: This mechanism is purely demonstrative. Changing the secret key will invalidate all existing tokens. In a real production system, you may need a more refined key management strategy.

## Rate Limiting

A custom rate limiter is defined in `cache.py` (`custom_rate_limiter`). It inspects the user’s role and applies different rate limits:

- `guest` role:
  - **Max requests**: 24 requests
  - **Window**: 24 hours

- All other roles:
  - **Max requests**: 3000 requests
  - **Window**: 1 minute

If the rate is exceeded, an HTTP `429` (Too Many Requests) is returned.

## Authentication and User Management API

### Authentication Endpoints

#### `POST /auth/login`
- **Body**: `{"username": "your_username", "password": "your_password"}`
- **Response**:
  ```json
  {
    "access_token": "<JWT_ACCESS_TOKEN>",
    "refresh_token": "<JWT_REFRESH_TOKEN>",
    "role": "<user_role>",
    "username": "<username>",
    "user_id": <user_id>
  }
  ```
  A successful response returns both an `access_token` (short-lived) and a `refresh_token` (long-lived).

#### `POST /auth/logout`
- **Body**: `{"refresh_token": "<JWT_REFRESH_TOKEN>"}`
- **Headers**: `Authorization: Bearer <JWT_ACCESS_TOKEN>`
- **Behaviour**:
  - Revokes the provided refresh token from Redis.
  - Removes the associated JWT `jti` from Redis.
  - If the access token is invalid or expired, it still clears out local references.

#### `POST /auth/refresh`
- **Body**: `{"refresh_token": "<JWT_REFRESH_TOKEN>"}`
- **Response**:
  ```json
  {
    "access_token": "<NEW_JWT_ACCESS_TOKEN>",
    "refresh_token": "<NEW_JWT_REFRESH_TOKEN>",
    "message": "Token refreshed successfully."
  }
  ```
  - Validates the provided refresh token.
  - Generates new access and refresh tokens if valid.

### User Management Endpoints

All these endpoints **require** a valid `Bearer` token with the `admin` role.

#### `POST /users/add_user`
- **Body**:
  ```json
  {
    "username": "new_username",
    "password": "new_password",
    "role": "user"
  }
  ```
- **Usage**: Creates a new user record. The role can be one of `admin`, `model_manager`, `user`, or `guest`.

#### `POST /users/delete_user`
- **Body**:
  ```json
  {
    "username": "username_to_delete"
  }
  ```
- **Usage**: Deletes the specified user from the database.

#### `PUT /users/update_username`
- **Body**:
  ```json
  {
    "old_username": "old_name",
    "new_username": "new_name"
  }
  ```
- **Usage**: Updates the username for the given user.

#### `PUT /users/update_password`
- **Body**:
  ```json
  {
    "username": "the_username",
    "new_password": "the_new_password"
  }
  ```
- **Usage**: Updates the password (re-hashes and stores it) for the specified user.

#### `PUT /users/set_user_active_status`
- **Body**:
  ```json
  {
    "username": "the_username",
    "is_active": true
  }
  ```
- **Usage**: Activates or deactivates the user’s account.

## Code Explanation

1. **JWT Logic**:
   - The `JwtAccessBearer` and `JwtRefreshBearer` (`jwt_access` and `jwt_refresh`) are configured with the secret key from `Settings`.
   - In `auth.py`, `create_access_token` validates user credentials and generates short-lived access tokens with a unique `jti`, which is tracked in Redis. A long-lived refresh token is also generated.

2. **Redis Storage**:
   - The user’s data structure in Redis contains:
     ```json
     {
       "db_user": {
         "id": <int>,
         "username": "<str>",
         "role": "<str>",
         "is_active": <bool>
       },
       "jti_list": ["<list_of_jti_strings>"],
       "refresh_tokens": ["<list_of_active_refresh_tokens>"]
     }
     ```
   - `get_user_data` and `set_user_data` abstract the JSON serialisation/deserialisation.

3. **Database**:
   - `database.py` sets up an asynchronous SQLAlchemy engine.
   - `models.py` shows how to define asynchronous ORM models, including many-to-many (`User` <-> `Site`) and one-to-many (`Site` -> `Violation`).

4. **User Operations**:
   - `user_operation.py` includes all CRUD operations for the `User` model.

5. **Lifespan and Scheduler**:
   - The `global_lifespan` context manager in `lifespan.py` starts the APScheduler job for rotating JWT secrets every 30 days and initialises Redis.
   - It also creates tables in the database if they do not exist.

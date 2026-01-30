
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Database Management Backend

This directory contains backend logic for managing user groups, permissions, sites, and stream configurations using FastAPI. It leverages asynchronous database interactions via SQLAlchemy and JWT-based authentication for secure access control.

Below is a detailed explanation of the files within `examples/db_management/`, instructions on running the backend, and recommendations for production deployment.

## File Structure

```
examples/db_management/
├── __init__.py
├── app.py
├── deps.py
├── routers/
│   ├── __init__.py
│   ├── auth.py
│   ├── features.py
│   ├── groups.py
│   ├── sites.py
│   ├── streams.py
│   └── users.py
├── schemas/
│   ├── __init__.py
│   ├── auth.py
│   ├── feature.py
│   ├── group.py
│   ├── site.py
│   ├── stream_config.py
│   └── user.py
└── services/
    ├── __init__.py
    ├── auth_service.py
    ├── feature_service.py
    ├── group_service.py
    ├── site_service.py
    ├── stream_config_service.py
    └── user_service.py
```

---

## Core Components Overview

### `app.py`

* **Purpose**: Main entry point for the FastAPI application, setting up routing, CORS, middleware, and the application's lifespan.
* **Key Highlights**:

  * Registers routers for authentication, user management, groups, sites, features, and stream configurations.
  * Configures JWT authentication and Redis caching mechanisms.
  * Starts the FastAPI server using `uvicorn`.

**Running the Application**:

```bash
uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8000
```

---

### `deps.py`

* **Purpose**: Provides dependency functions for authentication, role verification, and permission checking across different routes.
* **Key Functions**:

  * `get_current_user`: Retrieves and validates the current user from a JWT token.
  * `require_admin`: Ensures the user has admin privileges.
  * `require_super_admin`: Restricts endpoint usage exclusively to super administrators.
  * `_site_permission`: Checks if a user has permissions to perform operations on specific sites or groups.

---

### `routers/`

Defines API endpoints categorised into separate router modules:

* **`auth.py`**:

  * Login, logout, token refreshing endpoints using JWT.
* **`users.py`**:

  * User management endpoints: create, list, delete users, update passwords, usernames, and active status.
* **`groups.py`**:

  * Group management endpoints: create, update, delete, and list user groups.
* **`sites.py`**:

  * Site management endpoints: create, update, delete sites, and manage user-site associations.
* **`features.py`**:

  * Feature toggles management: create, update, delete features, and associate them with groups.
* **`streams.py`**:

  * Manage stream configurations: create, update, delete, and list streaming configurations, including group limits.

---

### `schemas/`

Contains Pydantic schemas for validating request/response bodies across endpoints:

* `auth.py`: JWT authentication schemas (`UserLogin`, `RefreshRequest`, `LogoutRequest`, `TokenPair`).
* `user.py`: User-related operations (`UserCreate`, `UserRead`, password/username updates).
* `group.py`: Group operations schemas (`GroupCreate`, `GroupUpdate`, `GroupDelete`, `GroupRead`).
* `site.py`: Site operations schemas (`SiteCreate`, `SiteUpdate`, `SiteDelete`, `SiteRead`, `SiteUserOp`).
* `feature.py`: Feature management schemas (`FeatureCreate`, `FeatureUpdate`, `FeatureDelete`, `FeatureRead`, `GroupFeatureUpdate`).
* `stream_config.py`: Stream configuration schemas (`StreamConfigCreate`, `StreamConfigUpdate`, `StreamConfigRead`).

---

### `services/`

Implements core database logic using asynchronous SQLAlchemy:

* `auth_service.py`: Authentication logic for handling JWT tokens, caching via Redis, and user login/logout workflows.
* `user_service.py`: User creation, deletion, updating usernames/passwords, and activating/deactivating users.
* `group_service.py`: Creating, updating, listing, and deleting user groups.
* `site_service.py`: Creating, updating, listing, and deleting sites, including associated cleanup tasks.
* `feature_service.py`: Managing feature toggles, assigning features to groups, and listing available features.
* `stream_config_service.py`: Managing stream configurations, enforcing group stream limits.

---

## Running the Application

### 1. **Install Dependencies**

Make sure to install required dependencies:

```bash
pip install fastapi uvicorn sqlalchemy aiomysql redis pydantic PyJWT python-multipart
```

### 2. **Setup and Run Redis**

Redis is required for caching and managing authentication sessions:

* Via local Redis server:

  ```bash
  redis-server
  ```
* Via Docker:

  ```bash
  docker run -p 6379:6379 redis
  ```

### 3. **Database Configuration**

Ensure your database settings (MySQL/other supported by SQLAlchemy) are properly configured in your application's `.env` or settings module.

### 4. **Run FastAPI Application**

```bash
uvicorn examples.db_management.app:app --host 127.0.0.1 --port 8000
```

Access API documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Testing the Backend

Tests are placed in the project's top-level `tests/` directory:

```bash
pytest --cov=examples.db_management --cov-report=term-missing
```

The tests validate:

* Authentication flows (`auth_service_test.py`)
* User, group, site, and feature CRUD operations
* Permission handling via dependencies (`deps_test.py`)

---

## Production Considerations

1. **Security**:

   * **CORS**: Configure allowed origins explicitly for enhanced security.
   * **JWT Tokens**: Ensure JWT secret keys and token lifetimes are securely managed.

2. **Performance & Scaling**:

   * Use `Gunicorn` with multiple `uvicorn` workers for high concurrency.
   * Consider Redis clustering for high availability.

3. **Database Management**:

   * Regularly back up your database.
   * Monitor performance and set up logging for database queries.

4. **Secure Connections**:

   * Deploy the application behind a reverse proxy with TLS/SSL enabled (e.g., Nginx).

5. **Logging & Monitoring**:

   * Integrate structured logging.
   * Use monitoring tools like Prometheus/Grafana for observability.

---

## Further Information & Support

For additional details on specific functionality, authentication flows (`examples/auth`), or support regarding advanced use cases (e.g., integration with frontend applications, deployments, or feature expansions), refer to the broader repository or reach out to the development team.

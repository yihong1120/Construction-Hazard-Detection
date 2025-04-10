
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Violation Records Management Example

This example demonstrates how to build a comprehensive violation records management system using FastAPI, SQLAlchemy, and related technologies. It showcases:

- CRUD operations for managing violation records.
- Secure image handling and storage.
- JWT-based user authentication and site-based access control.
- Multi-language keyword search using synonym expansion.
- Structured schemas using Pydantic for API responses and requests.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Example](#running-the-example)
7. [API Endpoints](#api-endpoints)
8. [Additional Notes](#additional-notes)

## Overview

The violation records management server provides a robust API to handle operations related to violations recorded at various sites. It features:

- Retrieval of the accessible site list for each user.
- Paginated retrieval and filtering of violation records by keyword, time range, and site.
- Detailed view of individual violations, including metadata and associated images.
- An upload function for new violations, supporting comprehensive metadata.

## Prerequisites

1. **Python 3.9+** – Recommended for FastAPI, asynchronous support, and typing.
2. **Redis** – Optional, dependent on your overall project setup (not strictly required unless your application uses it).
3. **SQLAlchemy and an asynchronous database** (e.g., PostgreSQL or MySQL) – for storing violation data.
4. **FastAPI and dependencies** – for creating and handling APIs.

## Directory Structure

Below is an outline of the `examples/violation_records` directory:

```
examples/violation_records/
├── app.py
├── routers.py
├── schemas.py
├── search_utils.py
├── violation_manager.py
├── static/ (created automatically for images)
└── README.md
```

- **`app.py`**: The main FastAPI application entry point.
- **`routers.py`**: FastAPI route definitions for handling violation records.
- **`schemas.py`**: Pydantic models defining request and response schemas.
- **`search_utils.py`**: A keyword search utility featuring synonym expansion.
- **`violation_manager.py`**: Manages storing violations and images to disk/database.
- **`static/`**: Directory used to store uploaded violation images (created automatically).

## Installation

1. **Clone this repository** or copy the `examples/violation_records` folder into your project.
2. **Create and activate** a Python virtual environment (optional, but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install the required packages**:

   ```bash
   pip install fastapi sqlalchemy databases[asyncpg] fastapi_jwt pydantic aiofiles uvicorn
   ```

## Configuration

- **Database & JWT**: Configure your database connection and JWT settings (e.g., secret key, expiry times) via environment variables or a config file, depending on your project's requirements.
- **Images & Static Directory**: By default, images are stored under the `static/` folder. In production, you may wish to use a cloud storage service or a dedicated CDN.

## Running the Example

To run the server locally using Uvicorn:

```bash
python examples/violation_records/app.py
```

By default, this starts FastAPI at `0.0.0.0:8081`. Adjust the host and port as needed.

## API Endpoints

### 1. Retrieve Accessible Sites

```
GET /my_sites
```
- Returns a list of sites accessible to the currently logged-in user.

### 2. Retrieve Violation Records

```
GET /violations
```
- Supports filters such as `site_id`, `keyword`, `start_time`, `end_time`, and pagination via `limit`/`offset`.

### 3. Retrieve Single Violation

```
GET /violations/{violation_id}
```
- Returns detailed information about a specific violation record, including metadata and associated images.

### 4. Retrieve Violation Image

```
GET /get_violation_image?image_path=...
```
- Returns the image stored for a particular violation record. The path must be relative to the `static/` directory.

### 5. Upload New Violation

```
POST /upload
```
- Allows the creation of a new violation record with associated metadata and an image to be stored in `static/`.

## Additional Notes

- **Security**: Ensure JWT tokens are properly validated and that each endpoint enforces user permissions (e.g., users may only access sites they are authorised to view).
- **Performance**: Consider indexing frequently queried columns in your database. For large-scale deployments, also consider storing images in a cloud storage service rather than locally.
- **Multi-Language Search**: The `search_utils.py` file can be extended to handle more languages or more complex synonym sets, depending on your project's scope.
- **Further Enhancements**:
  - Additional fields or logic in `violation_manager.py` (e.g., storing versioned images).
  - Enhanced pagination strategies or sorting criteria.
  - Webhooks or notifications upon new violation creation (integration with external systems).

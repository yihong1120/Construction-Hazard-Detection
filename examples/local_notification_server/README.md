
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Local Notification Server Example

This directory provides an example of how to integrate Firebase Cloud Messaging (FCM) into your FastAPI application for sending push notifications to client devices (e.g., Android, iOS, web). It demonstrates:

- How to initialise and use the Firebase Admin SDK to send FCM messages.
- How to store and manage device tokens in Redis (organised by user or site).
- How to design FastAPI endpoints for storing tokens, deleting tokens, and sending notifications.
- How to translate notification messages to multiple languages before dispatch (via a simple translation dictionary).

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setting Up a Firebase Project](#setting-up-a-firebase-project)
4. [Generating the Service Account Credential File](#generating-the-service-account-credential-file)
5. [Directory Structure](#directory-structure)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the Example](#running-the-example)
9. [Using the API](#using-the-api)
10. [Additional Notes](#additional-notes)

## Overview

This local notification server listens for requests to send push notifications to client devices via FCM. It:

1. Stores FCM device tokens in Redis, keyed by user ID.
2. Translates violation/warning messages into the recipient’s language using a simple dictionary-based approach.
3. Dispatches notifications in bulk, grouping by language.

When a violation event is detected (e.g., “no hardhat,” “too close to machinery,” etc.), the server can push customised notifications with relevant content and optional images.

## Prerequisites

1. **Redis** – the application depends on storing device tokens in Redis.
2. **A working FastAPI application** – see the main `examples/auth` example in this repository for authentication and general scaffolding.
3. **Firebase Project** – you must have a Firebase project with Cloud Messaging enabled.
4. **Python 3.9+** (recommended) – for running the code.

## Setting Up a Firebase Project

1. Go to the [Firebase Console](https://console.firebase.google.com/).
2. Create a new project (or select an existing one).
3. Once created, go to **Project Settings** -> **General** to see your **Project ID** (e.g., `construction-harzard-detection`).
4. Under the **Cloud Messaging** section, ensure that FCM is enabled by default.

## Generating the Service Account Credential File

1. In your Firebase project, navigate to **Project Settings** -> **Service Accounts** (under the Build or Settings section).
2. Click **Generate new private key** under the Firebase Admin SDK section.
   - This will download a `.json` file containing the service account credentials (private key, client email, etc.).
3. **Rename** (or keep) the `.json` file name as you see fit. In this example, we have used:
   ```
   construction-harzard-detection-firebase-adminsdk-fbsvc-ca9d30aff7.json
   ```
4. Place this file in the `examples/local_notification_server/` directory (or somewhere secure).
   - For production, you should keep this file outside the repository and manage secrets carefully.

## Directory Structure

Below is an overview of the files in `examples/local_notification_server`:

```
examples/local_notification_server/
├── app.py
├── fcm_service.py
├── lang_config.py
├── routers.py
├── schemas.py
├── construction-harzard-detection-firebase-adminsdk-fbsvc-ca9d30aff7.json  (Your Firebase credentials)
└── README.md                        <- You are here
```

- **`app.py`**
  The main FastAPI application that starts up with `global_lifespan` (from the parent auth example) and includes the notification router.

- **`fcm_service.py`**
  Responsible for sending push notifications via the Firebase Admin SDK. It handles batch dispatch and error logging.

- **`lang_config.py`**
  Contains translation dictionaries for multiple languages and a simple `Translator` class that translates violation messages based on keys and placeholders.

- **`routers.py`**
  Provides FastAPI routes for managing tokens (`/store_token`, `/delete_token`) and sending notifications (`/send_fcm_notification`).

- **`schemas.py`**
  Contains Pydantic models for request bodies (e.g., `TokenRequest`, `SiteNotifyRequest`).

- **`construction-harzard-detection-firebase-adminsdk-fbsvc-ca9d30aff7.json`**
  An example Firebase service account JSON file (rename and replace with your real one).

## Installation

1. **Clone this repository** (or copy the `examples/local_notification_server` folder into your project).
2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. **Install the dependencies** (in addition to those from the parent `examples/auth`):
   ```bash
   pip install -r requirements.txt
   ```
   (Or manually install `firebase-admin` and any other needed packages if they’re not already in your environment.)

## Configuration

1. **Firebase credential file path**
   Make sure the file path in `fcm_service.py` (or wherever you initialise Firebase) matches your `.json` file name and path. For example:
   ```python
   cred_path = (
       "examples/local_notification_server/"
       "construction-harzard-detection-firebase-adminsdk-fbsvc-ca9d30aff7.json"
   )
   ```
   If you rename or relocate the file, update this path accordingly.

2. **Redis**
   The Redis connection is managed by the parent `examples/auth` modules (`redis_pool.py`, etc.). Ensure your `.env` or environment variables are set for Redis (`REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` if any).

3. **Database**
   Similarly, ensure your `.env` (or environment variables) is configured for the SQLAlchemy database, as some endpoints (like token storage) may query for user existence.

## Running the Example

Below is an example command to start this local notification server (based on the `app.py` content):

```bash
python examples/local_notification_server/app.py
```

Alternatively, if you have a `main.py` in your root project that imports and runs this module, adapt accordingly. By default, it will run on `127.0.0.1:8003`.

You can then test it using tools like [HTTPie](https://httpie.io/) or cURL.

## Using the API

The main endpoints exposed by `routers.py` are:

### 1. Store FCM Token
```
POST /store_token
```
**Body** (`TokenRequest`):
```json
{
  "user_id": 1,
  "device_token": "AAAA-VVVV-1234-XYZ",
  "device_lang": "en-GB"
}
```
- Stores the device token in Redis under `fcm_tokens:{user_id}`, with the field as the token and value as the language code.

### 2. Delete FCM Token
```
DELETE /delete_token
```
**Body** (`TokenRequest`):
```json
{
  "user_id": 1,
  "device_token": "AAAA-VVVV-1234-XYZ"
}
```
- Deletes the specified token from the user’s Redis hash.

### 3. Send FCM Notification
```
POST /send_fcm_notification
```
**Body** (`SiteNotifyRequest`):
```json
{
  "site": "siteA",
  "stream_name": "Camera-01",
  "body": {
    "warning_no_hardhat": {"count": 2},
    "warning_close_to_machinery": {"count": 1}
  },
  "image_path": "https://example.com/image.jpg",
  "violation_id": 123
}
```
- Retrieves the site in the database, fetches associated users, and gathers all their device tokens from Redis.
- Groups tokens by language and uses the `Translator.translate_from_dict()` to translate the warnings.
- Sends out notifications with the relevant text and optional image.
- Returns a JSON response indicating success/failure.

## Additional Notes

1. **Device Tokens**
   - For Android, you can retrieve the FCM token via the Firebase SDK in your app.
   - For iOS, remember to enable push notifications and retrieve the APNs token, which the Firebase SDK will exchange for an FCM token.

2. **Security**
   - In production, ensure your endpoints (e.g., `/store_token`) are protected. We rely on the parent JWT-based auth system (via `jwt_access`) for the `/send_fcm_notification` endpoint.
   - Consider adding rate limiting and/or user-based checks for storing tokens.

3. **Multi-tenant or Per-Site**
   - This example groups users by site. You could adapt it to other grouping logic if desired (e.g., roles or certain user IDs).

4. **Translations**
   - The `lang_config.py` simply uses a dictionary approach. For large-scale i18n, consider a more sophisticated library or a translation management system.

5. **Logging & Error Handling**
   - `fcm_service.py` logs unsuccessful sends using `logging.error`. In a production system, you should have robust error-handling and retry logic.

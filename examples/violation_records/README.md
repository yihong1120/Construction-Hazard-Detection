🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Violation Records Backend

FastAPI service for storing, querying, and serving construction-hazard
violation records. `src/violation_sender.py` uploads violation images and
metadata here when the stream processor confirms a warning event.

## Responsibilities

- Store violation metadata in PostgreSQL.
- Save uploaded violation images under a controlled `static/` directory.
- Enforce JWT and site-based access control.
- Provide filtered, paginated violation search.
- Serve violation images by safe relative path.
- Support multilingual keyword search through synonym expansion.

## Files

- `app.py`: FastAPI application.
- `routers.py`: record list, detail, upload, site, and image routes.
- `schemas.py`: Pydantic request and response models.
- `violation_manager.py`: database and image-storage logic.
- `path_utils.py`: safe static-path validation.
- `search_utils.py`: dictionary-based multilingual search; no NLP model download.
- `violation_analytics.py`: aggregate SQL and chart response assembly.
- `settings.py`: static-directory settings.

## Run

```bash
uvicorn examples.violation_records.app:app \
  --host 127.0.0.1 \
  --port 8002 \
  --workers 2
```

The module `main()` still defaults to port `8081` for direct script execution,
but the project runtime uses port `8002`.

## Required Settings

```dotenv
VIOLATION_RECORD_API_URL=http://127.0.0.1:8002
DATABASE_URL=postgresql+asyncpg://username:password@127.0.0.1/construction_hazard_detection
OIDC_ENABLED=true
OIDC_ISSUER_URL=https://sso.example.com/realms/visionnaire
OIDC_JWKS_URL=https://sso.example.com/realms/visionnaire/protocol/openid-connect/certs
OIDC_AUDIENCE=visionnaire-api
OIDC_ACCOUNT_URL=https://sso.example.com/realms/visionnaire/account
```

## Endpoints

- `GET /my_sites`: sites visible to the authenticated user.
- `GET /violations/filter-options?site_id=...`: authorized cameras and fixed
  violation types for one site. Omitting `site_id` returns cameras across all
  authorized sites. `stream_id` is the numeric camera config ID, not
  a camera name.
- `GET /violations`: filtered and paginated violation list with optional
  `site_id`, `stream_id`, `violation_type`, and time-range filters. Admin
  reviewers can add `flagged=true` and optional
  `review_status=pending|resolved|dismissed`.
- `GET /violations/analytics`: analytics using the same authorized `site_id`,
  `stream_id`, `violation_type`, `start`, `end`, and `bucket` filters for all
  aggregates. Available only to `admin` and `super_admin` users.
- `GET /violations/{violation_id}`: single violation detail.
- `POST /violations/{violation_id}/feedback`: structured false-positive,
  missed-detection, wrong-class, or bad-bbox feedback for review.
- `PATCH /violations/{violation_id}/review`: admin/super-admin review update
  for a flagged record.
- `GET /get_violation_image?image_path=...`: image response from `static/`.
- `POST /upload`: image and metadata upload from the stream processor.

## Retention

For local-disk deployments without archive storage, keep both image files and
database records for the same retention window. The current recommendation is
18 months unless legal or contract rules require a different period.

When deleting old data, remove database rows and matching files together. Do
not keep orphaned images in `static/`, and do not keep records whose image paths
no longer exist.

## Storage Notes

- The `static/` directory is operational evidence storage, not a live stream
  buffer.
- HLS/WebRTC live segments belong to MediaMTX and should have a much shorter
  retention window.
- If disk use approaches the host limit, reduce retention or export records
  before deleting them.

Evidence reception counts bytes while streaming and verifies the image header,
dimensions and decodability in a bounded native pool before publication.
`VIOLATION_IMAGE_MAX_UPLOAD_BYTES` defaults to 20 MiB and
`VIOLATION_IMAGE_MAX_PIXELS` to 67,108,864 pixels; size violations return 413,
invalid image content returns 400 and invalid metadata returns 422. The pool has
two workers and eight admitted jobs; exhaustion returns 503 `IMAGE_WORK_BUSY`.
Warnings, detections and polygons are validated together and passed internally
as one metadata object.

PNG and JPEG are accepted, matching the protected original-media endpoint.
Original image bytes are retained. New filenames use the verified format's
extension; the recovery journal is updated before publishing a changed path.
Existing evidence paths remain readable and no SQL migration is required.
Historical detection IDs retain their original row positions when invalid rows
are skipped, so existing feedback continues to address the same detection.

Thumbnails use a temporary file and atomic replacement, share concurrent work
for the same source, and use the full source filename as the cache key (for
example `image.png.jpg`). Old cache files are unused and can be removed through
normal cache maintenance. Evidence decoding and thumbnail generation share the
bounded image pool, which is drained during application shutdown.

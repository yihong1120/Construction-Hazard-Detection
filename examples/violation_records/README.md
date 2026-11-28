🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

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

PostgreSQL upgrades do not install new application indexes. On existing
databases, administrators apply their locally maintained index migrations.
Migration scripts, measurements and tuning records are maintained separately
from this repository. Reserve `init.postgres.sql` for initialisation, as it
recreates tables.

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
- `GET /violations/filter-options?site_id=...`: authorised cameras and fixed
  violation types for one site. Omitting `site_id` returns cameras across all
  authorised sites. `stream_id` is the numeric camera config ID, not
  a camera name.
- `GET /violations`: filtered and paginated violation list with optional
  `site_id`, `stream_id`, `violation_type`, and time-range filters. Admin
  reviewers can add `flagged=true` and optional
  `review_status=pending|resolved|dismissed`.
- `GET /violations/analytics`: analytics using the same authorised `site_id`,
  `stream_id`, `violation_type`, `start`, `end`, and `bucket` filters for all
  aggregates, including trends, rankings and time-of-day statistics. Other
  roles receive `403 violation_analytics_forbidden`; `admin` and `super_admin`
  have access.
- `GET /violations/{violation_id}`: single violation detail.
- `POST /violations/{violation_id}/feedback`: structured false-positive,
  missed-detection, wrong-class or incorrect bounding-box feedback. New
  feedback enters the pending review state.
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

## Feedback and review options

The service builds these options for the authenticated user, their site
permissions and the selected record:

- `GET /violations/{violation_id}/feedback-options?locale=zh-TW`
- `GET /violations/{violation_id}/review-options?locale=zh-TW`

Native clients send `Authorization: Bearer <access_token>`. An example Nginx base
URL is `https://app.example.org/hazard/api/violations/`. For this
deployment the complete feedback options URL is
`https://app.example.org/hazard/api/violations/violations/{violation_id}/feedback-options?locale=zh-TW`.
The first `violations` is the proxy prefix. The second is the service's
resource path.

Responses contain `schema_version: 1`, `revision`, `options: [{id, label}]`,
`default_id: null` and `policy: {}`, and set
`Cache-Control: private, no-store`. Feedback classes come from the record's
original snapshot. An absent snapshot yields an empty option list. Review
options depend on the user's role and the current review status. The feedback
and review endpoints check permissions again when the user submits a decision.

The management scopes `violation_feedback_labels` and
`violation_review_actions` have been retired and return 404. Clients call
the violation endpoints directly. Restart the violation service after deployment
to load changed routes.

### Reporting a missed detection

Submit a manual annotation to `POST /violations/{id}/feedback`:

```json
{
  "type": "false_negative",
  "corrected_label": "person",
  "corrected_bbox": [10, 20, 100, 200],
  "note": "A person was missed here"
}
```

Use a stable class code from the feedback options for `corrected_label`, and
supply `corrected_bbox` as `[x1, y1, x2, y2]`. Manual feedback neither
requires nor returns a model ID or version. A `model_version` sent by an older
client is ignored. `original_label` preserves the original label text supplied
by the client.

The interface enables the feedback button according to whether options have
loaded and classes are available. Option lookup and submission both check
record access. Submission also validates the class and coordinates. Existing
inference version checks, record metadata and version fields on historical
feedback are preserved. The version field on new feedback is empty. This
requires no migration or backfill.

### Class names and colours

The classes from `/models`, the `class_metadata` in record details and lists,
and the feedback options share class codes and `#RRGGBB` colours. On creation,
the record stores the catalogue classes in `model_classes`. Its
`class_metadata` exposes their names, numeric IDs, codes and colours. The
backend normalises valid colours to upper case and derives a stable colour
from the class code if a colour is missing or invalid. Names and feedback
validation remain independent of colour.

Changing a model palette affects future snapshots. Historical records retain
their original colours and classes, including unknown classes. A feedback
POST needs no colour or model identifier. The service locks the record and
checks the class code against its snapshot. Clients can overlay the original
image, while live rendering uses class metadata supplied by the model that
ran inference. This contract uses existing routes and columns.
`/client-options` remains retired.

### Backend-defined box colours

Record details and lists include `overlay_style` version 1:

```json
{
  "schema_version": 1,
  "class_colors": {"person": "#FF9800", "vehicle": "#FFEB3B"},
  "unknown_class_color": "#9E9E9E",
  "cone_polygon_color": "#FF4081",
  "pole_polygon_color": "#448AFF",
  "warning_color": "#F44336"
}
```

The full response includes the complete backend palette in `class_colors`.
Valid snapshot colours override it for the same code. In the detail response,
`overlay_objects[].color` is the resolved colour and is independent of the
flagged state. If drawing `detection_items` directly, look up the numeric ID,
code and colour in `class_metadata` first, then consult `class_colors`. Use
`unknown_class_color` for an ID of unknown meaning, and the corresponding
polygon colour for each region boundary.

Flutter uses the backend palette. If a required colour is missing or invalid,
keep the original image visible, pause the affected overlay and offer to
retry loading the palette. Clients that use hard-coded or hashed fallback
colours need a client-side update. Feedback options and submissions use the
same resolved class table and check access to the record. Colours already
rendered into a historical JPEG are part of the image.

### Clustering and historical regions (overlay schema 1)

The stream's `DangerDetector` uses HDBSCAN. The resulting regions drive both
hazard detection and live rendering. Warnings and cone and pole polygons
from one `detect_danger` call are saved with the evidence image. The stream
feature settings control region detection independently of the client's
overlay switch. Upload polygons in original-image pixels as a JSON string
of the form `[[[x,y],...],...]`. The backend checks vertex counts, finite
coordinates and non-zero area. `null` means no polygons were supplied.
`[]` means an empty result was saved and does not establish whether clustering
ran.

`GET /violations/{id}` includes these fields in the existing response envelope:

```json
{
  "overlay_schema_version": 1,
  "image_size": {"width": 640, "height": 360},
  "overlay_coordinates": {
    "space": "normalized",
    "bbox_format": "xywh",
    "polygon_format": "xy",
    "origin": "top_left",
    "reference": "original_image"
  },
  "overlay_regions": [{
    "id": "cone_0",
    "kind": "cone",
    "display_name": "Traffic cone exclusion zone",
    "points": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.2]],
    "color": "#FF4081",
    "fill_opacity": 0.4,
    "closed": true
  }],
  "region_status": {"cone": "available", "pole": "empty"}
}
```

The points are illustrative. When reading a record, the service divides the
saved coordinates by the original image width and height. It never reruns
clustering or consults the camera's current configuration to reconstruct
historical regions. A region ID identifies a boundary within that record.
Coordinates can extend beyond the 0–1 range, for example when a buffer
extends beyond the image. Clip the polygon to the image rectangle when
drawing. Clamping individual vertices can change its shape.

`region_status` describes cone and pole regions independently:

- `available` means a saved boundary can be drawn.
- `empty` means the saved array was empty. Detection may have been disabled,
  or the objects may not have formed a valid cluster.
- `not_recorded` means no result was saved. Clusters may have existed.
- `invalid` means a saved boundary is malformed. Other evidence is still available.
- `image_unavailable` means the original dimensions are unavailable for converting saved coordinates.

Each entry in `overlay_objects` includes `code` and `display_name`. Use the
snapshot's `display_name` and `color`. The existing `label` remains available.
Unresolved numeric IDs use `class-N` and grey. Class resolution and region
geometry are independent.

Flutter fetches the detail endpoint when the user opens a list item. Draw
region fills and boundaries first, then object boxes and labels. For the
displayed `imageRect` after `BoxFit`, compute
`px = imageRect.left + x * imageRect.width` and
`py = imageRect.top + y * imageRect.height`. Scale the box width and height by
the displayed width and height respectively. These values are already
normalised dimensions. If the original image size is unavailable,
`image_size` is null and unconvertible pixel regions are omitted.

Older clients can still read the raw `cone_polygons` and `pole_polygons`
strings. New clients draw the resolved `overlay_regions` once. The existing
region columns suffice without a migration or historical class backfill.

### Inspecting the JSON received by a client

Set `VIOLATION_DEBUG_RESPONSE_IDS=*` to log all records or use a list such as
`155157,155158` to select records, then restart the service. For matching
detail, feedback-options and review-options GET requests, Uvicorn logs the
path, HTTP status and serialised JSON body under `[violation-response]`.
It omits request headers, cookies, tokens and image bytes. Each entry is
limited to 2 MiB and marked `truncated=true` if clipped. Responses are
unaffected. Clear the setting and restart to finish the investigation.

### Record class snapshots

Details, lists, feedback options and submission checks all use the record's
saved `model_classes`. If absent, the class options are empty. New uploads
must identify the model that performed inference through `model_id` and
`model_version`. The live renderer uses producer-supplied metadata and keeps
unknown IDs unresolved.

### Object visibility in normal and feedback views

Each `overlay_objects` entry includes four display flags. This partial example
shows a safety cone:

```json
{
  "object_id": "det_0",
  "code": "safety_cone",
  "display_name": "Traffic cone",
  "color": "#FF5722",
  "show_box": false,
  "show_label": false,
  "feedback_show_box": true,
  "feedback_show_label": true
}
```

The full object retains its bounding box and confidence. Use `show_box` and
`show_label` in detail and full-screen views. In feedback and debugging views,
use `feedback_show_box` and `feedback_show_label`. All four flags are currently
true for other objects, including unknown classes. Treat missing flags as
true when reading an older response.

Flutter retains the full set of objects even when boxes are hidden and
applies the flags supplied by the service. Flagged and manually annotated
cones follow the same policy. Feedback permissions remain subject to the
option endpoints and the checks on POST.

Live overlays apply this policy to the canonical `safety_cone` code. The
underlying detections still contribute to clustering, warnings, stored
evidence and feedback. Region visibility is independent of the object flags.
An empty cluster yields no region even when individual cone boxes are hidden.
The policy retains the existing HDBSCAN parameters and historical results.

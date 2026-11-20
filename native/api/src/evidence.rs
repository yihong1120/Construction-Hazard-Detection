//! An isolated evidence store for the standalone camera pipeline.
use std::{collections::HashMap, io::Cursor};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path, Query, State, multipart::MultipartRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{DateTime, Local, NaiveDateTime, Timelike, Utc};
use image::{ImageDecoder, ImageFormat, ImageReader, codecs::jpeg::JpegDecoder};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sqlx::{PgPool, types::Json as PgJson};
use tokio::sync::{Semaphore, SemaphorePermit};
use uuid::Uuid;

use crate::{
    ApiError, AppState,
    auth::{Actor, Grant},
};

const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
const MAX_METADATA_BYTES: usize = 1024 * 1024;
pub(crate) const UPLOAD_BODY_LIMIT: usize = MAX_IMAGE_BYTES + MAX_METADATA_BYTES + 64 * 1024;
const MAX_IMAGE_PIXELS: u64 = 16_000_000;
const PRODUCTION_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const PRODUCTION_IMAGE_PIXELS: u64 = 64 * 1024 * 1024;
pub(crate) const PRODUCTION_UPLOAD_BODY_LIMIT: usize =
    PRODUCTION_IMAGE_BYTES + MAX_METADATA_BYTES + 64 * 1024;
static UPLOAD_SLOTS: Semaphore = Semaphore::const_new(4);

pub fn routes(include_upload: bool) -> Router<AppState> {
    let router = Router::new();
    let router = if include_upload {
        router.route(
            "/upload",
            post(upload).layer(DefaultBodyLimit::max(UPLOAD_BODY_LIMIT)),
        )
    } else {
        router
    };
    router
        .route("/evidence", get(list))
        .route("/evidence/{id}", get(detail))
        .route("/evidence/{id}/image", get(image))
        .layer(middleware::map_response(
            |mut response: Response| async move {
                response.headers_mut().insert(
                    header::CACHE_CONTROL,
                    HeaderValue::from_static("private, no-store"),
                );
                response
            },
        ))
}

pub(crate) type Polygon = Vec<[f64; 2]>;

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub(crate) struct Evidence {
    pub(crate) event_id: Option<Uuid>,
    pub(crate) site_id: i64,
    pub(crate) site: String,
    pub(crate) stream_name: String,
    pub(crate) detection_time: Option<DateTime<Utc>>,
    #[serde(skip)]
    pub(crate) prior_local_detection_time: Option<DateTime<Utc>>,
    pub(crate) model_id: String,
    pub(crate) model_version: String,
    pub(crate) warnings: Option<HashMap<String, Value>>,
    pub(crate) detections: Option<Vec<Vec<f64>>>,
    pub(crate) cone_polygon: Option<Value>,
    pub(crate) pole_polygon: Option<Value>,
}

#[derive(Serialize)]
pub(crate) struct Receipt {
    pub(crate) message: &'static str,
    pub(crate) violation_id: i64,
}

pub(crate) struct AcceptedUpload {
    pub(crate) actor: Actor,
    pub(crate) evidence: Evidence,
    pub(crate) image: Bytes,
    pub(crate) bounds: (i32, i32),
    pub(crate) format: ImageFormat,
    _permit: SemaphorePermit<'static>,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum UploadMode {
    Preview,
    Production,
}

#[derive(Debug)]
pub(crate) enum UploadError {
    Api(ApiError),
    Image(StatusCode, &'static str),
}

impl From<ApiError> for UploadError {
    fn from(error: ApiError) -> Self {
        Self::Api(error)
    }
}

impl IntoResponse for UploadError {
    fn into_response(self) -> Response {
        match self {
            Self::Api(error) => error.into_response(),
            Self::Image(status, code) => (
                status,
                [(header::CACHE_CONTROL, "no-store")],
                Json(serde_json::json!({"detail": {"code": code}})),
            )
                .into_response(),
        }
    }
}

#[derive(Serialize)]
struct EvidenceView {
    id: i64,
    #[serde(flatten)]
    evidence: Evidence,
    width: i32,
    height: i32,
    received_at: DateTime<Utc>,
    image_url: String,
}

#[derive(Deserialize, Serialize)]
struct EvidenceSummary {
    id: i64,
    event_id: Uuid,
    site_id: i64,
    site: String,
    stream_name: String,
    detection_time: DateTime<Utc>,
    model_id: String,
    model_version: String,
    width: i32,
    height: i32,
    received_at: DateTime<Utc>,
}

#[derive(Serialize)]
struct EvidenceListItem {
    #[serde(flatten)]
    summary: EvidenceSummary,
    image_url: String,
}

#[derive(Serialize)]
struct EvidencePage {
    items: Vec<EvidenceListItem>,
    next_before: Option<i64>,
}

#[derive(Deserialize)]
struct Pagination {
    limit: Option<i64>,
    before: Option<i64>,
}

type EvidenceRow = (i64, PgJson<Evidence>, i32, i32, DateTime<Utc>);

impl From<EvidenceRow> for EvidenceView {
    fn from((id, evidence, width, height, received_at): EvidenceRow) -> Self {
        Self {
            id,
            evidence: evidence.0,
            width,
            height,
            received_at,
            image_url: format!("/evidence/{id}/image"),
        }
    }
}

async fn upload(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Receipt>, UploadError> {
    let upload = validate_upload(&state, &headers, multipart, UploadMode::Preview).await?;
    let id = store(&state.db, &upload.evidence, &upload.image, upload.bounds).await?;
    Ok(Json(Receipt {
        message: "Violation uploaded successfully.",
        violation_id: id,
    }))
}

pub(crate) async fn validate_upload(
    state: &AppState,
    headers: &HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
    mode: UploadMode,
) -> Result<AcceptedUpload, UploadError> {
    let actor = state.auth.authorize(headers, Grant::EvidenceWrite).await?;
    let mut keys = headers.get_all("idempotency-key").iter();
    let key = keys
        .next()
        .map(|value| {
            value
                .to_str()
                .ok()
                .and_then(|value| Uuid::parse_str(value).ok())
                .ok_or_else(|| {
                    ApiError::new(StatusCode::BAD_REQUEST, "Idempotency-Key must be a UUID")
                })
        })
        .transpose()?;
    if keys.next().is_some() || (mode == UploadMode::Preview && key.is_none()) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Exactly one Idempotency-Key is required",
        )
        .into());
    }

    // Bounds encompass multipart collection, JPEG decode and the write.
    let permit = UPLOAD_SLOTS
        .try_acquire()
        .map_err(|_| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "Evidence uploads busy"))?;
    let multipart =
        multipart.map_err(|error| ApiError::new(error.status(), "Invalid multipart upload"))?;
    let (fields, image) = parse_multipart(multipart, mode).await?;
    let mut evidence = parse_metadata(&fields, mode, |site| {
        actor
            .sites()
            .into_iter()
            .find_map(|(id, name)| (site == name).then_some(id))
    })?;
    if let (Some(key), Some(event_id)) = (key, evidence.event_id)
        && key != event_id
    {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Idempotency-Key must match event_id",
        )
        .into());
    }
    evidence.event_id = evidence.event_id.or(key);
    if !actor.allows_site(evidence.site_id, &evidence.site) {
        return Err(ApiError::new(StatusCode::FORBIDDEN, "No access to this site").into());
    }

    let (permit, bounds, format, image) = tokio::task::spawn_blocking(move || {
        // Keep the slot occupied even if the request is cancelled while decoding.
        let (bounds, format) = match mode {
            UploadMode::Preview => (jpeg_bounds(&image)?, ImageFormat::Jpeg),
            UploadMode::Production => production_image(&image)?,
        };
        Ok::<_, UploadError>((permit, bounds, format, image))
    })
    .await
    .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "Image decoding failed"))??;
    Ok(AcceptedUpload {
        actor,
        evidence,
        image,
        bounds,
        format,
        _permit: permit,
    })
}

async fn parse_multipart(
    mut multipart: Multipart,
    mode: UploadMode,
) -> Result<(HashMap<String, Bytes>, Bytes), UploadError> {
    let mut values = HashMap::new();
    let mut image = None;
    let mut metadata_size = 0;
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field.name().ok_or_else(invalid_metadata)?.to_owned();
        if name == "image" {
            if image.is_some() {
                return Err(invalid_metadata().into());
            }
            if mode == UploadMode::Preview && field.content_type() != Some("image/jpeg") {
                return Err(ApiError::new(
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "Evidence image must be JPEG",
                )
                .into());
            }
            let data = if mode == UploadMode::Preview {
                collect_field(field, MAX_IMAGE_BYTES).await?
            } else {
                collect_field(field, PRODUCTION_IMAGE_BYTES)
                    .await
                    .map_err(|error| {
                        if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
                            UploadError::Image(
                                StatusCode::PAYLOAD_TOO_LARGE,
                                "IMAGE_UPLOAD_LIMIT_EXCEEDED",
                            )
                        } else {
                            error.into()
                        }
                    })?
            };
            image = Some(data);
            continue;
        }
        const FIELDS: &[&str] = &[
            "event_id",
            "site_id",
            "site",
            "stream_name",
            "detection_time",
            "model_id",
            "model_version",
            "warnings_json",
            "detections_json",
            "cone_polygon_json",
            "pole_polygon_json",
        ];
        if !FIELDS.contains(&name.as_str()) || values.contains_key(&name) {
            return Err(invalid_metadata().into());
        }
        let data = collect_field(field, MAX_METADATA_BYTES - metadata_size).await?;
        metadata_size += data.len();
        values.insert(name, data);
    }
    let image =
        image.ok_or_else(|| ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "Image required"))?;
    Ok((values, image))
}

async fn collect_field(
    mut field: axum::extract::multipart::Field<'_>,
    limit: usize,
) -> Result<Bytes, ApiError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
        if chunk.len() > limit - bytes.len() {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "Evidence field too large",
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(bytes))
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> ApiError {
    ApiError::new(error.status(), "Invalid multipart upload")
}

fn invalid_metadata() -> ApiError {
    ApiError::new(
        StatusCode::UNPROCESSABLE_ENTITY,
        "Invalid evidence metadata",
    )
}

fn field_text<'a>(fields: &'a HashMap<String, Bytes>, key: &str) -> Result<&'a str, ApiError> {
    let value = fields.get(key).ok_or_else(invalid_metadata)?;
    std::str::from_utf8(value).map_err(|_| invalid_metadata())
}

fn bounded_text(
    fields: &HashMap<String, Bytes>,
    key: &str,
    limit: usize,
) -> Result<String, ApiError> {
    let value = field_text(fields, key)?;
    if value.trim().is_empty() || value.chars().count() > limit || value.contains('\0') {
        return Err(invalid_metadata());
    }
    Ok(value.to_owned())
}

fn json_field<T: DeserializeOwned>(
    fields: &HashMap<String, Bytes>,
    key: &str,
    mode: UploadMode,
) -> Result<Option<T>, ApiError> {
    optional_text(fields, key, mode)?
        .map(|value| serde_json::from_str(value).map_err(|_| invalid_metadata()))
        .transpose()
}

fn optional_text<'a>(
    fields: &'a HashMap<String, Bytes>,
    key: &str,
    mode: UploadMode,
) -> Result<Option<&'a str>, ApiError> {
    if mode == UploadMode::Production && fields.get(key).is_none_or(Bytes::is_empty) {
        return Ok(None);
    }
    field_text(fields, key).map(Some)
}

#[derive(Clone, Copy)]
struct UploadTime {
    instant: DateTime<Utc>,
    prior_local: Option<DateTime<Utc>>,
}

fn upload_time(time: &str, mode: UploadMode) -> Result<UploadTime, ApiError> {
    if mode == UploadMode::Preview {
        return DateTime::parse_from_rfc3339(time)
            .map(|timestamp| UploadTime {
                instant: timestamp.to_utc(),
                prior_local: None,
            })
            .map_err(|_| invalid_metadata());
    }
    let time = time.replacen(' ', "T", 1);
    // The deployed Python datetime contract and PostgreSQL both use microseconds.
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(&time) {
        let instant = timestamp
            .to_utc()
            .with_nanosecond(timestamp.timestamp_subsec_micros() * 1000)
            .ok_or_else(invalid_metadata)?;
        return Ok(UploadTime {
            instant,
            prior_local: None,
        });
    }
    let naive = time
        .parse::<NaiveDateTime>()
        .or_else(|_| NaiveDateTime::parse_from_str(&time, "%Y-%m-%dT%H:%M"))
        .map_err(|_| invalid_metadata())?;
    let naive = naive
        .with_nanosecond(naive.and_utc().timestamp_subsec_micros() * 1000)
        .ok_or_else(invalid_metadata)?;
    let instant = naive.and_utc();
    // Verify retries against events written with the former server-local clock.
    let prior_local = naive
        .and_local_timezone(Local)
        .earliest()
        .map(|time| time.to_utc())
        .filter(|time| *time != instant);
    Ok(UploadTime {
        instant,
        prior_local,
    })
}

fn parse_metadata(
    fields: &HashMap<String, Bytes>,
    mode: UploadMode,
    site_for_name: impl FnOnce(&str) -> Option<i64>,
) -> Result<Evidence, ApiError> {
    let site = bounded_text(fields, "site", 80)?;
    let site_id = match optional_text(fields, "site_id", mode)? {
        Some(value) => value.parse::<i64>().map_err(|_| invalid_metadata())?,
        None => site_for_name(&site)
            .ok_or_else(|| ApiError::new(StatusCode::FORBIDDEN, "No access to this site"))?,
    };
    if site_id <= 0 {
        return Err(invalid_metadata());
    }
    let warnings: Option<HashMap<String, Value>> = json_field(fields, "warnings_json", mode)?;
    if warnings.iter().flat_map(HashMap::values).any(|warning| {
        let count = &warning["count"];
        count.as_u64().is_none() && (mode == UploadMode::Preview || count.as_i64().is_none())
    }) {
        return Err(invalid_metadata());
    }
    let detections: Option<Vec<Vec<f64>>> = json_field(fields, "detections_json", mode)?;
    if detections.iter().flatten().any(|row| {
        let [x1, y1, x2, y2, confidence, class_id, track_id, ..] = row.as_slice() else {
            return true;
        };
        (mode == UploadMode::Preview && row.len() != 8)
            || x2 <= x1
            || y2 <= y1
            || !(0.0..=1.0).contains(confidence)
            || *class_id < 0.0
            || class_id.fract() != 0.0
            || *track_id < -1.0
            || track_id.fract() != 0.0
    }) {
        return Err(invalid_metadata());
    }
    let detection_time = optional_text(fields, "detection_time", mode)?
        .map(|value| upload_time(value, mode))
        .transpose()?;
    Ok(Evidence {
        event_id: optional_text(fields, "event_id", mode)?
            .map(|value| Uuid::parse_str(value).map_err(|_| invalid_metadata()))
            .transpose()?,
        site_id,
        site,
        stream_name: bounded_text(fields, "stream_name", 80)?,
        detection_time: detection_time.map(|time| time.instant),
        prior_local_detection_time: detection_time.and_then(|time| time.prior_local),
        model_id: bounded_text(fields, "model_id", 120)?,
        model_version: bounded_text(fields, "model_version", 120)?,
        warnings,
        detections,
        cone_polygon: polygons_field(fields, "cone_polygon_json", mode)?,
        pole_polygon: polygons_field(fields, "pole_polygon_json", mode)?,
    })
}

fn polygons_field(
    fields: &HashMap<String, Bytes>,
    key: &str,
    mode: UploadMode,
) -> Result<Option<Value>, ApiError> {
    let Some(value): Option<Value> = json_field(fields, key, mode)? else {
        return Ok(None);
    };
    if mode == UploadMode::Production && value.is_null() {
        return Ok(Some(Value::Null));
    }
    let rings: Vec<Polygon> =
        serde_json::from_value(value.clone()).map_err(|_| invalid_metadata())?;
    if mode == UploadMode::Preview {
        return serde_json::to_value(rings)
            .map(Some)
            .map_err(|_| invalid_metadata());
    }
    if rings.iter().any(|points| {
        let area = points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
            .sum::<f64>();
        points.len() < 3 || !area.is_finite() || area.abs() < 1e-9
    }) {
        return Err(invalid_metadata());
    }
    Ok(Some(value))
}

fn production_image(bytes: &[u8]) -> Result<((i32, i32), ImageFormat), UploadError> {
    if bytes.is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "Failed to read image file").into());
    }
    let invalid = || UploadError::Image(StatusCode::BAD_REQUEST, "INVALID_IMAGE");
    let format = image::guess_format(bytes).map_err(|_| invalid())?;
    if !matches!(format, ImageFormat::Jpeg | ImageFormat::Png) {
        return Err(UploadError::Image(
            StatusCode::BAD_REQUEST,
            "UNSUPPORTED_IMAGE_FORMAT",
        ));
    }
    let decoder = ImageReader::with_format(Cursor::new(bytes), format)
        .into_decoder()
        .map_err(|_| invalid())?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > PRODUCTION_IMAGE_PIXELS {
        return Err(UploadError::Image(
            StatusCode::PAYLOAD_TOO_LARGE,
            "IMAGE_DIMENSIONS_EXCEEDED",
        ));
    }
    image::DynamicImage::from_decoder(decoder).map_err(|_| invalid())?;
    Ok(((width as i32, height as i32), format))
}

fn jpeg_bounds(bytes: &[u8]) -> Result<(i32, i32), ApiError> {
    let invalid = || {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid JPEG evidence image",
        )
    };
    if !bytes.starts_with(b"\xff\xd8") || !bytes.ends_with(b"\xff\xd9") {
        return Err(invalid());
    }
    let decoder = JpegDecoder::new(Cursor::new(bytes)).map_err(|_| invalid())?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Evidence image dimensions too large",
        ));
    }
    // Fully decode here to reject corrupt data before it is durably accepted.
    image::DynamicImage::from_decoder(decoder).map_err(|_| invalid())?;
    Ok((width as i32, height as i32))
}

async fn store(
    db: &PgPool,
    evidence: &Evidence,
    image: &[u8],
    (width, height): (i32, i32),
) -> Result<i64, ApiError> {
    let digest = Sha256::digest(image);
    let mut tx = db.begin().await.map_err(ApiError::database)?;
    let new_id: Option<i64> = sqlx::query_scalar(
        "INSERT INTO rust_api.evidence (event_id,site_id,site,metadata,image,image_sha256,width,height)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT (site_id,event_id) DO NOTHING RETURNING id",
    )
    .bind(evidence.event_id)
    .bind(evidence.site_id)
    .bind(&evidence.site)
    .bind(PgJson(evidence))
    .bind(image)
    .bind(&digest[..])
    .bind(width)
    .bind(height)
    .fetch_optional(&mut *tx)
    .await
    .map_err(ApiError::database)?;

    let id = if let Some(id) = new_id {
        id
    } else {
        // A separate statement sees a competing transaction's committed row.
        sqlx::query_scalar(
            "SELECT id FROM rust_api.evidence
             WHERE event_id=$1 AND site_id=$2 AND site=$3
             AND metadata=$4 AND image_sha256=$5",
        )
        .bind(evidence.event_id)
        .bind(evidence.site_id)
        .bind(&evidence.site)
        .bind(PgJson(evidence))
        .bind(&digest[..])
        .fetch_optional(&mut *tx)
        .await
        .map_err(ApiError::database)?
        .ok_or_else(|| ApiError::new(StatusCode::CONFLICT, "event_id_conflict"))?
    };
    tx.commit().await.map_err(ApiError::database)?;
    Ok(id)
}

fn allowed_sites(actor: &Actor) -> (Vec<i64>, Vec<String>) {
    actor.sites().into_iter().unzip()
}

async fn list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(page): Query<Pagination>,
) -> Result<Json<EvidencePage>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::EvidenceRead).await?;
    let limit = page.limit.unwrap_or(20);
    if !(1..=100).contains(&limit) || page.before.is_some_and(|id| id <= 0) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Invalid evidence pagination",
        ));
    }
    let (ids, names) = allowed_sites(&actor);
    let rows: Vec<PgJson<EvidenceSummary>> = sqlx::query_scalar(
        "SELECT jsonb_build_object(
             'id', id, 'event_id', event_id, 'site_id', site_id, 'site', site,
             'stream_name', metadata->'stream_name',
             'model_id', metadata->'model_id', 'model_version', metadata->'model_version',
             'detection_time', metadata->'detection_time',
             'width', width, 'height', height, 'received_at', received_at
         )
         FROM rust_api.evidence AS e
         WHERE EXISTS (
             SELECT 1 FROM unnest($1::bigint[], $2::text[]) AS allowed(id, name)
             WHERE allowed.id=e.site_id AND allowed.name=e.site
         )
         AND ($3::bigint IS NULL OR id < $3)
         ORDER BY id DESC LIMIT $4",
    )
    .bind(ids)
    .bind(names)
    .bind(page.before)
    .bind(limit + 1)
    .fetch_all(&state.db)
    .await
    .map_err(ApiError::database)?;
    let mut entries: Vec<_> = rows
        .into_iter()
        .map(|PgJson(summary)| EvidenceListItem {
            image_url: format!("/evidence/{}/image", summary.id),
            summary,
        })
        .collect();
    let next_before = if entries.len() > limit as usize {
        entries.pop();
        entries.last().map(|entry| entry.summary.id)
    } else {
        None
    };
    Ok(Json(EvidencePage {
        items: entries,
        next_before,
    }))
}

fn missing_evidence() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "Evidence not found")
}

async fn detail(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
) -> Result<Json<EvidenceView>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::EvidenceRead).await?;
    let (ids, names) = allowed_sites(&actor);
    let row: EvidenceRow = sqlx::query_as(
        "SELECT id,metadata,width,height,received_at FROM rust_api.evidence AS e
         WHERE id=$1 AND EXISTS (
             SELECT 1 FROM unnest($2::bigint[], $3::text[]) AS allowed(id, name)
             WHERE allowed.id=e.site_id AND allowed.name=e.site
         )",
    )
    .bind(id)
    .bind(ids)
    .bind(names)
    .fetch_optional(&state.db)
    .await
    .map_err(ApiError::database)?
    .ok_or_else(missing_evidence)?;
    Ok(Json(row.into()))
}

async fn image(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
) -> Result<Response, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::EvidenceRead).await?;
    let (ids, names) = allowed_sites(&actor);
    let jpeg: Vec<u8> = sqlx::query_scalar(
        "SELECT image FROM rust_api.evidence AS e WHERE id=$1 AND EXISTS (
             SELECT 1 FROM unnest($2::bigint[], $3::text[]) AS allowed(id, name)
             WHERE allowed.id=e.site_id AND allowed.name=e.site
         )",
    )
    .bind(id)
    .bind(ids)
    .bind(names)
    .fetch_optional(&state.db)
    .await
    .map_err(ApiError::database)?
    .ok_or_else(missing_evidence)?;
    Ok((
        [
            (header::CONTENT_TYPE, "image/jpeg"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        jpeg,
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage, codecs::jpeg::JpegEncoder};

    fn fields() -> HashMap<String, Bytes> {
        [
            ("event_id", "123e4567-e89b-12d3-a456-426614174000"),
            ("site_id", "42"),
            ("site", "工地"),
            ("stream_name", "camera01"),
            ("detection_time", "2026-11-19T00:00:00+08:00"),
            ("model_id", "yolo26n"),
            ("model_version", "abc123"),
            ("warnings_json", r#"{"warning_no_hardhat":{"count":1}}"#),
            ("detections_json", "[[10,20,40,50,0.9,1,3,0]]"),
            ("cone_polygon_json", "[[[10,10],[20,10],[20,20]]]"),
            ("pole_polygon_json", "[]"),
        ]
        .into_iter()
        .map(|(field, value)| (field.to_owned(), Bytes::from_static(value.as_bytes())))
        .collect()
    }

    #[test]
    fn accepts_standalone_fields_and_normalizes_event_time() {
        let evidence =
            parse_metadata(&fields(), UploadMode::Preview, |_| None).expect("standalone metadata");
        assert_eq!(evidence.site, "工地");
        assert_eq!(evidence.site_id, 42);
        assert_eq!(
            evidence.detection_time.unwrap().to_rfc3339(),
            "2026-11-18T16:00:00+00:00"
        );
        assert_eq!(evidence.detections.unwrap()[0][6], 3.0);
        assert_eq!(
            evidence.cone_polygon.unwrap()[0][1],
            serde_json::json!([20.0, 10.0])
        );
    }

    #[test]
    fn rejects_invalid_fields_and_detection_rows() {
        for (key, value) in [
            ("site_id", "0"),
            ("site", ""),
            ("detection_time", "2026-11-19T00:00:00"),
            ("warnings_json", r#"{"warning_no_hardhat":{"count":"1"}}"#),
            ("warnings_json", "[]"),
            ("detections_json", "[[10,20,10,30,0.9,1,3,0]]"),
            ("detections_json", "[[10,20,30,40,1.1,1,3,0]]"),
            ("detections_json", "[[10,20,30,40,0.9,0.1,3,0]]"),
            ("cone_polygon_json", "[[[10,20,30]]]"),
        ] {
            let mut input = fields();
            input.insert(key.to_owned(), Bytes::from_static(value.as_bytes()));
            assert!(
                parse_metadata(&input, UploadMode::Preview, |_| None).is_err(),
                "accepted invalid {key}: {value}"
            );
        }
    }

    #[test]
    fn production_metadata_accepts_the_legacy_multipart_contract() {
        let mut input = fields();
        for field in [
            "event_id",
            "site_id",
            "detection_time",
            "warnings_json",
            "detections_json",
            "cone_polygon_json",
            "pole_polygon_json",
        ] {
            input.remove(field);
        }
        let resolve = |name: &str| (name == "工地").then_some(42);
        let evidence = parse_metadata(&input, UploadMode::Production, resolve).unwrap();
        assert_eq!(evidence.site_id, 42);
        assert!(evidence.event_id.is_none());
        assert!(evidence.detection_time.is_none());
        assert!(evidence.detections.is_none());
        assert!(evidence.warnings.is_none());
        assert!(evidence.cone_polygon.is_none());
        assert!(evidence.pole_polygon.is_none());
        input.insert(
            "detections_json".into(),
            Bytes::from_static(b"[[1,2,3,4,0.9,1,-1]]"),
        );
        input.insert(
            "detection_time".into(),
            Bytes::from_static(b"2026-11-19T00:00:00"),
        );
        let evidence = parse_metadata(&input, UploadMode::Production, resolve).unwrap();
        assert_eq!(evidence.detections.unwrap()[0].len(), 7);
        assert_eq!(
            evidence.detection_time.unwrap(),
            DateTime::parse_from_rfc3339("2026-11-19T00:00:00Z")
                .unwrap()
                .to_utc()
        );
    }

    #[test]
    fn legacy_naive_and_explicit_evidence_times_share_one_utc_instant() {
        let expected = DateTime::parse_from_rfc3339("2026-11-19T09:30:22.554Z")
            .unwrap()
            .to_utc();
        let naive = upload_time("2026-11-19T09:30:22.554", UploadMode::Production).unwrap();
        let explicit =
            upload_time("2026-11-19T17:30:22.554+08:00", UploadMode::Production).unwrap();
        assert_eq!(naive.instant, expected);
        assert_eq!(explicit.instant, expected);
        assert_eq!(
            naive.prior_local,
            expected
                .naive_utc()
                .and_local_timezone(Local)
                .earliest()
                .map(|time| time.to_utc())
                .filter(|time| *time != expected)
        );
        assert!(explicit.prior_local.is_none());
    }

    #[test]
    fn decodes_jpeg_and_rejects_truncated_or_excessive_images() {
        let mut jpeg = Vec::new();
        let pixels = RgbImage::from_pixel(10, 12, Rgb([10, 20, 30]));
        JpegEncoder::new(&mut jpeg)
            .encode_image(&pixels)
            .expect("test JPEG");
        let size = jpeg_bounds(&jpeg).expect("valid JPEG");
        assert_eq!(size, (10, 12));
        assert!(jpeg_bounds(&jpeg[..jpeg.len() - 2]).is_err());
        assert!(jpeg_bounds(b"\xff\xd8not JPEG\xff\xd9").is_err());

        let sof = jpeg
            .windows(2)
            .position(|bytes| bytes == b"\xff\xc0")
            .expect("baseline JPEG SOF segment");
        jpeg[sof + 5..sof + 9].copy_from_slice(&[0x40, 0, 0x40, 0]);
        assert!(jpeg_bounds(&jpeg).is_err());
    }
}

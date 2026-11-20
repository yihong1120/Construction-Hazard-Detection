use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use construction_inference::core::Image;
use construction_inference::engine::NativeEngine;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use super::{
    Config, MAX_UPLOAD_BYTES, Model, management,
    overlay::Options,
    preparation::{Preparation, PreparedImage},
    websocket,
};
use crate::auth::{Actor, Grant};
use crate::{ApiError, AppState};

const MAX_ENGINE_BYTES: u64 = 512 * 1024 * 1024;
pub(super) const UPLOAD_TIMEOUT: Duration = Duration::from_secs(30);
const WORK_TIMEOUT: Duration = Duration::from_secs(30);
const CONFLICT_CLASSES: [(&str, &str); 2] =
    [("helmet", "no_helmet"), ("safety_vest", "no_safety_vest")];

static DETECTION_QUOTA: LazyLock<redis::Script> = LazyLock::new(|| {
    redis::Script::new(
        "local count = redis.call('INCR', KEYS[1])
         local ttl = redis.call('TTL', KEYS[1])
         if ttl < 0 then
             redis.call('EXPIRE', KEYS[1], ARGV[1])
             ttl = tonumber(ARGV[1])
         end
         return {count, ttl}",
    )
});

pub struct Inference {
    models: Arc<Vec<LoadedModel>>,
    default_model_id: Option<String>,
    catalog_tenant_id: Option<String>,
    jobs: Option<mpsc::Sender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
    uploads: Arc<Semaphore>,
    preparation: Preparation,
    outstanding: Arc<Semaphore>,
    overlays: Arc<Semaphore>,
    pub(super) sessions: Arc<Semaphore>,
    pub(super) max_upload_bytes: usize,
}

impl Drop for Inference {
    fn drop(&mut self) {
        drop(self.jobs.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(super) struct LoadedModel {
    pub(super) spec: Model,
    artifact: File,
    pub(super) source_artifact: Option<management::Artifact>,
}

struct Job {
    model: usize,
    image: Option<PreparedImage>,
    reply: oneshot::Sender<Result<Option<Detection>, Fault>>,
    // Keep accounting for running jobs even if their clients disconnect.
    _permit: OwnedSemaphorePermit,
}

pub(super) struct Detection {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) rows: Vec<DetectionRow>,
}

// Tuple serialization keeps integer coordinates and class IDs on the legacy wire.
pub(super) type DetectionRow = (i64, i64, i64, i64, f32, u32);

pub fn start(config: Config) -> Result<Arc<Inference>> {
    config.validate()?;
    let models = Arc::new(
        config
            .models
            .into_iter()
            .map(stage_model)
            .collect::<Result<Vec<_>>>()?,
    );
    let (jobs, receiver) = mpsc::channel(config.capacity);
    let owned = models.clone();
    let worker = std::thread::Builder::new()
        .name("rust-image-inference".into())
        .spawn(move || run_jobs(receiver, owned, config.device))?;
    Ok(Arc::new(Inference {
        models,
        default_model_id: config.default_model_id,
        catalog_tenant_id: config.catalog_tenant_id,
        jobs: Some(jobs),
        worker: Some(worker),
        uploads: Arc::new(Semaphore::new((config.capacity * 2).min(16))),
        preparation: Preparation::new(config.capacity, config.max_image_pixels),
        outstanding: Arc::new(Semaphore::new(config.capacity)),
        overlays: Arc::new(Semaphore::new(config.capacity)),
        sessions: Arc::new(Semaphore::new(config.capacity)),
        max_upload_bytes: config.max_upload_bytes,
    }))
}

fn stage_model(mut spec: Model) -> Result<LoadedModel> {
    let artifact = verified_copy(&spec.engine, &spec.sha256, MAX_ENGINE_BYTES)
        .with_context(|| format!("Verify engine for {} {}", spec.id, spec.version))?
        .0;
    let source_artifact = spec
        .source_artifact
        .as_ref()
        .map(management::Artifact::stage)
        .transpose()
        .with_context(|| format!("Verify published model for {} {}", spec.id, spec.version))?;
    for class in &mut spec.classes {
        class.color.make_ascii_uppercase();
    }
    Ok(LoadedModel {
        spec,
        artifact,
        source_artifact,
    })
}

pub(super) fn verified_copy(
    path: &std::path::Path,
    sha256: &str,
    max_bytes: u64,
) -> Result<(File, u64)> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "Configured model artifact must be a regular file"
    );
    let mut input = file.take(max_bytes + 1);
    // Readers reopen this anonymous verified copy via procfs. Changes to the
    // configured source cannot alter the approved artifact.
    let mut artifact = tempfile::tempfile()?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 256 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        artifact.write_all(&buffer[..count])?;
    }
    let size = max_bytes + 1 - input.limit();
    ensure!(
        (1..=max_bytes).contains(&size)
            && format!("{:x}", hash.finalize()).eq_ignore_ascii_case(sha256),
        "Artifact size or SHA-256 mismatch",
    );
    Ok((artifact, size))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/models", get(list_models))
        .route("/detect", post(detect))
        .route("/v2/detect", post(detect_v2))
        .route("/get_new_model", post(management::get_new_model))
        .route("/model_file_update", post(management::model_file_update))
        .route("/ws/detect", get(websocket::upgrade))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES + 128 * 1024))
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, Fault> {
    let inference = state.inference.as_ref().ok_or_else(Fault::unavailable)?;
    let actor = state
        .auth
        .authorize(&headers, Grant::ImageDetection)
        .await?;
    let (visible, default) = inference.visible_models(&state.db, &actor).await?;
    let models: Vec<_> = visible
        .into_iter()
        .map(|model| {
            json!({
                "id": model.id,
                "display_name": model.display_name,
                "version": model.version,
                "classes": model.classes,
            })
        })
        .collect();
    Ok((
        [(header::CACHE_CONTROL, "private, no-store")],
        Json(json!({"default_model_id": default, "models": models})),
    )
        .into_response())
}

async fn detect(State(state): State<AppState>, request: Request) -> Result<Response, Fault> {
    run_request(state, request, false).await
}

async fn detect_v2(State(state): State<AppState>, request: Request) -> Result<Response, Fault> {
    run_request(state, request, true).await
}

async fn run_request(state: AppState, request: Request, v2: bool) -> Result<Response, Fault> {
    let inference = state.inference.clone().ok_or_else(Fault::unavailable)?;
    let actor = state
        .auth
        .authorize(request.headers(), Grant::ImageDetection)
        .await?;
    let headers = detection_quota(&state, &actor, request.uri().path()).await?;
    let upload = inference
        .uploads
        .clone()
        .try_acquire_owned()
        .map_err(|_| Fault::Code(StatusCode::SERVICE_UNAVAILABLE, "INFERENCE_BUSY"))?;
    let multipart = Multipart::from_request(request, &state)
        .await
        .map_err(|error| ApiError::new(error.status(), "Invalid multipart upload"))?;
    let form = tokio::time::timeout(
        UPLOAD_TIMEOUT,
        read_upload(multipart, inference.max_upload_bytes),
    )
    .await
    .map_err(|_| ApiError::new(StatusCode::REQUEST_TIMEOUT, "Upload timed out"))??;
    let overlay = if v2 {
        let options = Options::parse(form.view.as_deref(), form.compute_regions.as_deref())?;
        let slot = inference
            .overlays
            .clone()
            .try_acquire_owned()
            .map_err(|_| Fault::Code(StatusCode::SERVICE_UNAVAILABLE, "OVERLAY_BUSY"))?;
        Some((options, slot))
    } else {
        None
    };
    let requested = form
        .model
        .as_deref()
        .ok_or_else(|| ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "model is required"))?;
    let (model, _) = inference
        .select_model(&state.db, requested, form.version.as_deref(), &actor)
        .await?;
    let image = form
        .image
        .ok_or_else(|| ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "image is required"))?;
    let result = inference.detect(model, image.into(), Some(upload)).await?;
    let Some((options, slot)) = overlay else {
        return Ok((headers, Json(result.rows)).into_response());
    };
    let task = tokio::task::spawn_blocking(move || {
        let _slot = slot;
        options.render(&result, &inference.models[model].spec)
    });
    let payload = tokio::time::timeout(WORK_TIMEOUT, task)
        .await
        .map_err(|_| Fault::unavailable())?
        .map_err(|_| Fault::unavailable())??;
    Ok((headers, Json(payload)).into_response())
}

async fn detection_quota(state: &AppState, actor: &Actor, path: &str) -> Result<HeaderMap, Fault> {
    let role = actor.role.as_deref().unwrap_or("service");
    let username = actor.username.as_deref().unwrap_or(&actor.subject);
    let (limit, seconds) = if role == "guest" {
        (24_u64, 86_400_i64)
    } else {
        (3000, 60)
    };
    let key = format!("construction-hazard-detection:rate_limit:{role}:{username}:POST:{path}");
    let (count, ttl): (u64, i64) = DETECTION_QUOTA
        .key(key)
        .arg(seconds)
        .invoke_async(&mut state.auth.redis())
        .await
        .map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "Detection quota unavailable",
            )
        })?;
    if count > limit {
        return Err(ApiError::new(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded").into());
    }
    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert("x-ratelimit-remaining", HeaderValue::from(limit - count));
    headers.insert("x-ratelimit-limit", HeaderValue::from(limit));
    headers.insert(
        "x-ratelimit-reset",
        HeaderValue::from(if ttl < 0 { seconds } else { ttl }),
    );
    Ok(headers)
}

impl Inference {
    fn catalog_scope<'a>(&'a self, actor: &'a Actor) -> Option<(&'a str, &'a str)> {
        self.catalog_tenant_id.as_deref().zip(actor.role.as_deref())
    }

    async fn visible_models(
        &self,
        db: &PgPool,
        actor: &Actor,
    ) -> Result<(Vec<&Model>, Option<&str>), Fault> {
        let mut models = Vec::new();
        let mut default = None;
        if let Some((tenant, role)) = self.catalog_scope(actor) {
            let keys: Vec<_> = self.models.iter().map(|model| &model.spec.id).collect();
            let rows: Vec<(String, Value, bool)> = sqlx::query_as(
                "SELECT model_key, definition, is_default FROM detection_model_catalog
                 WHERE model_key = ANY($1) ORDER BY display_order, model_key",
            )
            .bind(keys)
            .fetch_all(db)
            .await
            .map_err(catalog_error)?;
            for (key, definition, is_default) in rows {
                let definition = parse_catalog(definition)?;
                let Some(model) = self
                    .models
                    .iter()
                    .map(|model| &model.spec)
                    .find(|model| model.id == key && definition.version == model.version)
                else {
                    continue;
                };
                if definition.enabled && definition.permits(tenant, role) {
                    if is_default {
                        default = Some(model.id.as_str());
                    }
                    models.push(model);
                }
            }
        } else {
            for model in self.models.iter().map(|model| &model.spec) {
                if model.allowed_subjects.contains(&actor.subject) {
                    if self.default_model_id.as_deref() == Some(model.id.as_str()) {
                        default = Some(model.id.as_str());
                    }
                    models.push(model);
                }
            }
        }
        Ok((models, default))
    }

    pub(super) fn model(&self, index: usize) -> &LoadedModel {
        &self.models[index]
    }

    pub(super) async fn select_model(
        &self,
        db: &PgPool,
        requested: &str,
        version: Option<&str>,
        actor: &Actor,
    ) -> Result<(usize, Option<String>), Fault> {
        let index = self
            .models
            .iter()
            .position(|model| model.spec.id == requested)
            .ok_or(Fault::Code(StatusCode::CONFLICT, "MODEL_UNAVAILABLE"))?;
        let model = &self.models[index].spec;
        if version.is_some_and(|version| version != model.version) {
            return Err(Fault::Code(StatusCode::CONFLICT, "MODEL_UNAVAILABLE"));
        }
        let mut source_digest = None;
        let allowed = if let Some((tenant, role)) = self.catalog_scope(actor) {
            let definition: Option<Value> = sqlx::query_scalar(
                "SELECT definition FROM detection_model_catalog WHERE model_key = $1",
            )
            .bind(requested)
            .fetch_optional(db)
            .await
            .map_err(catalog_error)?;
            let definition = parse_catalog(definition.ok_or(unavailable_model())?)?;
            if !definition.enabled || definition.version != model.version {
                return Err(unavailable_model());
            }
            let allowed = definition.permits(tenant, role);
            source_digest = Some(definition.sha256);
            allowed
        } else {
            model.allowed_subjects.contains(&actor.subject)
        };
        if !allowed {
            return Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "Model is not allowed for this resource",
            )
            .into());
        }
        Ok((index, source_digest))
    }

    pub(super) async fn prepare(&self, model: usize) -> Result<(), Fault> {
        tokio::time::timeout(WORK_TIMEOUT, self.submit(model, None))
            .await
            .map_err(|_| inference_timeout())??;
        Ok(())
    }

    pub(super) async fn detect(
        &self,
        model: usize,
        image: Bytes,
        upload: Option<OwnedSemaphorePermit>,
    ) -> Result<Detection, Fault> {
        tokio::time::timeout(WORK_TIMEOUT, async {
            let decoded = self.preparation.decode(image, upload).await?;
            self.submit(model, Some(decoded))
                .await?
                .ok_or_else(Fault::unavailable)
        })
        .await
        .map_err(|_| inference_timeout())?
    }

    async fn submit(
        &self,
        model: usize,
        image: Option<PreparedImage>,
    ) -> Result<Option<Detection>, Fault> {
        // Only prepared frames enter the FIFO GPU admission queue.
        let permit = self
            .outstanding
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Fault::unavailable())?;
        let (reply, receive) = oneshot::channel();
        self.jobs
            .as_ref()
            .ok_or_else(Fault::unavailable)?
            .try_send(Job {
                model,
                image,
                reply,
                _permit: permit,
            })
            .map_err(|_| Fault::unavailable())?;
        receive.await.map_err(|_| Fault::unavailable())?
    }
}

fn inference_timeout() -> Fault {
    ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "Image inference timed out").into()
}

#[derive(Deserialize)]
struct CatalogModel {
    version: String,
    sha256: String,
    #[serde(default = "enabled")]
    enabled: bool,
    capabilities: Vec<String>,
    #[serde(default)]
    tenant_ids: Vec<String>,
    #[serde(default = "catalog_roles")]
    roles: Vec<String>,
}

fn enabled() -> bool {
    true
}

fn catalog_roles() -> Vec<String> {
    ["admin", "user", "guest", "super_admin"]
        .map(String::from)
        .into()
}

impl CatalogModel {
    fn permits(&self, tenant: &str, role: &str) -> bool {
        self.capabilities.iter().any(|item| item == "image")
            && self.tenant_ids.iter().any(|item| item == tenant)
            && self.roles.iter().any(|item| item == role)
    }
}

fn unavailable_model() -> Fault {
    Fault::Code(StatusCode::CONFLICT, "MODEL_UNAVAILABLE")
}

fn catalog_error(error: sqlx::Error) -> Fault {
    tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "model catalog query failed");
    Fault::Code(
        StatusCode::SERVICE_UNAVAILABLE,
        "MODEL_REGISTRY_UNAVAILABLE",
    )
}

fn parse_catalog(definition: Value) -> Result<CatalogModel, Fault> {
    serde_json::from_value(definition).map_err(|_| {
        Fault::Code(
            StatusCode::SERVICE_UNAVAILABLE,
            "MODEL_REGISTRY_UNAVAILABLE",
        )
    })
}

#[derive(Default)]
struct Upload {
    image: Option<Vec<u8>>,
    model: Option<String>,
    version: Option<String>,
    view: Option<String>,
    compute_regions: Option<String>,
}

async fn read_upload(mut form: Multipart, limit: usize) -> Result<Upload, Fault> {
    let mut result = Upload::default();
    while let Some(mut field) = form
        .next_field()
        .await
        .map_err(|error| ApiError::new(error.status(), "Invalid multipart field"))?
    {
        let name = field.name().unwrap_or_default().to_owned();
        let (size, slot) = match name.as_str() {
            "image" if result.image.is_none() => (limit, None),
            "model" => (256, Some(&mut result.model)),
            "model_version" => (256, Some(&mut result.version)),
            "view" => (256, Some(&mut result.view)),
            "compute_regions" => (256, Some(&mut result.compute_regions)),
            _ => {
                return Err(ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "Unexpected or duplicate form field",
                )
                .into());
            }
        };
        let mut bytes = Vec::new();
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|error| ApiError::new(error.status(), "Invalid multipart field"))?
        {
            if chunk.len() > size - bytes.len() {
                return Err(ApiError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Upload exceeds the configured limit",
                )
                .into());
            }
            bytes.extend_from_slice(&chunk);
        }
        if let Some(slot) = slot {
            let value = String::from_utf8(bytes).map_err(|_| {
                ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "Form fields must be UTF-8",
                )
            })?;
            if slot.replace(value).is_some() {
                return Err(ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "Duplicate form field",
                )
                .into());
            }
        } else if bytes.is_empty() {
            return Err(ApiError::new(StatusCode::BAD_REQUEST, "Empty upload file").into());
        } else {
            result.image = Some(bytes);
        }
    }
    Ok(result)
}

fn run_jobs(mut jobs: mpsc::Receiver<Job>, models: Arc<Vec<LoadedModel>>, device: i32) {
    let mut resident: Option<(usize, NativeEngine)> = None;
    while let Some(job) = jobs.blocking_recv() {
        if job.reply.is_closed() {
            continue;
        }
        let result = (|| {
            let model = &models[job.model];
            if resident
                .as_ref()
                .is_none_or(|(index, _)| *index != job.model)
            {
                drop(resident.take());
                let path = format!("/proc/self/fd/{}", model.artifact.as_raw_fd());
                let engine =
                    NativeEngine::new(&path, 1, model.spec.imgsz, device, false, 0.25, false)
                        .map_err(|_| Fault::unavailable())?;
                resident = Some((job.model, engine));
            }
            let Some(image) = job.image.as_ref() else {
                return Ok(None);
            };
            if job.reply.is_closed() {
                return Err(Fault::unavailable());
            }
            let engine = &mut resident.as_mut().ok_or_else(Fault::unavailable)?.1;
            let mut batch = engine
                .predict(&[Image {
                    w: image.width as usize,
                    h: image.height as usize,
                    data: &image.bgr,
                }])
                .map_err(|_| Fault::unavailable())?;
            let rows = batch.pop().ok_or_else(Fault::unavailable)?;
            let rows = rows
                .into_iter()
                .map(|[x1, y1, x2, y2, score, class]| {
                    (
                        x1 as i64,
                        y1 as i64,
                        x2 as i64,
                        y2 as i64,
                        score,
                        class as u32,
                    )
                })
                .collect();
            Ok(Some(Detection {
                width: image.width,
                height: image.height,
                rows: filter_conflicts(rows, &model.spec),
            }))
        })();
        let _ = job.reply.send(result);
    }
}

fn filter_conflicts(mut rows: Vec<DetectionRow>, model: &Model) -> Vec<DetectionRow> {
    let conflict_ids = CONFLICT_CLASSES
        .into_iter()
        .filter_map(|(positive, negative)| {
            let class = |code| {
                model
                    .classes
                    .iter()
                    .find(|class| class.code == code)
                    .map(|class| class.id)
            };
            Some((class(positive)?, class(negative)?))
        })
        .collect::<Vec<_>>();
    for remove_overlap in [true, false] {
        let mut discard = vec![false; rows.len()];
        for &(positive, negative) in &conflict_ids {
            for (i, outer) in rows.iter().enumerate().filter(|(_, row)| row.5 == positive) {
                for (j, inner) in rows.iter().enumerate().filter(|(_, row)| row.5 == negative) {
                    if remove_overlap {
                        if iou(outer, inner) > 0.5 {
                            discard[j] = true;
                        }
                    } else if contains(outer, inner) {
                        discard[j] = true;
                    } else if contains(inner, outer) {
                        discard[i] = true;
                    }
                }
            }
        }
        rows = rows
            .into_iter()
            .zip(discard)
            .filter_map(|(row, discard)| (!discard).then_some(row))
            .collect();
    }
    rows
}

fn iou(left: &DetectionRow, right: &DetectionRow) -> f64 {
    let area = |x1: i64, y1: i64, x2: i64, y2: i64| (x2 - x1 + 1).max(0) * (y2 - y1 + 1).max(0);
    let overlap = area(
        left.0.max(right.0),
        left.1.max(right.1),
        left.2.min(right.2),
        left.3.min(right.3),
    );
    let union =
        area(left.0, left.1, left.2, left.3) + area(right.0, right.1, right.2, right.3) - overlap;
    if union > 0 {
        overlap as f64 / union as f64
    } else {
        0.0
    }
}

fn contains(outer: &DetectionRow, inner: &DetectionRow) -> bool {
    outer.0 <= inner.0 && outer.1 <= inner.1 && outer.2 >= inner.2 && outer.3 >= inner.3
}

pub(super) enum Fault {
    Api(ApiError),
    Code(StatusCode, &'static str),
}

impl Fault {
    pub(super) fn unavailable() -> Self {
        Self::Code(StatusCode::SERVICE_UNAVAILABLE, "INFERENCE_UNAVAILABLE")
    }
}

impl From<ApiError> for Fault {
    fn from(error: ApiError) -> Self {
        Self::Api(error)
    }
}

impl IntoResponse for Fault {
    fn into_response(self) -> Response {
        match self {
            Self::Api(error) => error.into_response(),
            Self::Code(status, code) => (
                status,
                [(header::CACHE_CONTROL, "no-store")],
                Json(json!({"detail": {"code": code}})),
            )
                .into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use image::{DynamicImage, ImageFormat};

    use super::*;
    use crate::detection::Class;

    fn model() -> Model {
        Model {
            id: "test".into(),
            display_name: "Test Model".into(),
            version: "v1".into(),
            engine: "test.engine".into(),
            sha256: "0".repeat(64),
            imgsz: 640,
            allowed_subjects: HashSet::from([String::from("fixture")]),
            source_artifact: None,
            classes: vec![
                Class {
                    id: 7,
                    code: "helmet".into(),
                    display_name: "Hard hat".into(),
                    color: "#4CAF50".into(),
                },
                Class {
                    id: 8,
                    code: "no_helmet".into(),
                    display_name: "No hard hat".into(),
                    color: "#F44336".into(),
                },
            ],
        }
    }

    fn rgb_png(width: u32, height: u32) -> Vec<u8> {
        let mut png = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            width,
            height,
            image::Rgb([10, 20, 30]),
        ))
        .write_to(&mut png, ImageFormat::Png)
        .unwrap();
        png.into_inner()
    }

    #[test]
    fn verifies_and_keeps_exact_engine_bytes_after_source_changes() {
        let directory = tempfile::tempdir().unwrap();
        let mut spec = model();
        spec.engine = directory.path().join("test.engine");
        std::fs::write(&spec.engine, b"original bytes").unwrap();
        let wrong = stage_model(spec);
        assert!(wrong.is_err());

        let mut spec = model();
        spec.engine = directory.path().join("test.engine");
        spec.sha256 = format!("{:x}", Sha256::digest(b"original bytes"));
        let mut loaded = stage_model(spec).unwrap();
        std::fs::write(&loaded.spec.engine, b"modified input").unwrap();
        loaded.artifact.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = Vec::new();
        loaded.artifact.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"original bytes");
    }

    #[test]
    fn preserves_legacy_rows_and_callers_class_order() {
        let model = model();
        let rows = filter_conflicts(
            vec![
                (0, 0, 10, 10, 0.5, 7),
                (0, 0, 10, 10, 0.9, 8),
                (10, 10, 30, 30, 0.25, 99),
            ],
            &model,
        );
        let values = serde_json::to_value(&rows).unwrap();
        assert_eq!(
            values,
            json!([[0, 0, 10, 10, 0.5, 7], [10, 10, 30, 30, 0.25, 99]])
        );
    }

    #[tokio::test]
    #[ignore = "requires the local fixture CHD_RUST_API_TEST_ENGINE and a compatible GPU"]
    async fn runs_one_bounded_inference_on_a_verified_engine() {
        let path = std::env::var("CHD_RUST_API_TEST_ENGINE").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let mut spec = model();
        spec.engine = path.into();
        spec.sha256 = format!("{:x}", Sha256::digest(&bytes));
        let inference = start(Config {
            default_model_id: Some(spec.id.clone()),
            catalog_tenant_id: None,
            device: 0,
            max_upload_bytes: 1024 * 1024,
            max_image_pixels: 640 * 640,
            capacity: 1,
            models: vec![spec],
        })
        .unwrap();
        assert!(inference.prepare(0).await.is_ok());
        let Ok(result) = inference.detect(0, rgb_png(640, 640).into(), None).await else {
            panic!("configured local TensorRT engine failed to infer");
        };
        assert_eq!((result.width, result.height), (640, 640));
        drop(inference);
    }
}

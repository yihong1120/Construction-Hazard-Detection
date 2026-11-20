use std::{
    collections::VecDeque,
    fs::{File, Metadata},
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    time::SystemTime,
};

use axum::{
    Extension,
    body::{Body, Bytes},
    extract::{Query, State},
    http::{HeaderMap, StatusCode, header},
    response::Response,
};
use image::{ImageFormat, ImageReader, codecs::jpeg::JpegEncoder};
use serde::Deserialize;
use tokio::sync::Semaphore;

use super::{Settings, database_error, inaccessible, query::SITE_CONDITION, scope};
use crate::{ApiError, AppState, auth::Grant};

const MAX_BYTES: u64 = 20 * 1024 * 1024;
const MAX_PIXELS: u64 = 64 * 1024 * 1024;
const THUMBNAIL_EDGE: u32 = 360;
const CACHE_ITEMS: usize = 32;

pub(super) struct Store {
    root: PathBuf,
    permits: Arc<Semaphore>,
    cache: Mutex<VecDeque<Thumbnail>>,
}

struct Thumbnail {
    source: PathBuf,
    modified: SystemTime,
    size: u64,
    bytes: Bytes,
}

impl Store {
    pub(super) fn new(root: PathBuf) -> Self {
        Self {
            root,
            permits: Arc::new(Semaphore::new(4)),
            cache: Mutex::new(VecDeque::new()),
        }
    }

    pub(super) fn root(&self) -> &Path {
        self.root.as_path()
    }

    pub(super) fn file_digest(&self, path: &str) -> Result<[u8; 32], ApiError> {
        use sha2::{Digest, Sha256};

        let (file, _, _) = self.open(path)?;
        let mut reader = file.take(MAX_BYTES + 1);
        let mut digest = Sha256::new();
        let bytes = std::io::copy(&mut reader, &mut digest).map_err(|_| unavailable())?;
        if bytes > MAX_BYTES {
            return Err(too_large());
        }
        Ok(digest.finalize().into())
    }

    pub(super) async fn dimensions(self: &Arc<Self>, path: String) -> Option<(u32, u32)> {
        let permit = self.permits.clone().try_acquire_owned().ok()?;
        let store = self.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let (file, _, format) = store.open(&path).ok()?;
            dimensions(ImageReader::with_format(BufReader::new(file), format)).ok()
        })
        .await
        .ok()
        .flatten()
    }

    async fn serve(
        self: &Arc<Self>,
        path: String,
        thumbnail: bool,
    ) -> Result<(Bytes, &'static str), ApiError> {
        let permit =
            self.permits.clone().try_acquire_owned().map_err(|_| {
                ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "Evidence media busy")
            })?;
        let store = self.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let (file, metadata, format) = store.open(&path)?;
            if thumbnail {
                Ok((
                    store.thumbnail(&path, file, metadata, format)?,
                    "image/jpeg",
                ))
            } else {
                let kind = match format {
                    ImageFormat::Jpeg => "image/jpeg",
                    _ => "image/png",
                };
                let mut bytes = Vec::with_capacity(metadata.len() as usize);
                file.take(MAX_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|_| unavailable())?;
                if bytes.len() as u64 > MAX_BYTES {
                    return Err(too_large());
                }
                let expected = image::guess_format(&bytes).ok();
                if expected != Some(format) {
                    return Err(invalid_media());
                }
                Ok((Bytes::from(bytes), kind))
            }
        })
        .await
        .map_err(|_| unavailable())?
    }

    fn open(&self, name: &str) -> Result<(File, Metadata, ImageFormat), ApiError> {
        let relative = safe_relative(name)?;
        let path = self
            .root
            .join(relative)
            .canonicalize()
            .map_err(|_| unavailable())?;
        if !path.starts_with(&self.root) {
            return Err(inaccessible());
        }
        let format = match relative
            .extension()
            .and_then(|item| item.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("jpg" | "jpeg") => ImageFormat::Jpeg,
            Some("png") => ImageFormat::Png,
            _ => return Err(invalid_media()),
        };
        let metadata = path.metadata().map_err(|_| unavailable())?;
        if !metadata.is_file() {
            return Err(unavailable());
        }
        if metadata.len() > MAX_BYTES {
            return Err(too_large());
        }
        Ok((
            File::open(path).map_err(|_| unavailable())?,
            metadata,
            format,
        ))
    }

    fn thumbnail(
        &self,
        path: &str,
        file: File,
        metadata: Metadata,
        format: ImageFormat,
    ) -> Result<Bytes, ApiError> {
        let modified = metadata.modified().ok();
        if let Some(stamp) = modified
            && let Ok(cache) = self.cache.lock()
            && let Some(item) = cache.iter().find(|item| {
                item.source.as_os_str() == path
                    && item.modified == stamp
                    && item.size == metadata.len()
            })
        {
            return Ok(item.bytes.clone());
        }
        dimensions(ImageReader::with_format(BufReader::new(&file), format))?;
        // Rewind after the header-only guard.
        use std::io::Seek;
        let mut file = file;
        file.rewind().map_err(|_| unavailable())?;
        let output = ImageReader::with_format(BufReader::new(file), format)
            .decode()
            .map_err(|_| invalid_media())?
            .thumbnail(THUMBNAIL_EDGE, THUMBNAIL_EDGE)
            .into_rgb8();
        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, 78)
            .encode_image(&output)
            .map_err(|_| invalid_media())?;
        let bytes = Bytes::from(jpeg);
        if let Some(stamp) = modified
            && let Ok(mut cache) = self.cache.lock()
        {
            if cache.len() == CACHE_ITEMS {
                cache.pop_front();
            }
            cache.push_back(Thumbnail {
                source: path.into(),
                modified: stamp,
                size: metadata.len(),
                bytes: bytes.clone(),
            });
        }
        Ok(bytes)
    }
}

fn safe_relative(name: &str) -> Result<&Path, ApiError> {
    let path = Path::new(name);
    if name.is_empty()
        || name.len() > 1024
        || name.chars().any(|ch| ch.is_control() || ch == '\\')
        || name
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "Invalid path"));
    }
    Ok(path)
}

fn dimensions<R: std::io::BufRead + std::io::Seek>(
    source: ImageReader<R>,
) -> Result<(u32, u32), ApiError> {
    let (width, height) = source.into_dimensions().map_err(|_| invalid_media())?;
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(too_large());
    }
    Ok((width, height))
}

fn too_large() -> ApiError {
    ApiError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "Evidence image exceeds limits",
    )
}

fn invalid_media() -> ApiError {
    ApiError::new(StatusCode::BAD_REQUEST, "Unsupported image content")
}

fn unavailable() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "Image not found")
}

#[derive(Deserialize)]
pub(super) struct MediaQuery {
    image_path: String,
}

pub(super) async fn image(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Query(query): Query<MediaQuery>,
) -> Result<Response, ApiError> {
    media_response(state, settings, headers, query, false).await
}

pub(super) async fn thumbnail(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Query(query): Query<MediaQuery>,
) -> Result<Response, ApiError> {
    media_response(state, settings, headers, query, true).await
}

async fn media_response(
    state: AppState,
    settings: Settings,
    headers: HeaderMap,
    query: MediaQuery,
    thumbnail: bool,
) -> Result<Response, ApiError> {
    let (mut db, scope) = scope(&state, &settings, &headers, Grant::ViolationRead).await?;
    safe_relative(&query.image_path)?;
    let (ids, names) = scope.select_sites(None)?;
    let query_text = format!(
        "SELECT EXISTS (SELECT 1 FROM violations AS v WHERE {SITE_CONDITION} AND v.image_path = $3)",
    );
    let authorised = sqlx::query_scalar::<_, bool>(&query_text)
        .bind(ids)
        .bind(names)
        .bind(&query.image_path)
        .fetch_one(&mut *db)
        .await
        .map_err(database_error)?;
    if !authorised {
        return Err(inaccessible());
    }
    drop(db);
    let (bytes, kind) = settings.media.serve(query.image_path, thumbnail).await?;
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, kind)
        .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff")
        .header(header::CONTENT_DISPOSITION, "inline")
        .header(header::CACHE_CONTROL, "private, no-store")
        .body(Body::from(bytes))
        .expect("static evidence media headers"))
}

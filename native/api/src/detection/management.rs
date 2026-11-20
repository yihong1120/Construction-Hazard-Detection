use std::fs::File;
use std::os::fd::AsRawFd;

use axum::Json;
use axum::body::Body;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use chrono::{DateTime, NaiveDate, NaiveDateTime};
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use serde::Deserialize;
use serde_json::json;
use tokio_util::io::ReaderStream;

use super::SourceArtifact;
use super::runtime::{Fault, UPLOAD_TIMEOUT, verified_copy};
use crate::{ApiError, AppState, auth::Grant};

const MAX_SOURCE_BYTES: u64 = 1024 * 1024 * 1024;

pub(super) struct Artifact {
    file: File,
    size: u64,
    sha256: String,
    filename: String,
}

impl Artifact {
    pub(super) fn stage(spec: &SourceArtifact) -> anyhow::Result<Self> {
        let (file, size) = verified_copy(&spec.path, &spec.sha256, MAX_SOURCE_BYTES)?;
        let filename = spec
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("Published model filename must be UTF-8"))?;
        Ok(Self {
            file,
            size,
            sha256: spec.sha256.to_ascii_lowercase(),
            filename: filename.to_owned(),
        })
    }

    async fn response(
        &self,
        headers: &HeaderMap,
        id: &str,
        version: &str,
    ) -> Result<Response, Fault> {
        let etag = format!("\"{}\"", self.sha256);
        if headers
            .get(header::IF_NONE_MATCH)
            .is_some_and(|value| value.as_bytes() == etag.as_bytes())
        {
            return Ok((StatusCode::NOT_MODIFIED, [(header::ETAG, &etag)]).into_response());
        }

        // Reopening the verified descriptor gives each request its own read
        // offset, including when clients download concurrently.
        let path = format!("/proc/self/fd/{}", self.file.as_raw_fd());
        let file = tokio::fs::File::open(path)
            .await
            .map_err(|_| unavailable())?;
        let mut response = Response::new(Body::from_stream(ReaderStream::with_capacity(
            file,
            64 * 1024,
        )));
        let headers = response.headers_mut();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/octet-stream"),
        );
        headers.insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("private, no-store"),
        );
        let disposition = if self
            .filename
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
        {
            format!("attachment; filename=\"{}\"", self.filename)
        } else {
            format!(
                "attachment; filename*=utf-8''{}",
                utf8_percent_encode(&self.filename, NON_ALPHANUMERIC)
            )
        };
        for (name, value) in [
            ("content-disposition", disposition.as_str()),
            ("content-length", &self.size.to_string()),
            ("etag", &etag),
            ("x-model-sha256", &self.sha256),
            ("x-model-id", id),
            ("x-model-version", version),
        ] {
            headers.insert(
                name,
                HeaderValue::from_str(value).map_err(|_| unavailable())?,
            );
        }
        Ok(response)
    }
}

fn unavailable() -> Fault {
    Fault::Code(
        StatusCode::SERVICE_UNAVAILABLE,
        "MODEL_DOWNLOAD_UNAVAILABLE",
    )
}

#[derive(Deserialize)]
pub(super) struct UpdateModelRequest {
    model: String,
    last_update_time: String,
}

fn is_iso_timestamp(value: &str) -> bool {
    DateTime::parse_from_rfc3339(value).is_ok()
        || NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").is_ok()
        || NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f").is_ok()
        || NaiveDate::parse_from_str(value, "%Y-%m-%d").is_ok()
}

pub(super) async fn get_new_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<UpdateModelRequest>,
) -> Result<Response, Fault> {
    let actor = state.auth.authorize(&headers, Grant::ModelDownload).await?;
    let inference = state.inference.as_ref().ok_or_else(Fault::unavailable)?;
    if !is_iso_timestamp(&request.last_update_time) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Invalid last_update_time ISO 8601 timestamp",
        )
        .into());
    }
    let (index, digest) = inference
        .select_model(&state.db, &request.model, None, &actor)
        .await?;
    let model = inference.model(index);
    let artifact = model.source_artifact.as_ref().ok_or_else(unavailable)?;
    if digest.is_some_and(|digest| digest != artifact.sha256) {
        return Err(Fault::Code(StatusCode::CONFLICT, "MODEL_UNAVAILABLE"));
    }
    artifact
        .response(&headers, &model.spec.id, &model.spec.version)
        .await
}

pub(super) async fn model_file_update(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, Fault> {
    state
        .auth
        .authorize(request.headers(), Grant::ModelManage)
        .await?;
    let form = Multipart::from_request(request, &state)
        .await
        .map_err(|error| ApiError::new(error.status(), "Invalid multipart upload"))?;
    tokio::time::timeout(UPLOAD_TIMEOUT, require_update_fields(form))
        .await
        .map_err(|_| ApiError::new(StatusCode::REQUEST_TIMEOUT, "Upload timed out"))??;
    Ok((
        StatusCode::CONFLICT,
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({"detail": {
            "code": "IMMUTABLE_MODEL_REGISTRY",
            "message": "Publish to Hugging Face and synchronize the catalog key.",
        }})),
    )
        .into_response())
}

async fn require_update_fields(mut form: Multipart) -> Result<(), Fault> {
    let mut model = false;
    let mut file = false;
    while let Some(field) = form
        .next_field()
        .await
        .map_err(|error| ApiError::new(error.status(), "Invalid multipart field"))?
    {
        match field.name() {
            Some("model") if field.file_name().is_none() => model = true,
            Some("file") if field.file_name().is_some() => file = true,
            _ => (),
        }
        if model && file {
            return Ok(());
        }
    }
    Err(ApiError::new(
        StatusCode::UNPROCESSABLE_ENTITY,
        "model and file are required",
    )
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use sha2::{Digest, Sha256};

    #[tokio::test]
    async fn downloads_verified_original_through_independent_streams_and_etag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.pt");
        let original = b"verified published weights";
        std::fs::write(&path, original).unwrap();
        let sha256 = format!("{:x}", Sha256::digest(original));
        assert!(
            Artifact::stage(&SourceArtifact {
                path: path.clone(),
                sha256: "0".repeat(64)
            })
            .is_err()
        );
        let artifact = Artifact::stage(&SourceArtifact {
            path: path.clone(),
            sha256: sha256.clone(),
        })
        .unwrap();
        std::fs::write(&path, b"rewritten source").unwrap();

        let mut headers = HeaderMap::new();
        let first = artifact
            .response(&headers, "fixture", "v1")
            .await
            .ok()
            .unwrap();
        let second = artifact
            .response(&headers, "fixture", "v1")
            .await
            .ok()
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(
            first.headers()["content-disposition"],
            "attachment; filename=\"fixture.pt\""
        );
        assert_eq!(first.headers()["x-model-sha256"], sha256);
        assert_eq!(first.headers()["x-model-version"], "v1");
        assert_eq!(first.headers()["x-model-id"], "fixture");
        assert_eq!(first.headers()["cache-control"], "private, no-store");
        headers.insert(header::IF_NONE_MATCH, first.headers()[header::ETAG].clone());
        for response in [first, second] {
            let body = to_bytes(response.into_body(), 1024).await.unwrap();
            assert_eq!(body.as_ref(), original);
        }

        let cached = artifact
            .response(&headers, "fixture", "v1")
            .await
            .ok()
            .unwrap();
        assert_eq!(cached.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(
            cached.headers()[header::ETAG],
            headers[header::IF_NONE_MATCH]
        );
        assert!(to_bytes(cached.into_body(), 1).await.unwrap().is_empty());
    }

    #[test]
    fn accepts_client_iso_timestamps() {
        assert!(is_iso_timestamp("2026-11-20T03:12:04.050000+00:00"));
        assert!(is_iso_timestamp("2026-11-20T03:12:04"));
        assert!(is_iso_timestamp("2026-11-20"));
        assert!(!is_iso_timestamp("yesterday"));
    }
}

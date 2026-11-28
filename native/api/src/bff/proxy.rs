use std::{
    collections::HashMap,
    fs::File,
    net::{IpAddr, SocketAddr},
    os::fd::AsRawFd,
    sync::{Arc, LazyLock},
    time::Duration,
};

use axum::{
    body::{Body, Bytes, to_bytes},
    extract::{ConnectInfo, Path, Request, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header},
    response::{IntoResponse, Response},
};
use futures_util::{StreamExt, stream};
use serde_json::Value;
use tokio::{
    io::AsyncWriteExt,
    sync::{OwnedSemaphorePermit, Semaphore},
};
use tokio_util::io::ReaderStream;
use url::Url;

use crate::{ApiError, AppState};

use super::{Bff, loopback, no_store, session, single_header, state_bff};

const MAX_BODY: usize = 24 * 1024 * 1024;
const MAX_DOCUMENT_BODY: usize = 100 * 1024 * 1024;
static LARGE_REQUESTS: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(4)));

fn route_denied() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "bff_route_not_allowed")
}

fn unavailable() -> ApiError {
    ApiError::new(StatusCode::BAD_GATEWAY, "bff_upstream_unavailable")
}

pub(super) fn allowed_service(value: &str) -> bool {
    matches!(
        value,
        "db_management"
            | "streaming"
            | "streaming_web"
            | "fcm"
            | "notifications"
            | "violations"
            | "detect"
            | "detection"
            | "chat"
            | "company_agent_chat"
            | "files"
            | "file_manage"
            | "ai_access"
            | "ai_usage"
    )
}

pub(super) fn allowed_upstream(service: &str, upstream: &Url, public: &Url) -> bool {
    if service == "company_agent_chat" {
        return loopback(upstream)
            && matches!(
                upstream.path(),
                "/api/visionnaire/chat" | "/api/visionnaire/chat/"
            );
    }
    if loopback(upstream) && upstream.path() == "/" {
        return true;
    }
    let chat = format!("{}/chat", public.path().trim_end_matches('/'));
    service == "chat"
        && upstream.scheme() == "https"
        && upstream.origin() == public.origin()
        && (upstream.path() == chat || upstream.path().strip_suffix('/') == Some(chat.as_str()))
}

fn segments(path: &str) -> Option<Vec<&str>> {
    if path.is_empty() {
        return Some(Vec::new());
    }
    if path.len() > 2048 {
        return None;
    }
    let segments: Vec<_> = path.split('/').collect();
    segments
        .iter()
        .all(|part| {
            !part.is_empty()
                && *part != "."
                && *part != ".."
                && part.chars().all(|character| {
                    !character.is_control() && !matches!(character, '%' | '\\' | '?' | '#' | ':')
                })
        })
        .then_some(segments)
}

fn numeric_id(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_digit())
}

fn download_token(token: &str) -> bool {
    let mut parts = token.split('.');
    (0..3).all(|_| {
        parts.next().is_some_and(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_".contains(&byte))
        })
    }) && parts.next().is_none()
}

fn files(service: &str) -> bool {
    matches!(service, "files" | "file_manage")
}

fn allowed_path(service: &str, method: &Method, path: &[&str]) -> bool {
    let read = matches!(*method, Method::GET | Method::HEAD);
    let post = *method == Method::POST;
    let patch = *method == Method::PATCH;
    let delete = *method == Method::DELETE;
    let put = *method == Method::PUT;
    match service {
        "streaming" | "streaming_web" => match path {
            ["labels" | "overlay-languages"] => read,
            ["webrtc", "ice-servers"]
            | ["streams", _]
            | ["metadata", "stream-id", _, _]
            | ["stream-playback", "sessions", _, "index.m3u8"] => read,
            ["stream-playback"] | ["stream-playback", "batch" | "release"] => post,
            _ => false,
        },
        "db_management" => match path {
            ["list_sites" | "list_users" | "list_groups" | "group_stream_limit" | "me"]
            | ["admin", "users" | "capabilities"]
            | ["users", "assignable-roles"]
            | ["legal", "documents"]
            | ["list_pending_users" | "list_features" | "list_group_features"] => read,
            ["admin", "website-access"]
            | ["admin", "website-access", "context" | "gateway" | "audit"] => read,
            ["admin", "tenants" | "deployments"] => read || post,
            ["admin", "tenants" | "deployments", id] => uuid::Uuid::parse_str(id).is_ok() && patch,
            ["admin", "website-access", "main" | "ai" | "wiki"] => put,
            ["admin", "website-access", "main" | "ai" | "wiki", "rules"] => post,
            [
                "admin",
                "website-access",
                "main" | "ai" | "wiki",
                "rules",
                id,
            ] => uuid::Uuid::parse_str(id).is_ok() && (put || delete),
            ["deployment-enrollment-codes"] => read || post,
            ["deployment-enrollment-codes", "expiry-options"] => read,
            ["deployment-enrollment-codes", id] => uuid::Uuid::parse_str(id).is_ok() && delete,
            ["sites", id, "stream-config"] => numeric_id(id) && (read || put),
            ["sites", id, "stream-models"] => numeric_id(id) && read,
            [
                "add_user_to_site"
                | "add_user"
                | "remove_user_from_site"
                | "create_stream_config"
                | "create_group"
                | "create_site"
                | "add_group_to_site"
                | "remove_group_from_site"
                | "create_feature"
                | "update_group_feature",
            ] => post,
            [
                "update_group"
                | "update_site"
                | "approve_user_signup"
                | "update_user_role"
                | "update_user_group"
                | "update_feature"
                | "set_user_status"
                | "update_user_profile"
                | "update_username"
                | "update_username_id"
                | "admin_update_password"
                | "admin_update_password_userid",
            ] => put,
            ["admin", "users", id, "approval"] => numeric_id(id) && patch,
            ["delete_group" | "delete_site" | "delete_feature" | "delete_user"] => delete,
            ["stream_config", "update", id] => numeric_id(id) && put,
            ["delete_stream_config", id] => numeric_id(id) && delete,
            ["api", "playback", "sessions" | "walls"]
            | ["api", "playback", "sessions", "renew"] => post,
            ["api", "playback", "sessions", _] => delete,
            _ => false,
        },
        "violations" => match path {
            ["my_sites" | "violations" | "get_violation_image" | "get_violation_thumbnail"]
            | ["violations", "analytics" | "filter-options" | "next"] => read,
            [
                "violations",
                id,
                "feedback-options" | "review-options" | "audit-log",
            ]
            | ["violations", id] => numeric_id(id) && read,
            ["violations", id, "feedback"] => numeric_id(id) && post,
            ["violations", id, "review"] => numeric_id(id) && patch,
            _ => false,
        },
        "fcm" | "notifications" => match path {
            ["notifications"] | ["notifications", "unread_count" | "device-status"] => read,
            ["notifications", "site_preferences"] => read || put,
            ["notifications", "test"] => post,
            ["devices"] => put || delete,
            ["notifications", "read_all"] => patch,
            ["notifications", id, "read"] => numeric_id(id) && patch,
            ["notifications", id] => numeric_id(id) && delete,
            _ => false,
        },
        "detection" | "detect" => match path {
            ["models"] => read,
            ["detect" | "get_new_model" | "model_file_update"] | ["v2", "detect"] => post,
            _ => false,
        },
        "files" | "file_manage" => match path {
            ["files" | "document_categories" | "document_types"]
            | ["files", "workflow_counts"]
            | ["sign", "tasks" | "groups" | "sites" | "users"] => read,
            ["upload" | "create_from_template" | "create_photo_doc" | "create_audit_fix_doc"]
            | ["files", "bundle"] => post,
            ["files" | "versions" | "fields" | "export-jobs", id]
            | ["sign", "task" | "assignments", id] => numeric_id(id) && read,
            ["files", id, "links"] => numeric_id(id) && (read || post || put),
            ["files", id, "bundle"] => numeric_id(id) && (read || post),
            ["files", id, "exports"] => numeric_id(id) && post,
            ["version" | "delete" | "file-links", id] => numeric_id(id) && delete,
            ["update", id] => numeric_id(id) && put,
            ["fill", id]
            | ["fill", "audit-fix", id]
            | ["sign", "setup" | "submit" | "finalize", id] => numeric_id(id) && post,
            ["drafts", _] => read || put || delete,
            ["drafts", _, "attachments"] => post,
            ["drafts", _, "attachments", _] => read || delete,
            ["versions" | "exports", id, "generate_temp_url"] => numeric_id(id) && read,
            ["download", "temp", token] => download_token(token) && read,
            _ => false,
        },
        "ai_access" => match path {
            ["api", "company-ai", "policy" | "devices"]
            | ["api", "company-ai", "admin", "devices"]
            | ["api", "company-ai", "enrollments", _] => read,
            ["api", "company-ai", "devices", _, "revoke"]
            | ["api", "company-ai", "admin", "devices", _, "revoke"]
            | ["api", "company-ai", "enrollments", _, "approve"] => post,
            _ => false,
        },
        "ai_usage" => match path {
            ["v1", "me", "usage", "daily"]
            | [
                "v1",
                "admin",
                "usage",
                "capabilities"
                | "groups"
                | "summary"
                | "users"
                | "records"
                | "unattributed-records",
            ] => read,
            _ => false,
        },
        "chat" if path.first() == Some(&"v1") => match path {
            ["v1", "chats"] => *method == Method::GET || post,
            ["v1", "chats" | "jobs" | "artifacts", id] => {
                uuid::Uuid::parse_str(id).is_ok() && *method == Method::GET
            }
            ["v1", "chats", id, "messages" | "attachments"] | ["v1", "jobs", id, "cancel"] => {
                uuid::Uuid::parse_str(id).is_ok() && post
            }
            _ => false,
        },
        // Unversioned chat remains on the configured legacy service.
        "chat" => read || post || put || patch || delete,
        _ => false,
    }
}

fn upstream_url(origin: &Url, path: &[&str], uri: &Uri) -> Result<Url, ApiError> {
    let mut target = origin.clone();
    let path = if path.is_empty() { &[""][..] } else { path };
    target
        .path_segments_mut()
        .map_err(|_| unavailable())?
        .pop_if_empty()
        .extend(path);
    let query = uri.query();
    if query.is_some_and(|value| value.len() > 8192) {
        return Err(ApiError::new(
            StatusCode::URI_TOO_LONG,
            "bff_query_too_large",
        ));
    }
    target.set_query(query);
    Ok(target)
}

fn proxied_headers(source: &HeaderMap) -> HeaderMap {
    let mut result = HeaderMap::new();
    for name in [
        header::ACCEPT,
        header::ACCEPT_LANGUAGE,
        header::CONTENT_ENCODING,
        header::CONTENT_TYPE,
        header::IF_MATCH,
        header::IF_NONE_MATCH,
        header::IF_MODIFIED_SINCE,
        header::IF_UNMODIFIED_SINCE,
        header::RANGE,
        header::IF_RANGE,
        header::ORIGIN,
        header::REFERER,
        header::HeaderName::from_static("idempotency-key"),
    ] {
        if let Some(value) = source.get(&name) {
            result.insert(name, value.clone());
        }
    }
    result
}

fn response_headers(source: &HeaderMap) -> HeaderMap {
    let mut result = HeaderMap::new();
    for name in [
        header::ACCEPT_RANGES,
        header::CONTENT_DISPOSITION,
        header::CONTENT_LANGUAGE,
        header::CONTENT_RANGE,
        header::CONTENT_TYPE,
        header::ETAG,
        header::LAST_MODIFIED,
        header::RETRY_AFTER,
        header::HeaderName::from_static("x-model-id"),
        header::HeaderName::from_static("x-model-version"),
        header::HeaderName::from_static("x-model-sha256"),
        header::HeaderName::from_static("x-ratelimit-limit"),
        header::HeaderName::from_static("x-ratelimit-remaining"),
        header::HeaderName::from_static("x-ratelimit-reset"),
    ] {
        if let Some(value) = source.get(&name) {
            result.insert(name, value.clone());
        }
    }
    result.insert(header::X_CONTENT_TYPE_OPTIONS, "nosniff".parse().unwrap());
    result.insert(
        header::CONTENT_SECURITY_POLICY,
        "sandbox; default-src 'none'".parse().unwrap(),
    );
    result
}

fn rewrite_media(value: &mut Value, public_origin: &str) -> bool {
    match value {
        Value::Array(items) => items.iter_mut().fold(false, |changed, item| {
            rewrite_media(item, public_origin) || changed
        }),
        Value::Object(items) => items.iter_mut().fold(false, |changed, (name, item)| {
            if matches!(name.as_str(), "image_url" | "thumbnail_url")
                && let Some(url) = item.as_str()
                && let relative = url.strip_prefix(public_origin).unwrap_or(url)
                && let Some((path, query)) = relative.split_once('?')
                && matches!(path, "/get_violation_image" | "/get_violation_thumbnail")
                && !query.is_empty()
                && query.bytes().all(|character| {
                    character.is_ascii_graphic() && !matches!(character, b'#' | b'\\')
                })
            {
                *item = Value::String(format!("{public_origin}/bff/violations{relative}"));
                true
            } else {
                rewrite_media(item, public_origin) || changed
            }
        }),
        _ => false,
    }
}

fn rewrite_download(value: &mut Value, service: &str, public_origin: &str, api_root: &str) -> bool {
    let Some(url) = value.get_mut("temp_url") else {
        return false;
    };
    let Some(source) = url.as_str() else {
        return false;
    };
    let file_root = format!("{api_root}/file_manage/download/temp/");
    let Some(token) = source
        .strip_prefix("/download/temp/")
        .or_else(|| source.strip_prefix(&file_root))
        .filter(|token| download_token(token))
    else {
        return false;
    };
    *url = Value::String(format!(
        "{public_origin}/bff/{service}/download/temp/{token}"
    ));
    true
}

fn trusted_ai_client(headers: &HeaderMap, peer: Option<SocketAddr>) -> Option<IpAddr> {
    if !peer?.ip().is_loopback() {
        return None;
    }
    // The dedicated Nginx AI Access location overwrites this header with one
    // peer address. The listener is loopback-only. Never trust a forwarded chain.
    single_header(headers, header::HeaderName::from_static("x-forwarded-for"))?
        .parse::<IpAddr>()
        .ok()
        .map(|ip| ip.to_canonical())
}

enum ReplayBody {
    Memory(Bytes),
    Disk {
        file: File,
        length: usize,
        _permit: OwnedSemaphorePermit,
    },
}

fn body_too_large() -> ApiError {
    ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "bff_body_too_large")
}

impl ReplayBody {
    async fn read(body: Body, large: bool, known_length: Option<usize>) -> Result<Self, ApiError> {
        if !large || known_length.is_some_and(|length| length <= MAX_BODY) {
            return to_bytes(body, MAX_BODY)
                .await
                .map(Self::Memory)
                .map_err(|_| body_too_large());
        }
        if known_length.is_some_and(|length| length > MAX_DOCUMENT_BODY) {
            return Err(body_too_large());
        }
        let permit = LARGE_REQUESTS
            .clone()
            .try_acquire_owned()
            .map_err(|_| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "bff_upload_busy"))?;
        let mut stream = body.into_data_stream();
        let mut output =
            tokio::fs::File::from_std(tempfile::tempfile().map_err(|_| unavailable())?);
        let mut length = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| unavailable())?;
            if chunk.len() > MAX_DOCUMENT_BODY - length {
                return Err(body_too_large());
            }
            output.write_all(&chunk).await.map_err(|_| unavailable())?;
            length += chunk.len();
        }
        output.flush().await.map_err(|_| unavailable())?;
        Ok(Self::Disk {
            file: output.into_std().await,
            length,
            _permit: permit,
        })
    }

    async fn send(
        &self,
        bff: &Bff,
        method: &Method,
        target: &Url,
        request_headers: &HeaderMap,
        access_token: &str,
    ) -> Result<reqwest::Response, ApiError> {
        let request = bff
            .0
            .upstream_http
            .request(method.clone(), target.clone())
            .headers(request_headers.clone())
            .bearer_auth(access_token);
        let request =
            match self {
                Self::Memory(body) => request.body(body.clone()),
                Self::Disk { file, length, .. } => {
                    let file = tokio::fs::File::open(format!("/proc/self/fd/{}", file.as_raw_fd()))
                        .await
                        .map_err(|_| unavailable())?;
                    request.header(header::CONTENT_LENGTH, *length).body(
                        reqwest::Body::wrap_stream(ReaderStream::with_capacity(file, 64 * 1024)),
                    )
                }
            };
        tokio::time::timeout(Duration::from_secs(30), request.send())
            .await
            .map_err(|_| unavailable())?
            .map_err(|_| unavailable())
    }
}

pub(super) async fn chat_root(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, ApiError> {
    forward(State(state), Path(("chat".into(), String::new())), request).await
}

fn upstream_for<'a>(
    upstreams: &'a HashMap<String, Url>,
    service: &str,
    path: &[&str],
) -> Option<&'a Url> {
    if service == "chat" && path.first() == Some(&"v1") {
        return upstreams.get("company_agent_chat");
    }
    let alias = match service {
        "streaming_web" => "streaming",
        "streaming" => "streaming_web",
        "detect" => "detection",
        "detection" => "detect",
        "notifications" => "fcm",
        "fcm" => "notifications",
        "files" => "file_manage",
        "file_manage" => "files",
        service => service,
    };
    upstreams.get(service).or_else(|| upstreams.get(alias))
}

pub(super) async fn forward(
    State(state): State<AppState>,
    Path((service, path)): Path<(String, String)>,
    request: Request,
) -> Result<Response, ApiError> {
    let bff = state_bff(&state, request.headers())?;
    let path = segments(&path).ok_or_else(route_denied)?;
    if !allowed_path(&service, request.method(), &path) {
        return Err(route_denied());
    }
    let origin = upstream_for(&bff.0.urls.upstreams, &service, &path).ok_or_else(route_denied)?;
    let target = upstream_url(origin, &path, request.uri())?;
    let method = request.method().clone();
    let mutation = !matches!(method, Method::GET | Method::HEAD);
    let mut active = session::active(&state, &bff, request.headers(), mutation).await?;
    let mut request_headers = proxied_headers(request.headers());
    let authority =
        HeaderValue::from_str(&bff.0.urls.browser_authority).map_err(|_| unavailable())?;
    request_headers.insert(header::HOST, authority.clone());
    request_headers.insert("x-forwarded-host", authority);
    request_headers.insert(
        "x-forwarded-proto",
        HeaderValue::from_static(if bff.0.urls.secure { "https" } else { "http" }),
    );
    request_headers.insert("x-bff-request", HeaderValue::from_static("1"));
    if service == "ai_access" {
        let peer = request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|value| value.0);
        let ip = trusted_ai_client(request.headers(), peer)
            .ok_or_else(|| ApiError::new(StatusCode::FORBIDDEN, "ai_access_network_denied"))?;
        let proof = bff.0.company_ai_network_proxy_key.clone().ok_or_else(|| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "company_ai_network_proxy_not_configured",
            )
        })?;
        request_headers.insert("x-company-ai-network-proxy", proof);
        request_headers.insert(
            "x-company-ai-client-ip",
            HeaderValue::from_str(&ip.to_string()).map_err(|_| unavailable())?,
        );
        // The browser's explicit action marker is separate from the proxy's
        // network proof and the BFF's own CSRF check.
        if let Some(action) = single_header(
            request.headers(),
            header::HeaderName::from_static("x-company-ai-action"),
        ) {
            request_headers.insert(
                "x-company-ai-action",
                HeaderValue::from_str(action).map_err(|_| unavailable())?,
            );
        }
    }
    if service == "db_management"
        && path.starts_with(&["admin", "website-access"])
        && let Some(ip) = crate::accounts::website_client_ip(&request)
    {
        let value = HeaderValue::from_str(&ip.to_string()).map_err(|_| unavailable())?;
        request_headers.insert("x-real-ip", value);
    }
    if bff.0.shared_session_key.is_some()
        && service == "db_management"
        && path.starts_with(&["api", "playback"])
    {
        let mut cookie = HeaderValue::from_str(&format!("{}={}", bff.session_cookie(), active.id))
            .map_err(|_| unavailable())?;
        cookie.set_sensitive(true);
        request_headers.insert(header::COOKIE, cookie);
    }
    let length = single_header(request.headers(), header::CONTENT_LENGTH)
        .and_then(|value| value.parse().ok());
    let body = ReplayBody::read(
        request.into_body(),
        mutation && (files(&service) || service == "chat"),
        length,
    )
    .await?;
    let mut upstream = body
        .send(
            &bff,
            &method,
            &target,
            &request_headers,
            &active.session.access_token,
        )
        .await?;
    if upstream.status() == StatusCode::UNAUTHORIZED {
        drop(upstream);
        session::refresh_for_proxy_retry(&state, &bff, &mut active).await?;
        upstream = body
            .send(
                &bff,
                &method,
                &target,
                &request_headers,
                &active.session.access_token,
            )
            .await?;
    }
    let ttl = active.session.ttl(&bff)?;
    let status = upstream.status();
    if status.is_redirection() && status != StatusCode::NOT_MODIFIED {
        return Err(unavailable());
    }
    let mut headers = response_headers(upstream.headers());
    let media_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.split(';').next())
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    let event_stream = media_type == "text/event-stream" && status == StatusCode::OK;
    let model_file = status.is_success()
        && media_type == "application/octet-stream"
        && matches!(service.as_str(), "detect" | "detection")
        && path == ["get_new_model"];
    let file_download = status.is_success()
        && !mutation
        && ((files(&service)
            && matches!(
                path.as_slice(),
                ["download", "temp", _] | ["drafts", _, "attachments", _]
            ))
            || (service == "chat" && matches!(path.as_slice(), ["v1", "artifacts", _])));
    if (file_download || model_file || method == Method::HEAD)
        && let Some(length) = upstream.headers().get(header::CONTENT_LENGTH)
    {
        headers.insert(header::CONTENT_LENGTH, length.clone());
    }
    let body = if event_stream || model_file || file_download {
        if event_stream {
            headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
        }
        Body::from_stream(stream::try_unfold(
            (upstream, state.shutdown),
            |(mut upstream, shutdown)| async move {
                tokio::select! {
                    _ = shutdown.cancelled() => Ok(None),
                    chunk = upstream.chunk() => chunk.map(|chunk| chunk.map(|chunk| (chunk, (upstream, shutdown)))),
                }
            },
        ))
    } else {
        let json_response = media_type == "application/json";
        let data = tokio::time::timeout(Duration::from_secs(30), async {
            let mut bytes = Vec::new();
            while let Some(chunk) = upstream.chunk().await.map_err(|_| unavailable())? {
                if bytes.len() + chunk.len() > MAX_BODY {
                    return Err(unavailable());
                }
                bytes.extend_from_slice(&chunk);
            }
            Ok(bytes)
        })
        .await
        .map_err(|_| unavailable())??;
        let temp_url = files(&service)
            && status == StatusCode::OK
            && matches!(
                path.as_slice(),
                ["versions" | "exports", _, "generate_temp_url"]
            );
        let evidence = service == "violations" && status.is_success();
        if (temp_url || evidence)
            && json_response
            && let Ok(mut value) = serde_json::from_slice::<Value>(&data)
            && if evidence {
                rewrite_media(&mut value, &bff.0.urls.browser_origin)
            } else {
                let public_origin = &bff.0.urls.browser_origin;
                let api_root = bff
                    .0
                    .urls
                    .public_api_base_url
                    .strip_prefix(public_origin)
                    .unwrap_or_default();
                rewrite_download(&mut value, &service, public_origin, api_root)
            }
        {
            headers.remove(header::ETAG);
            Body::from(serde_json::to_vec(&value).map_err(|_| unavailable())?)
        } else {
            Body::from(data)
        }
    };
    let mut response = no_store((status, headers, body).into_response());
    bff.set_cookie(&mut response, bff.session_cookie(), &active.id, ttl);
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwards_only_explicit_services_operations_and_encoded_paths() {
        for path in ["../auth/session", "a/%2F", "https://example.org/a", "a/\\b"] {
            assert!(segments(path).is_none());
        }
        assert!(allowed_path(
            "violations",
            &Method::PATCH,
            &["violations", "42", "review"]
        ));
        assert!(!allowed_path("violations", &Method::POST, &["upload"]));
        assert!(!allowed_path("streaming", &Method::GET, &["media-auth"]));
        let target = upstream_url(
            &Url::parse("http://127.0.0.1:9123").unwrap(),
            &segments("metadata/stream-id/demo/Y2FtZXJhLTE=").unwrap(),
            &"/router?overlay=on".parse().unwrap(),
        )
        .unwrap();
        assert_eq!(
            target.as_str(),
            "http://127.0.0.1:9123/metadata/stream-id/demo/Y2FtZXJhLTE=?overlay=on",
        );
    }

    #[test]
    fn rewrites_only_local_evidence_paths() {
        let mut value = serde_json::json!({"items": [{
            "image_url": "/get_violation_image?image_path=demo%2Fevidence.jpg",
            "thumbnail_url": "https://untrusted.example/get_violation_thumbnail?path=image",
        }]});
        assert!(rewrite_media(&mut value, "https://demo.example"));
        assert_eq!(
            value["items"][0]["image_url"],
            "https://demo.example/bff/violations/get_violation_image?image_path=demo%2Fevidence.jpg"
        );
        assert_eq!(
            value["items"][0]["thumbnail_url"],
            "https://untrusted.example/get_violation_thumbnail?path=image"
        );
    }

    #[test]
    fn keeps_chat_on_the_direct_same_origin_nginx_path() {
        let public = Url::parse("https://demo.example/hazard/api").unwrap();
        let chat = Url::parse("https://demo.example/hazard/api/chat/").unwrap();
        assert!(allowed_upstream("chat", &chat, &public));
        assert!(!allowed_upstream("files", &chat, &public));
        for raw in [
            "https://foreign.example/hazard/api/chat",
            "https://demo.example/bff/chat",
        ] {
            assert!(!allowed_upstream(
                "chat",
                &Url::parse(raw).unwrap(),
                &public
            ));
        }
        let uri = "/bff/chat/chat/upload-policy?kind=pdf".parse().unwrap();
        let target = upstream_url(&chat, &["chat", "upload-policy"], &uri).unwrap();
        assert_eq!(
            target.as_str(),
            "https://demo.example/hazard/api/chat/chat/upload-policy?kind=pdf"
        );
    }

    #[test]
    fn keeps_company_agent_and_legacy_chat_routes_separate() {
        let public = Url::parse("https://demo.example/hazard/api").unwrap();
        let legacy = public.join("/hazard/api/chat/").unwrap();
        let agent = Url::parse("http://127.0.0.1:3000/api/visionnaire/chat").unwrap();
        let mut upstreams = HashMap::from([("chat".to_string(), legacy.clone())]);
        let path = ["v1", "chats"];
        assert!(upstream_for(&upstreams, "chat", &path).is_none());
        assert!(allowed_upstream("company_agent_chat", &agent, &public));
        assert!(!allowed_upstream("company_agent_chat", &legacy, &public));
        assert!(!allowed_upstream(
            "company_agent_chat",
            &agent.join("/").unwrap(),
            &public
        ));
        upstreams.insert("company_agent_chat".to_string(), agent.clone());
        assert_eq!(upstream_for(&upstreams, "chat", &["chats"]), Some(&legacy));
        let target = upstream_url(
            upstream_for(&upstreams, "chat", &path).unwrap(),
            &path,
            &"/bff/chat/v1/chats".parse().unwrap(),
        )
        .unwrap();
        assert_eq!(
            target.as_str(),
            "http://127.0.0.1:3000/api/visionnaire/chat/v1/chats"
        );
    }

    #[test]
    fn company_agent_chat_exposes_only_owner_operations_and_request_credentials_stay_server_side() {
        let id = "359d0d8a-6882-4b36-9360-cf84194857d9";
        for (method, suffix) in [
            (Method::GET, "chats".into()),
            (Method::POST, "chats".into()),
            (Method::GET, format!("chats/{id}")),
            (Method::POST, format!("chats/{id}/messages")),
            (Method::POST, format!("chats/{id}/attachments")),
            (Method::GET, format!("jobs/{id}")),
            (Method::POST, format!("jobs/{id}/cancel")),
            (Method::GET, format!("artifacts/{id}")),
        ] {
            let versioned = format!("v1/{suffix}");
            let path = segments(&versioned).unwrap();
            assert!(allowed_path("chat", &method, &path));
            assert!(!allowed_path("company_agent_chat", &method, &path));
        }
        assert!(!allowed_path("chat", &Method::GET, &["v1", "jobs"]));
        assert!(!allowed_path("chat", &Method::GET, &["v1", "chats", "123"]));
        assert!(!allowed_path(
            "chat",
            &Method::DELETE,
            &["v1", "artifacts", id]
        ));
        let mut browser = HeaderMap::new();
        for name in [
            "authorization",
            "cookie",
            "x-user-id",
            "x-openwebui-user-id",
            "x-forwarded-user",
        ] {
            browser.insert(name, HeaderValue::from_static("browser-supplied"));
        }
        assert!(proxied_headers(&browser).is_empty());
    }

    #[test]
    fn honours_file_actions_and_rewrites_only_relative_download_capabilities() {
        assert!(allowed_path(
            "file_manage",
            &Method::GET,
            &["versions", "42", "generate_temp_url"]
        ));
        assert!(allowed_path(
            "files",
            &Method::GET,
            &["download", "temp", "abc.def.ghi"]
        ));
        assert!(!allowed_path(
            "files",
            &Method::POST,
            &["notifications", "send_user_notification"]
        ));
        assert!(!allowed_path(
            "files",
            &Method::GET,
            &["download", "temp", "not-a-token"]
        ));
        let mut value =
            serde_json::json!({"temp_url": "/hazard/api/file_manage/download/temp/abc.def.ghi"});
        assert!(rewrite_download(
            &mut value,
            "files",
            "https://demo.example",
            "/hazard/api"
        ));
        assert_eq!(
            value["temp_url"],
            "https://demo.example/bff/files/download/temp/abc.def.ghi"
        );
        assert!(!rewrite_download(
            &mut value,
            "files",
            "https://demo.example",
            "/hazard/api"
        ));
    }

    #[test]
    fn accepts_only_the_loopback_proxy_network_provenance() {
        let mut headers = HeaderMap::new();
        let loopback = Some("127.0.0.1:8006".parse().unwrap());
        headers.insert("x-forwarded-for", "192.168.1.23".parse().unwrap());
        headers.insert("x-company-ai-client-ip", "203.0.113.2".parse().unwrap());
        assert_eq!(
            trusted_ai_client(&headers, loopback),
            Some("192.168.1.23".parse().unwrap())
        );
        assert!(trusted_ai_client(&headers, Some("192.168.1.23:40000".parse().unwrap())).is_none());
        headers.append("x-forwarded-for", "192.168.1.24".parse().unwrap());
        assert!(trusted_ai_client(&headers, loopback).is_none());
        assert!(!allowed_path(
            "ai_access",
            &Method::POST,
            &["api", "company-ai", "device", "start"]
        ));
        assert!(allowed_path(
            "ai_usage",
            &Method::GET,
            &["v1", "me", "usage", "daily"]
        ));
    }
}

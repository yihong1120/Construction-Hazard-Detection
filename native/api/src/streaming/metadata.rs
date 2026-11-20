//! Read-only, site-scoped live metadata for the standalone Rust profile.
use std::{
    collections::HashMap,
    convert::Infallible,
    io::Write,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::{
        Path, Query, RawQuery, State,
        ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade, close_code},
    },
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::Response,
    routing::get,
};
use base64::{Engine, engine::general_purpose::URL_SAFE};
use construction_stream_contract::{generation_key, metadata_key_from_stream_id};
use futures_util::{StreamExt, stream};
use redis::{
    AsyncConnectionConfig, Client, RedisResult,
    aio::{ConnectionLike, ConnectionManager, MultiplexedConnection},
    streams::{StreamId, StreamRangeReply, StreamReadReply},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    sync::watch,
    task::JoinHandle,
    time::{Instant, sleep_until, timeout},
};
use tokio_util::sync::CancellationToken;

use super::playback::{
    Playback,
    overlay_lease::{OverlayLease, backend_mode},
};
use crate::{
    ApiError, AppState,
    auth::{Authorizer, Grant, websocket_credentials},
};

const CONNECTION_TIMEOUT: Duration = Duration::from_secs(3);
const GENERATION_CHECK_INTERVAL: Duration = Duration::from_secs(1);
const ACCESS_CHECK_INTERVAL: Duration = Duration::from_secs(5);
const SSE_KEEPALIVE: Duration = Duration::from_secs(15);
const WEBSOCKET_SEND_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_WEBSOCKET_COMMAND_BYTES: usize = 4 * 1024;
const REDIS_BLOCK_MS: usize = 2_000;
const READ_BATCH: usize = 64;
const MAX_CAMERAS: usize = 128;
const MAX_SUBSCRIBERS: usize = 1_024;
const MAX_OVERLAY_BYTES: usize = 1024 * 1024;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/metadata/stream-id/{site}/{stream_id}",
            get(metadata_stream),
        )
        .route(
            "/ws/metadata-id/{site}/{stream_id}",
            get(metadata_websocket),
        )
}

#[derive(Clone)]
pub struct Metadata {
    hub: Arc<Fanout>,
    prefix: Arc<str>,
}

struct Fanout {
    client: Client,
    control: ConnectionManager,
    state: Mutex<FanoutState>,
}

#[derive(Default)]
struct FanoutState {
    readers: HashMap<String, Reader>,
    subscribers: usize,
}

struct Reader {
    updates: watch::Receiver<Option<Arc<MetadataFrame>>>,
    task: JoinHandle<()>,
    subscribers: usize,
}

struct Subscription {
    hub: Arc<Fanout>,
    key: String,
    updates: watch::Receiver<Option<Arc<MetadataFrame>>>,
}

struct Source {
    site_id: i64,
    generation: i64,
    generation_key: String,
    stream_key: String,
}

fn unavailable() -> ApiError {
    ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "Live metadata unavailable")
}

impl Metadata {
    pub fn new(client: Client, control: ConnectionManager) -> Self {
        Self {
            hub: Arc::new(Fanout {
                client,
                control,
                state: Mutex::new(FanoutState::default()),
            }),
            prefix: Arc::from(""),
        }
    }

    pub fn with_prefix(mut self, prefix: String) -> Self {
        self.prefix = Arc::from(prefix);
        self
    }

    pub(crate) async fn invalidate_site_generation(&self, site: &str) -> Result<(), ApiError> {
        redis::cmd("INCR")
            .arg(generation_key(&self.prefix, site))
            .query_async::<i64>(&mut self.hub.control.clone())
            .await
            .map_err(|_| unavailable())?;
        Ok(())
    }

    pub(crate) async fn invalidate_site_notification_cache(&self, sites: &[&str]) {
        let mut command = redis::cmd("DEL");
        for site in sites {
            command.arg(format!("site_notification_users:{site}"));
            command.arg(format!("site_notification_users_ready:{site}"));
        }
        if command
            .query_async::<i64>(&mut self.hub.control.clone())
            .await
            .is_err()
        {
            tracing::warn!("could not invalidate site notification cache");
        }
    }

    async fn subscribe(
        &self,
        site_id: i64,
        site: &str,
        stream_id: &str,
    ) -> Result<Subscription, ApiError> {
        let generation_key = generation_key(&self.prefix, site);
        let generation = site_generation(&mut self.hub.control.clone(), &generation_key)
            .await
            .map_err(|_| unavailable())?;
        let key = metadata_key_from_stream_id(&self.prefix, site, stream_id, generation);

        let mut state = self.hub.state.lock().map_err(|_| unavailable())?;
        if state.subscribers >= MAX_SUBSCRIBERS
            || (state.readers.len() >= MAX_CAMERAS && !state.readers.contains_key(&key))
        {
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "Live metadata capacity reached",
            ));
        }

        let reader = state.readers.entry(key.clone()).or_insert_with(|| {
            let (sender, updates) = watch::channel(None);
            let task = tokio::spawn(read_camera(
                self.hub.client.clone(),
                Source {
                    site_id,
                    generation,
                    generation_key,
                    stream_key: key.clone(),
                },
                sender,
            ));
            Reader {
                updates,
                task,
                subscribers: 0,
            }
        });
        // A new reader can fail before its first viewer has registered. Give
        // that viewer the closed receiver so its drop removes the reader.
        if reader.subscribers > 0 && reader.task.is_finished() {
            return Err(unavailable());
        }
        reader.subscribers += 1;
        let mut updates = reader.updates.clone();
        // A new viewer starts from the next update, not the buffered old frame.
        updates.borrow_and_update();
        state.subscribers += 1;
        Ok(Subscription {
            hub: self.hub.clone(),
            key,
            updates,
        })
    }
}

impl Subscription {
    async fn next(&mut self) -> Option<Arc<MetadataFrame>> {
        self.updates.changed().await.ok()?;
        // Discard a buffered update if its reader has already closed.
        self.updates.has_changed().ok()?;
        self.updates.borrow_and_update().clone()
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let Ok(mut state) = self.hub.state.lock() else {
            return;
        };
        let Some(reader) = state.readers.get_mut(&self.key) else {
            return;
        };
        reader.subscribers -= 1;
        if reader.subscribers == 0 {
            state
                .readers
                .remove(&self.key)
                .expect("subscriber has a registered reader")
                .task
                .abort();
        }
        state.subscribers -= 1;
    }
}

fn valid_stream_id(encoded: &str) -> bool {
    // The database's stream name is at most eighty UTF-8 characters.
    if encoded.is_empty() || encoded.len() > 428 {
        return false;
    }
    let Ok(decoded) = URL_SAFE.decode(encoded) else {
        return false;
    };
    let Ok(name) = std::str::from_utf8(&decoded) else {
        return false;
    };
    !name.trim().is_empty()
        && name.chars().count() <= 80
        && !name.contains('\0')
        && URL_SAFE.encode(decoded) == encoded
}

async fn connect(client: &Client) -> RedisResult<MultiplexedConnection> {
    client
        .get_multiplexed_async_connection_with_config(
            &AsyncConnectionConfig::new()
                .set_connection_timeout(CONNECTION_TIMEOUT)
                .set_response_timeout(CONNECTION_TIMEOUT),
        )
        .await
}

async fn site_generation(connection: &mut impl ConnectionLike, key: &str) -> RedisResult<i64> {
    let generation: Option<i64> = redis::cmd("GET").arg(key).query_async(connection).await?;
    Ok(generation.unwrap_or(0).max(0))
}

async fn read_camera(
    client: Client,
    source: Source,
    sender: watch::Sender<Option<Arc<MetadataFrame>>>,
) {
    // XREAD can block. Each active camera uses its own connection so it cannot
    // hold up authentication, generation lookups, or other cameras.
    let Ok(mut connection) = connect(&client).await else {
        return;
    };
    let mut last_id = String::from("$");
    let mut next_scope_check = Instant::now();
    loop {
        let update = match latest_update(&mut connection, &source.stream_key, &last_id).await {
            Ok(update) => update,
            Err(_) => return,
        };
        if Instant::now() >= next_scope_check {
            // Scope changes close the old generation. A new HTTP request is
            // authenticated before subscribing to the replacement generation.
            if !matches!(
                site_generation(&mut connection, &source.generation_key).await,
                Ok(generation) if generation == source.generation
            ) {
                return;
            }
            next_scope_check = Instant::now() + GENERATION_CHECK_INTERVAL;
        }
        if let Some(entry) = update {
            last_id.clone_from(&entry.id);
            if let Some(frame) = MetadataFrame::parse(entry, source.site_id) {
                sender.send_replace(Some(Arc::new(frame)));
            }
        }
    }
}

async fn latest_update(
    connection: &mut MultiplexedConnection,
    key: &str,
    last_id: &str,
) -> RedisResult<Option<StreamId>> {
    let mut reply: StreamReadReply = redis::cmd("XREAD")
        .arg("COUNT")
        .arg(READ_BATCH)
        .arg("BLOCK")
        .arg(REDIS_BLOCK_MS)
        .arg("STREAMS")
        .arg(key)
        .arg(last_id)
        .query_async(connection)
        .await?;
    let Some(mut events) = reply.keys.pop() else {
        return Ok(None);
    };
    if events.ids.len() >= READ_BATCH {
        let mut latest: StreamRangeReply = redis::cmd("XREVRANGE")
            .arg(key)
            .arg("+")
            .arg("-")
            .arg("COUNT")
            .arg(1)
            .query_async(connection)
            .await?;
        if let Some(entry) = latest.ids.pop() {
            return Ok(Some(entry));
        }
    }
    Ok(events.ids.pop())
}

#[derive(Serialize)]
struct Payload<'a> {
    has_warning: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    overlay: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
}

struct MetadataFrame {
    id: String,
    has_warning: bool,
    overlay: Option<Value>,
    sse: OnceLock<Bytes>,
    websocket: OnceLock<Utf8Bytes>,
}

impl MetadataFrame {
    fn parse(entry: StreamId, site_id: i64) -> Option<Self> {
        let (milliseconds, sequence) = entry.id.split_once('-')?;
        milliseconds.parse::<u64>().ok()?;
        sequence.parse::<u64>().ok()?;
        let warning = entry.get::<Vec<u8>>("has_warning")?;
        if !matches!(warning.as_slice(), b"0" | b"1") {
            return None;
        }
        let overlay = entry
            .get::<Vec<u8>>("overlay")
            .filter(|value| value.len() <= MAX_OVERLAY_BYTES)
            .and_then(|value| serde_json::from_slice::<Value>(&value).ok())
            .filter(Value::is_object);
        // Legacy warnings omit the overlay and use the authorised site key.
        if let Some(declared_site) = overlay.as_ref().and_then(|value| value.get("site_id"))
            && declared_site.as_i64() != Some(site_id)
        {
            return None;
        }
        Some(Self {
            id: entry.id,
            has_warning: warning == b"1",
            overlay,
            sse: OnceLock::new(),
            websocket: OnceLock::new(),
        })
    }

    fn payload(&self, with_id: bool) -> Payload<'_> {
        Payload {
            has_warning: self.has_warning,
            overlay: self.overlay.as_ref(),
            id: with_id.then_some(self.id.as_str()),
        }
    }

    fn sse(&self) -> Bytes {
        self.sse
            .get_or_init(|| {
                let mut bytes = Vec::new();
                write!(bytes, "id: {}\nevent: metadata\ndata: ", self.id)
                    .expect("writing to a Vec is infallible");
                serde_json::to_writer(&mut bytes, &self.payload(true))
                    .expect("metadata contains serialisable JSON values");
                bytes.extend_from_slice(b"\n\n");
                Bytes::from(bytes)
            })
            .clone()
    }

    fn websocket(&self) -> Utf8Bytes {
        self.websocket
            .get_or_init(|| {
                serde_json::to_string(&self.payload(false))
                    .expect("metadata contains serialisable JSON values")
                    .into()
            })
            .clone()
    }
}

struct Viewer {
    subscription: Subscription,
    auth: Authorizer,
    credentials: HeaderMap,
    site_id: i64,
    site: String,
    next_auth: Instant,
    shutdown: CancellationToken,
    playback_scope: Option<(Playback, sqlx::PgPool)>,
}

impl Viewer {
    async fn subscribe(
        state: AppState,
        site: String,
        stream_id: String,
        headers: &HeaderMap,
    ) -> Result<Self, ApiError> {
        let actor = state.auth.authorize(headers, Grant::StreamRead).await?;
        let (site_id, _) = actor
            .sites()
            .into_iter()
            .find(|(_, name)| name == &site)
            .ok_or_else(site_denied)?;
        if !valid_stream_id(&stream_id) {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "Invalid camera identifier",
            ));
        }
        let playback_scope = state
            .playback
            .filter(Playback::has_live_scope)
            .map(|playback| (playback, state.db));
        if let Some((playback, db)) = &playback_scope
            && !playback
                .allows_live_site(db, &actor, site_id, &site)
                .await?
        {
            return Err(site_denied());
        }
        let subscription = state.metadata.subscribe(site_id, &site, &stream_id).await?;
        let mut credentials = HeaderMap::new();
        for host in headers.get_all(header::HOST) {
            credentials.append(header::HOST, host.clone());
        }
        if let Some(bearer) = headers.get(header::AUTHORIZATION) {
            let mut bearer = bearer.clone();
            bearer.set_sensitive(true);
            credentials.insert(header::AUTHORIZATION, bearer);
        }
        Ok(Self {
            subscription,
            auth: state.auth,
            credentials,
            site_id,
            site,
            next_auth: Instant::now() + ACCESS_CHECK_INTERVAL,
            shutdown: state.shutdown,
            playback_scope,
        })
    }

    async fn refresh_access(&mut self) -> Result<(), ApiError> {
        let actor = self
            .auth
            .authorize(&self.credentials, Grant::StreamRead)
            .await?;
        let allowed = if let Some((playback, db)) = &self.playback_scope {
            playback
                .allows_live_site(db, &actor, self.site_id, &self.site)
                .await?
        } else {
            actor.allows_site(self.site_id, &self.site)
        };
        if !allowed {
            return Err(site_denied());
        }
        self.next_auth = Instant::now() + ACCESS_CHECK_INTERVAL;
        Ok(())
    }
}

fn site_denied() -> ApiError {
    ApiError::new(StatusCode::FORBIDDEN, "No access to this site")
}

struct SseViewer {
    viewer: Viewer,
    next_heartbeat: Instant,
    overlay: Option<OverlayLease>,
    ended: bool,
}

impl SseViewer {
    async fn update(mut self) -> Option<(Result<Bytes, Infallible>, Self)> {
        if self.ended {
            return None;
        }
        if let Some(ready) = self.overlay.as_mut().and_then(OverlayLease::take_pending) {
            self.next_heartbeat = Instant::now() + SSE_KEEPALIVE;
            return Some((Ok(ready), self));
        }
        loop {
            let overlay_at = self
                .overlay
                .as_ref()
                .map_or(self.next_heartbeat, OverlayLease::next_tick);
            tokio::select! {
                biased;
                () = self.viewer.shutdown.cancelled() => return None,
                () = sleep_until(self.viewer.next_auth) => self.viewer.refresh_access().await.ok()?,
                () = sleep_until(overlay_at), if self.overlay.is_some() => {
                    match self.overlay.as_mut().expect("overlay clock armed").tick().await {
                        Ok(Some(event)) => {
                            self.next_heartbeat = Instant::now() + SSE_KEEPALIVE;
                            return Some((Ok(event), self));
                        }
                        Ok(None) => (),
                        Err(error) => {
                            self.ended = true;
                            let payload = serde_json::json!({
                                "code": error.detail, "status": error.status().as_u16(),
                            });
                            return Some((Ok(Bytes::from(format!("event: overlay_error\ndata: {payload}\n\n"))), self));
                        }
                    }
                }
                () = sleep_until(self.next_heartbeat) => {
                    self.next_heartbeat = Instant::now() + SSE_KEEPALIVE;
                    return Some((Ok(Bytes::from_static(b": keepalive\n\n")), self));
                }
                frame = self.viewer.subscription.next() => {
                    self.next_heartbeat = Instant::now() + SSE_KEEPALIVE;
                    return Some((Ok(frame?.sse()), self));
                }
            }
        }
    }
}

#[derive(Deserialize)]
struct OverlayQuery {
    overlay: Option<String>,
    language: Option<String>,
}

async fn metadata_stream(
    State(state): State<AppState>,
    Path((site, stream_id)): Path<(String, String)>,
    Query(query): Query<OverlayQuery>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let viewer =
        Viewer::subscribe(state.clone(), site.clone(), stream_id.clone(), &headers).await?;
    let overlay = if backend_mode(query.overlay.as_deref()) {
        let playback = state.playback.as_ref().ok_or_else(|| {
            ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "playback_unavailable")
        })?;
        Some(
            playback
                .metadata_overlay(
                    &state.db,
                    state.media_auth.playback_settings(),
                    viewer.site_id,
                    &site,
                    &stream_id,
                    query.language.as_deref(),
                )
                .await?,
        )
    } else {
        None
    };
    let viewer = SseViewer {
        viewer,
        next_heartbeat: Instant::now() + SSE_KEEPALIVE,
        overlay,
        ended: false,
    };
    let initial = stream::once(async {
        Ok::<_, Infallible>(Bytes::from_static(b"retry: 15000\n: connected\n\n"))
    });
    let frames = stream::unfold(viewer, SseViewer::update);
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(header::CACHE_CONTROL, "private, no-store")
        .header(header::PRAGMA, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(initial.chain(frames)))
        .expect("static SSE headers are valid"))
}

async fn metadata_websocket(
    State(state): State<AppState>,
    Path((site, stream_id)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Result<Response, ApiError> {
    let credentials = websocket_credentials(&headers, query.as_deref())?;
    let lifetime = state.websocket_lifetime;
    let viewer = Viewer::subscribe(state, site, stream_id, &credentials).await?;
    let mut response = upgrade
        .read_buffer_size(MAX_WEBSOCKET_COMMAND_BYTES)
        .write_buffer_size(MAX_WEBSOCKET_COMMAND_BYTES)
        .max_write_buffer_size(MAX_OVERLAY_BYTES + 8192)
        .max_frame_size(MAX_WEBSOCKET_COMMAND_BYTES)
        .max_message_size(MAX_WEBSOCKET_COMMAND_BYTES)
        .on_upgrade(move |socket| serve_websocket(socket, viewer, lifetime));
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    Ok(response)
}

fn is_application_ping(text: &str) -> bool {
    #[derive(serde::Deserialize)]
    struct Command {
        action: Option<String>,
        #[serde(rename = "type")]
        kind: Option<String>,
    }
    serde_json::from_str::<Command>(text).is_ok_and(|command| {
        command
            .action
            .filter(|action| !action.is_empty())
            .or(command.kind)
            .as_deref()
            == Some("ping")
    })
}

async fn serve_websocket(mut socket: WebSocket, mut viewer: Viewer, lifetime: Duration) {
    let expires_at = Instant::now() + lifetime;
    let (code, reason) = loop {
        let outgoing = tokio::select! {
            biased;
            () = viewer.shutdown.cancelled() => break (close_code::AWAY, "Server shutting down"),
            () = sleep_until(expires_at) => {
                let message = serde_json::json!({
                    "status": "closing",
                    "reason": "session_timeout",
                    "message": format!("WebSocket session reached {}-second limit.", lifetime.as_secs_f64()),
                });
                let _ = timeout(WEBSOCKET_SEND_TIMEOUT, socket.send(Message::Text(message.to_string().into()))).await;
                break (close_code::NORMAL, "Session timeout");
            }
            () = sleep_until(viewer.next_auth) => {
                if let Err(error) = viewer.refresh_access().await {
                    let code = match error.status() {
                        StatusCode::FORBIDDEN => 4003,
                        status if status.is_server_error() => close_code::ERROR,
                        _ => close_code::POLICY,
                    };
                    break (code, "Metadata access expired");
                }
                continue;
            }
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Text(text))) if is_application_ping(&text) => {
                    Message::Text(Utf8Bytes::from_static(r#"{"action":"pong"}"#))
                }
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                _ => continue,
            },
            update = viewer.subscription.next() => match update {
                Some(frame) => Message::Text(frame.websocket()),
                None => break (close_code::AWAY, "Metadata stream ended"),
            },
        };
        tokio::select! {
            biased;
            () = viewer.shutdown.cancelled() => break (close_code::AWAY, "Server shutting down"),
            sent = timeout(WEBSOCKET_SEND_TIMEOUT, socket.send(outgoing)) => {
                if !matches!(sent, Ok(Ok(()))) {
                    return;
                }
            }
        }
    };
    let close = Message::Close(Some(CloseFrame {
        code,
        reason: Utf8Bytes::from_static(reason),
    }));
    let _ = timeout(WEBSOCKET_SEND_TIMEOUT, socket.send(close)).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_canonical_framework_keys() {
        assert!(valid_stream_id("Y2FtZXJhMQ=="));
        assert!(valid_stream_id(&URL_SAFE.encode("入口攝影機")));
        for invalid in [
            "",
            "Y2FtZXJhMQ",
            "Y2FtZXJhMQ===",
            "====",
            "Zg",
            "////",
            "AA==",
        ] {
            assert!(!valid_stream_id(invalid), "accepted {invalid}");
        }
        assert_eq!(
            generation_key("test:", "demo"),
            "test:stream_metadata_generation:ZGVtbw=="
        );
        assert_eq!(
            metadata_key_from_stream_id("test:", "demo", "Y2FtZXJhMQ==", 3),
            "test:stream_metadata:ZGVtbw==:g3|Y2FtZXJhMQ=="
        );
        assert_eq!(
            metadata_key_from_stream_id("", "demo", "Y2FtZXJhMQ==", 0),
            "stream_metadata:ZGVtbw==|Y2FtZXJhMQ=="
        );
    }

    #[test]
    fn preserves_both_payloads_and_rejects_mismatched_site() {
        let mut entry = StreamId {
            id: "1234-6".to_owned(),
            map: HashMap::from([
                (
                    "has_warning".to_owned(),
                    redis::Value::BulkString(b"1".to_vec()),
                ),
                (
                    "overlay".to_owned(),
                    redis::Value::BulkString(r#"{"site_id":3,"label":"工地"}"#.as_bytes().to_vec()),
                ),
            ]),
        };
        assert!(MetadataFrame::parse(entry.clone(), 4).is_none());
        let frame = MetadataFrame::parse(entry.clone(), 3).unwrap();
        assert!(frame.sse.get().is_none());
        assert_eq!(
            frame.websocket(),
            "{\"has_warning\":true,\"overlay\":{\"label\":\"工地\",\"site_id\":3}}"
        );
        assert_eq!(frame.websocket().as_ptr(), frame.websocket().as_ptr());
        assert!(frame.sse.get().is_none());
        assert_eq!(
            frame.sse(),
            "id: 1234-6\nevent: metadata\ndata: {\"has_warning\":true,\"overlay\":{\"label\":\"工地\",\"site_id\":3},\"id\":\"1234-6\"}\n\n"
        );
        assert_eq!(frame.sse().as_ptr(), frame.sse().as_ptr());
        entry.map.remove("overlay");
        let frame = MetadataFrame::parse(entry.clone(), 3).unwrap();
        assert_eq!(frame.websocket(), "{\"has_warning\":true}");
        assert_eq!(
            frame.sse(),
            "id: 1234-6\nevent: metadata\ndata: {\"has_warning\":true,\"id\":\"1234-6\"}\n\n"
        );
        entry.id = "123\ninjected: true-6".to_owned();
        assert!(MetadataFrame::parse(entry, 3).is_none());
    }

    #[test]
    fn recognises_ping_in_either_legacy_command_field() {
        assert!(is_application_ping(r#"{"action":"ping"}"#));
        assert!(is_application_ping(r#"{"type":"ping"}"#));
        assert!(is_application_ping(r#"{"action":"","type":"ping"}"#));
        assert!(!is_application_ping(r#"{"action":"other","type":"ping"}"#));
        assert!(!is_application_ping(r#"{"action":"pong"}"#));
        assert!(!is_application_ping("ping"));
    }
}

use std::collections::HashMap;
use std::convert::Infallible;
use std::error::Error;
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::Response;
use axum::routing::get;
use axum::{Json, Router};
use base64::{Engine, engine::general_purpose::URL_SAFE};
use clap::Parser;
use futures_util::{StreamExt, stream};
use redis::aio::MultiplexedConnection;
use redis::streams::{StreamId, StreamRangeReply, StreamReadReply};
use redis::{AsyncConnectionConfig, Client, ConnectionAddr, RedisResult};
use serde::Serialize;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};

const REDIS_READ_BATCH_SIZE: usize = 64;
const REDIS_BLOCK_MS: usize = 2000;
const REDIS_TIMEOUT: Duration = Duration::from_secs(5);
const SSE_KEEPALIVE: Duration = Duration::from_secs(15);
const SSE_CONNECTED: &[u8] = b"retry: 15000\n: connected\n\n";
const SSE_HEARTBEAT: &[u8] = b": keepalive\n\n";
const WEBSOCKET_CONTROL_MESSAGE_LIMIT: usize = 4096;

#[derive(Parser)]
#[command(about = "Loopback server for the isolated Redis metadata benchmark")]
struct Options {
    #[arg(long)]
    port: u16,
    #[arg(long)]
    redis_url: String,
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u16).range(1..=4096))]
    streams: u16,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..=64))]
    workers: u16,
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
    fn from_redis(entry: StreamId) -> Option<Self> {
        let warning = entry.get::<Vec<u8>>("has_warning")?;
        let overlay = entry
            .get::<Vec<u8>>("overlay")
            .and_then(|data| serde_json::from_slice::<Value>(&data).ok())
            .filter(Value::is_object);
        Some(Self {
            id: entry.id,
            has_warning: warning == b"1",
            overlay,
            sse: OnceLock::new(),
            websocket: OnceLock::new(),
        })
    }

    fn payload(&self, include_id: bool) -> Payload<'_> {
        Payload {
            has_warning: self.has_warning,
            overlay: self.overlay.as_ref(),
            id: include_id.then_some(self.id.as_str()),
        }
    }

    fn sse(&self) -> Bytes {
        self.sse
            .get_or_init(|| {
                let mut bytes = Vec::new();
                write!(bytes, "id: {}\nevent: metadata\ndata: ", self.id)
                    .expect("writing to a vector cannot fail");
                serde_json::to_writer(&mut bytes, &self.payload(true))
                    .expect("the public metadata payload is serialisable");
                bytes.extend_from_slice(b"\n\n");
                Bytes::from(bytes)
            })
            .clone()
    }

    fn websocket(&self) -> Utf8Bytes {
        self.websocket
            .get_or_init(|| {
                serde_json::to_string(&self.payload(false))
                    .expect("the public metadata payload is serialisable")
                    .into()
            })
            .clone()
    }
}

type LatestFrame = Option<Arc<MetadataFrame>>;

struct CameraReader {
    sender: watch::Sender<LatestFrame>,
    task: JoinHandle<()>,
    subscribers: usize,
}

struct Fanout {
    redis: Client,
    streams: u16,
    readers: Mutex<HashMap<u16, CameraReader>>,
}

impl Fanout {
    fn subscribe(self: &Arc<Self>, camera: u16) -> Subscription {
        let mut readers = self.readers.lock().expect("fanout lock poisoned");
        let state = readers.entry(camera).or_insert_with(|| {
            let (sender, _) = watch::channel(None);
            let task = tokio::spawn(read_camera(
                self.redis.clone(),
                camera_key(camera),
                sender.clone(),
            ));
            CameraReader {
                sender,
                task,
                subscribers: 0,
            }
        });
        state.subscribers += 1;
        Subscription {
            fanout: Arc::clone(self),
            camera,
            receiver: state.sender.subscribe(),
        }
    }
}

struct Subscription {
    fanout: Arc<Fanout>,
    camera: u16,
    receiver: watch::Receiver<LatestFrame>,
}

impl Subscription {
    async fn next(&mut self) -> LatestFrame {
        self.receiver.changed().await.ok()?;
        self.receiver.borrow_and_update().clone()
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let mut readers = self.fanout.readers.lock().expect("fanout lock poisoned");
        let state = readers.get_mut(&self.camera).expect("registered reader");
        state.subscribers -= 1;
        if state.subscribers == 0 {
            let state = readers.remove(&self.camera).expect("registered reader");
            state.task.abort();
        }
    }
}

fn camera_key(camera: u16) -> String {
    let site = URL_SAFE.encode("benchmark");
    let name = URL_SAFE.encode(format!("camera-{camera:03}"));
    format!("stream_metadata:{site}|{name}")
}

async fn connect(client: &Client) -> RedisResult<MultiplexedConnection> {
    client
        .get_multiplexed_async_connection_with_config(
            &AsyncConnectionConfig::new()
                .set_connection_timeout(REDIS_TIMEOUT)
                .set_response_timeout(REDIS_TIMEOUT),
        )
        .await
}

async fn read_camera(client: Client, key: String, updates: watch::Sender<LatestFrame>) {
    let mut last_id = String::from("$");
    loop {
        match connect(&client).await {
            Ok(mut connection) => loop {
                match read_update(&mut connection, &key, &last_id).await {
                    Ok(Some(entry)) => {
                        last_id.clone_from(&entry.id);
                        if let Some(frame) = MetadataFrame::from_redis(entry) {
                            updates.send_replace(Some(Arc::new(frame)));
                        }
                        tokio::task::yield_now().await;
                    }
                    Ok(None) => sleep(Duration::from_millis(50)).await,
                    Err(error) => {
                        eprintln!("Redis metadata read failed for {key}: {error}");
                        break;
                    }
                }
            },
            Err(error) => eprintln!("Redis metadata connection failed for {key}: {error}"),
        }
        sleep(Duration::from_secs(1)).await;
    }
}

async fn read_update(
    connection: &mut MultiplexedConnection,
    key: &str,
    last_id: &str,
) -> RedisResult<Option<StreamId>> {
    let mut result: StreamReadReply = redis::cmd("XREAD")
        .arg("COUNT")
        .arg(REDIS_READ_BATCH_SIZE)
        .arg("BLOCK")
        .arg(REDIS_BLOCK_MS)
        .arg("STREAMS")
        .arg(key)
        .arg(last_id)
        .query_async(connection)
        .await?;
    let Some(mut stream) = result.keys.pop() else {
        return Ok(None);
    };
    if stream.ids.len() >= REDIS_READ_BATCH_SIZE {
        let mut newest: StreamRangeReply = redis::cmd("XREVRANGE")
            .arg(key)
            .arg("+")
            .arg("-")
            .arg("COUNT")
            .arg(1)
            .query_async(connection)
            .await?;
        if let Some(entry) = newest.ids.pop() {
            return Ok(Some(entry));
        }
    }
    Ok(stream.ids.pop())
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    readers: usize,
    subscribers: usize,
}

async fn health(State(fanout): State<Arc<Fanout>>) -> Json<Health> {
    let readers = fanout.readers.lock().expect("fanout lock poisoned");
    Json(Health {
        status: "ok",
        readers: readers.len(),
        subscribers: readers.values().map(|reader| reader.subscribers).sum(),
    })
}

async fn sse(
    State(fanout): State<Arc<Fanout>>,
    Path(camera): Path<u16>,
) -> Result<Response, StatusCode> {
    if camera >= fanout.streams {
        return Err(StatusCode::NOT_FOUND);
    }
    let subscription = fanout.subscribe(camera);
    let frames = stream::unfold(subscription, |mut subscription| async {
        let bytes = match timeout(SSE_KEEPALIVE, subscription.next()).await {
            Ok(Some(frame)) => frame.sse(),
            Ok(None) => return None,
            Err(_) => Bytes::from_static(SSE_HEARTBEAT),
        };
        Some((Ok::<_, Infallible>(bytes), subscription))
    });
    let initial = stream::once(async { Ok::<_, Infallible>(Bytes::from_static(SSE_CONNECTED)) });
    let response = Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store, no-cache, must-revalidate")
        .header(header::PRAGMA, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(initial.chain(frames)))
        .expect("static SSE headers are valid");
    Ok(response)
}

async fn websocket(
    State(fanout): State<Arc<Fanout>>,
    Path(camera): Path<u16>,
    upgrade: WebSocketUpgrade,
) -> Result<Response, StatusCode> {
    if camera >= fanout.streams {
        return Err(StatusCode::NOT_FOUND);
    }
    Ok(upgrade
        .read_buffer_size(WEBSOCKET_CONTROL_MESSAGE_LIMIT)
        .max_message_size(WEBSOCKET_CONTROL_MESSAGE_LIMIT)
        .max_frame_size(WEBSOCKET_CONTROL_MESSAGE_LIMIT)
        .on_upgrade(move |socket| serve_websocket(socket, fanout.subscribe(camera))))
}

async fn serve_websocket(mut socket: WebSocket, mut subscription: Subscription) {
    loop {
        let outgoing = tokio::select! {
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Text(text))) => {
                    let value = serde_json::from_str::<Value>(text.as_str()).ok();
                    if value.as_ref().and_then(|data| data["action"].as_str()) != Some("ping") {
                        continue;
                    }
                    Message::Text(Utf8Bytes::from_static(r#"{"action":"pong"}"#))
                }
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                Some(Ok(_)) => continue,
            },
            update = subscription.next() => match update {
                Some(frame) => Message::Text(frame.websocket()),
                None => return,
            },
        };
        if !matches!(
            timeout(REDIS_TIMEOUT, socket.send(outgoing)).await,
            Ok(Ok(()))
        ) {
            return;
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    let workers = usize::from(options.workers);
    let mut runtime = if workers == 1 {
        tokio::runtime::Builder::new_current_thread()
    } else {
        let mut runtime = tokio::runtime::Builder::new_multi_thread();
        runtime.worker_threads(workers);
        runtime
    };
    runtime.enable_all().build()?.block_on(serve(options))
}

async fn serve(options: Options) -> Result<(), Box<dyn Error>> {
    let redis = Client::open(options.redis_url)?;
    // This stand-alone fixture has no authentication middleware.
    if !matches!(&redis.get_connection_info().addr, ConnectionAddr::Tcp(host, _)
        if host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback()))
    {
        return Err("the benchmark Redis URL must use a loopback IP address".into());
    }
    let mut connection = connect(&redis).await?;
    let _: String = redis::cmd("PING").query_async(&mut connection).await?;
    drop(connection);

    let fanout = Arc::new(Fanout {
        redis,
        streams: options.streams,
        readers: Mutex::default(),
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/sse/{camera}", get(sse))
        .route("/ws/{camera}", get(websocket))
        .with_state(fanout);
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, options.port)).await?;
    println!("Listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use redis::Value::BulkString;
    use serde_json::json;

    fn fixture(id: &str, overlay: &[u8]) -> MetadataFrame {
        MetadataFrame::from_redis(StreamId {
            id: id.to_owned(),
            map: HashMap::from([
                ("has_warning".to_owned(), BulkString(b"1".to_vec())),
                ("overlay".to_owned(), BulkString(overlay.to_vec())),
            ]),
        })
        .expect("fixture contains the required warning field")
    }

    #[test]
    fn serialises_the_public_frame_payload_for_each_protocol() {
        let overlay = json!({"message": "安全帽", "sequence": 123});
        let frame = fixture("123-0", overlay.to_string().as_bytes());
        let sse = frame.sse();
        let sse_data = std::str::from_utf8(&sse)
            .unwrap()
            .strip_prefix("id: 123-0\nevent: metadata\ndata: ")
            .unwrap()
            .strip_suffix("\n\n")
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(sse_data).unwrap(),
            json!({"has_warning": true, "overlay": overlay, "id": "123-0"}),
        );
        assert_eq!(
            serde_json::from_str::<Value>(frame.websocket().as_str()).unwrap(),
            json!({"has_warning": true, "overlay": overlay}),
        );
        assert_eq!(sse.as_ptr(), frame.sse().as_ptr());
        assert_eq!(
            frame.websocket().as_str().as_ptr(),
            frame.websocket().as_str().as_ptr(),
        );
    }

    #[test]
    fn omits_invalid_and_non_object_overlays() {
        for overlay in [b"[1, 2]".as_slice(), b"invalid", b"\xff"] {
            let frame = fixture("123-0", overlay);
            assert_eq!(frame.websocket().as_str(), r#"{"has_warning":true}"#);
        }
    }

    #[tokio::test]
    async fn shares_one_reader_and_coalesces_pending_frames() {
        let fanout = Arc::new(Fanout {
            redis: Client::open("redis://127.0.0.1:1/").unwrap(),
            streams: 32,
            readers: Mutex::default(),
        });
        let mut viewer = fanout.subscribe(1);
        let other_viewer = fanout.subscribe(1);
        let sender = {
            let readers = fanout.readers.lock().unwrap();
            assert_eq!(readers.len(), 1);
            assert_eq!(readers[&1].subscribers, 2);
            readers[&1].sender.clone()
        };
        sender.send_replace(Some(Arc::new(fixture("123-0", b"{}"))));
        sender.send_replace(Some(Arc::new(fixture("124-0", b"{}"))));
        assert_eq!(viewer.next().await.unwrap().id, "124-0");
        drop(viewer);
        assert_eq!(fanout.readers.lock().unwrap()[&1].subscribers, 1);
        drop(other_viewer);
        assert!(fanout.readers.lock().unwrap().is_empty());
    }
}

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{OwnedSemaphorePermit, watch};
use tokio::time::{sleep, timeout};

use super::runtime::{Fault, Inference, UPLOAD_TIMEOUT};
use crate::auth::{Actor, Grant, websocket_credentials};
use crate::{ApiError, AppState};

const MODEL_CONFIG_LIMIT: usize = 4096;
const REAUTHORIZE_INTERVAL: Duration = Duration::from_secs(5);
const SEND_TIMEOUT: Duration = Duration::from_secs(10);
const CLOSE_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Deserialize)]
struct ModelConfig {
    model_key: String,
}

struct ModelSelection {
    key: Option<String>,
    version: Option<String>,
}

fn valid_model_field(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 256
}

fn model_header(headers: &HeaderMap, name: &'static str) -> Result<Option<String>, ApiError> {
    let mut values = headers.get_all(name).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .ok()
        .filter(|value| valid_model_field(value))
        .filter(|_| values.next().is_none())
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "Invalid model header"))?;
    Ok(Some(value.to_owned()))
}

fn model_from_message(message: &str) -> Option<String> {
    if message.len() > MODEL_CONFIG_LIMIT {
        return None;
    }
    let config = serde_json::from_str::<ModelConfig>(message).ok()?;
    valid_model_field(&config.model_key).then_some(config.model_key)
}

pub(super) async fn upgrade(
    State(state): State<AppState>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    websocket: WebSocketUpgrade,
) -> Result<Response, Fault> {
    let inference = state.inference.clone().ok_or_else(Fault::unavailable)?;
    let credentials = websocket_credentials(&headers, query.as_deref())?;
    let actor = state
        .auth
        .authorize(&credentials, Grant::ImageDetection)
        .await?;
    let selection = ModelSelection {
        key: model_header(&headers, "x-model-key")?,
        version: model_header(&headers, "x-model-version")?,
    };
    // A connected client occupies a socket slot, not an inference work slot.
    let slot = inference
        .sessions
        .clone()
        .try_acquire_owned()
        .map_err(|_| Fault::Code(StatusCode::SERVICE_UNAVAILABLE, "WEBSOCKET_BUSY"))?;
    let limit = inference.max_upload_bytes.max(MODEL_CONFIG_LIMIT);
    Ok(websocket
        .read_buffer_size(MODEL_CONFIG_LIMIT)
        .max_frame_size(limit)
        .max_message_size(limit)
        .on_upgrade(move |socket| {
            session(
                socket,
                state,
                inference,
                selection,
                credentials,
                actor,
                slot,
            )
        }))
}

async fn reauthorize(
    state: &AppState,
    inference: &Inference,
    credentials: &HeaderMap,
    model: watch::Receiver<Option<(String, Option<String>)>>,
) -> ApiError {
    loop {
        sleep(REAUTHORIZE_INTERVAL).await;
        let actor = match state
            .auth
            .authorize(credentials, Grant::ImageDetection)
            .await
        {
            Ok(actor) => actor,
            Err(error) => return error,
        };
        let selected = model.borrow().clone();
        if let Some((key, version)) = selected
            && let Err(error) = inference
                .select_model(&state.db, &key, version.as_deref(), &actor)
                .await
        {
            return ApiError::new(
                fault_details(&error).0,
                "Model unavailable or access denied",
            );
        }
    }
}

async fn session(
    mut socket: WebSocket,
    state: AppState,
    inference: Arc<Inference>,
    selection: ModelSelection,
    credentials: HeaderMap,
    actor: Actor,
    _slot: OwnedSemaphorePermit,
) {
    let (model_tx, model_rx) = watch::channel(None);
    let close = tokio::select! {
        biased;
        () = state.shutdown.cancelled() => Some((1001, "Service shutting down")),
        () = sleep(state.websocket_lifetime) => Some((1000, "Session timeout")),
        error = reauthorize(&state, &inference, &credentials, model_rx) => Some(if error.status().is_server_error() {
            (1011, "Authentication service unavailable")
        } else {
            (1008, "Invalid token or deployment")
        }),
        close = serve(&mut socket, &state.db, &inference, selection, &actor, model_tx) => close,
    };
    if let Some((code, reason)) = close {
        let close = Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        }));
        let _ = timeout(CLOSE_TIMEOUT, socket.send(close)).await;
    }
}

async fn serve(
    socket: &mut WebSocket,
    db: &sqlx::PgPool,
    inference: &Inference,
    selection: ModelSelection,
    actor: &Actor,
    model_tx: watch::Sender<Option<(String, Option<String>)>>,
) -> Option<(u16, &'static str)> {
    let key = if let Some(key) = selection.key {
        key
    } else {
        let message = timeout(UPLOAD_TIMEOUT, receive(socket)).await;
        let key = match message {
            Ok(Some(Message::Text(message))) => model_from_message(message.as_str()),
            Ok(None) => return None,
            _ => None,
        };
        let Some(key) = key else {
            return Some((1008, "Missing or invalid model configuration"));
        };
        key
    };
    let mut pending = None;
    let model = match inference
        .select_model(db, &key, selection.version.as_deref(), actor)
        .await
    {
        Ok((model, _)) => model,
        Err(_) => return Some((1008, "Model unavailable or access denied")),
    };
    model_tx.send_replace(Some((key.clone(), selection.version)));
    if while_connected(socket, &mut pending, inference.prepare(model))
        .await?
        .is_err()
    {
        return Some((1011, "Model unavailable or access denied"));
    }
    send_json(
        socket,
        json!({
            "status": "ready",
            "model": key,
            "message": "Model loaded successfully, ready to process images",
        }),
    )
    .await?;
    loop {
        let message = match pending.take() {
            Some(message) => message,
            None => receive(socket).await?,
        };
        let Message::Binary(image) = message else {
            return Some((1003, "Send a binary image"));
        };
        let result = if image.len() > inference.max_upload_bytes {
            Err(Fault::Code(
                StatusCode::PAYLOAD_TOO_LARGE,
                "IMAGE_TOO_LARGE",
            ))
        } else {
            while_connected(socket, &mut pending, inference.detect(model, image, None)).await?
        };
        match result {
            Ok(detection) => send_json(socket, detection.rows).await?,
            Err(error) => {
                let (status, code) = fault_details(&error);
                send_json(
                    socket,
                    json!({"status": "error", "code": code, "http_status": status.as_u16()}),
                )
                .await?;
                return Some((
                    match status {
                        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => 1003,
                        StatusCode::PAYLOAD_TOO_LARGE => 1009,
                        _ => 1013,
                    },
                    code,
                ));
            }
        }
    }
}

fn fault_details(fault: &Fault) -> (StatusCode, &'static str) {
    match fault {
        Fault::Code(status, code) => (*status, code),
        Fault::Api(error) => (error.status(), "DETECTION_FAILED"),
    }
}

async fn receive(socket: &mut WebSocket) -> Option<Message> {
    while let Some(Ok(message)) = socket.recv().await {
        match message {
            Message::Text(_) | Message::Binary(_) => return Some(message),
            Message::Close(_) => return None,
            Message::Ping(_) | Message::Pong(_) => (),
        }
    }
    None
}

async fn while_connected<T>(
    socket: &mut WebSocket,
    pending: &mut Option<Message>,
    work: impl Future<Output = T>,
) -> Option<T> {
    // One pipelined message is enough to detect ordinary disconnects while
    // working. Once buffered, stop reading to bound the memory per socket.
    tokio::pin!(work);
    if pending.is_none() {
        tokio::select! {
            result = &mut work => return Some(result),
            received = receive(socket) => *pending = Some(received?),
        }
    }
    Some(work.await)
}

async fn send_json(socket: &mut WebSocket, payload: impl serde::Serialize) -> Option<()> {
    let message = serde_json::to_string(&payload).ok()?;
    timeout(SEND_TIMEOUT, socket.send(Message::Text(message.into())))
        .await
        .ok()?
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_models_using_the_legacy_header_and_text_formats() {
        let mut headers = HeaderMap::new();
        assert!(model_header(&headers, "x-model-key").unwrap().is_none());
        headers.insert("x-model-key", "yolo26n".parse().unwrap());
        assert_eq!(
            model_header(&headers, "x-model-key").unwrap().as_deref(),
            Some("yolo26n")
        );
        headers.append("x-model-key", "another".parse().unwrap());
        assert!(model_header(&headers, "x-model-key").is_err());
        assert_eq!(
            model_from_message(r#"{"model_key":"yolo26n"}"#).as_deref(),
            Some("yolo26n")
        );
        assert!(model_from_message(r#"{"model_key":" "}"#).is_none());
        assert!(model_from_message(r#"{"model_key":1}"#).is_none());
        assert!(model_from_message(&" ".repeat(MODEL_CONFIG_LIMIT + 1)).is_none());
    }

    #[test]
    fn preserves_detection_error_codes() {
        assert_eq!(
            fault_details(&Fault::Code(StatusCode::BAD_REQUEST, "INVALID_IMAGE")),
            (StatusCode::BAD_REQUEST, "INVALID_IMAGE")
        );
        assert_eq!(
            fault_details(&Fault::Api(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "busy"
            ))),
            (StatusCode::SERVICE_UNAVAILABLE, "DETECTION_FAILED")
        );
    }
}

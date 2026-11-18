use crate::{
    config::Delivery,
    outbox::{Outbox, Pending},
};
use anyhow::{Context, Result, ensure};
use reqwest::{Client, Response, StatusCode, multipart};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

struct Token {
    value: String,
    expires: Instant,
}
#[derive(Clone)]
pub struct Transport {
    client: Client,
    settings: Delivery,
    token: Arc<Mutex<Option<Token>>>,
}
impl Transport {
    pub fn new(settings: Delivery) -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(15))
                .connect_timeout(Duration::from_secs(5))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            settings,
            token: Arc::new(Mutex::new(None)),
        })
    }
    async fn token(&self, force: bool) -> Result<Option<String>> {
        let Some(url) = &self.settings.token_url else {
            return Ok(None);
        };
        let mut token = self.token.lock().await;
        if !force {
            if let Some(token) = token.as_ref() {
                if token.expires > Instant::now() {
                    return Ok(Some(token.value.clone()));
                }
            }
        }
        let id = std::env::var(
            self.settings
                .client_id_env
                .as_ref()
                .context("Missing OAuth client ID environment name")?,
        )?;
        let secret = std::env::var(
            self.settings
                .client_secret_env
                .as_ref()
                .context("Missing OAuth secret environment name")?,
        )?;
        let response = self
            .client
            .post(url)
            .form(&[
                ("grant_type", "client_credentials"),
                ("client_id", &id),
                ("client_secret", &secret),
            ])
            .send()
            .await?;
        ensure!(response.status().is_success(), "OAuth token rejected");
        let body: Value = response.json().await?;
        let value = body["access_token"]
            .as_str()
            .context("OAuth token missing")?
            .to_string();
        let seconds = body["expires_in"]
            .as_u64()
            .unwrap_or(60)
            .saturating_sub(10)
            .clamp(1, 86400);
        *token = Some(Token {
            value: value.clone(),
            expires: Instant::now() + Duration::from_secs(seconds),
        });
        Ok(Some(value))
    }
    async fn upload(&self, pending: &Pending) -> Result<Response> {
        for attempt in 0..2 {
            let token = self.token(attempt > 0).await?;
            let e = &pending.metadata;
            let fields = HashMap::from([
                ("event_id", e.event_id.clone()),
                ("site_id", e.site_id.to_string()),
                ("site", e.site.clone()),
                ("stream_name", e.stream_name.clone()),
                ("detection_time", e.detection_time.clone()),
                ("model_id", e.model_id.clone()),
                ("model_version", e.model_version.clone()),
                ("warnings_json", serde_json::to_string(&e.warnings)?),
                ("detections_json", serde_json::to_string(&e.detections)?),
                ("cone_polygon_json", serde_json::to_string(&e.cone_polygon)?),
                ("pole_polygon_json", serde_json::to_string(&e.pole_polygon)?),
            ]);
            let mut form = multipart::Form::new().part(
                "image",
                multipart::Part::bytes(pending.image.clone())
                    .file_name("violation.jpg")
                    .mime_str("image/jpeg")?,
            );
            for (key, value) in fields {
                form = form.text(key, value)
            }
            let mut request = self
                .client
                .post(
                    self.settings
                        .upload_url
                        .as_ref()
                        .context("Missing upload endpoint")?,
                )
                .header("Idempotency-Key", &e.event_id)
                .multipart(form);
            if let Some(token) = token {
                request = request.bearer_auth(token)
            }
            let response = request.send().await?;
            if response.status() != StatusCode::UNAUTHORIZED || attempt == 1 {
                return Ok(response);
            }
        }
        unreachable!()
    }
    async fn notify(&self, pending: &Pending, violation_id: &str) -> Result<Response> {
        for attempt in 0..2 {
            let token = self.token(attempt > 0).await?;
            let e = &pending.metadata;
            let id = violation_id
                .parse::<i64>()
                .context("Invalid violation identifier")?;
            let body = json!({"site":e.site,"site_id":e.site_id,"stream_name":e.stream_name,
                "body":e.warnings,"image_path":null,"violation_id":id,"type":"violation",
                "title":"Safety violation alert","deep_link":format!("/violations?violation_id={id}"),
                "metadata":{"site":e.site,"stream_name":e.stream_name,"violation_id":id}});
            let mut request = self
                .client
                .post(
                    self.settings
                        .notification_url
                        .as_ref()
                        .context("Missing notification endpoint")?,
                )
                .header("Idempotency-Key", &e.event_id)
                .json(&body);
            if let Some(token) = token {
                request = request.bearer_auth(token)
            }
            let response = request.send().await?;
            if response.status() != StatusCode::UNAUTHORIZED || attempt == 1 {
                return Ok(response);
            }
        }
        unreachable!()
    }
    pub async fn deliver(&self, store: &Outbox, pending: &mut Pending) -> Result<()> {
        let status = if pending.violation_id.is_none() {
            match self.upload(pending).await {
                Ok(response) if response.status().is_success() => {
                    let body: Value = response.json().await?;
                    let id = body["violation_id"]
                        .as_i64()
                        .map(|id| id.to_string())
                        .or_else(|| body["violation_id"].as_str().map(str::to_owned))
                        .context("Upload receipt missing")?;
                    id.parse::<i64>()?;
                    store.uploaded(pending, &id).await?;
                    pending.violation_id = Some(id);
                    None
                }
                Ok(response) => Some(response.status()),
                Err(_) => Some(StatusCode::SERVICE_UNAVAILABLE),
            }
        } else {
            None
        };
        if let Some(status) = status {
            return store
                .retry(
                    pending,
                    chrono::Utc::now().timestamp(),
                    permanent(status),
                    "Evidence endpoint rejected or unavailable",
                )
                .await;
        }
        if self.settings.notification_url.is_some() {
            match self
                .notify(
                    pending,
                    pending.violation_id.as_deref().expect("uploaded receipt"),
                )
                .await
            {
                Ok(response) if response.status().is_success() => {
                    let body: Value = response.json().await?;
                    let outcome = body["outcome"].as_str().unwrap_or(
                        if body["success"].as_bool().unwrap_or(false) {
                            "completed"
                        } else {
                            "retryable"
                        },
                    );
                    match outcome {
                        "completed" | "skipped" => (),
                        "retryable" => {
                            return store
                                .retry(
                                    &pending,
                                    chrono::Utc::now().timestamp(),
                                    false,
                                    "Notification not completed",
                                )
                                .await;
                        }
                        "permanent_failure" => {
                            return store
                                .retry(
                                    &pending,
                                    chrono::Utc::now().timestamp(),
                                    true,
                                    "Notification permanently rejected",
                                )
                                .await;
                        }
                        _ => anyhow::bail!("Unknown notification outcome"),
                    }
                }
                Ok(response) => {
                    return store
                        .retry(
                            pending,
                            chrono::Utc::now().timestamp(),
                            permanent(response.status()),
                            "Notification endpoint rejected",
                        )
                        .await;
                }
                Err(_) => {
                    return store
                        .retry(
                            pending,
                            chrono::Utc::now().timestamp(),
                            false,
                            "Notification transport unavailable",
                        )
                        .await;
                }
            }
        }
        store.complete(pending).await
    }
}
fn permanent(status: StatusCode) -> bool {
    status.is_client_error() && !matches!(status.as_u16(), 401 | 408 | 425 | 429)
}
pub async fn dispatch(store: Outbox, settings: Delivery, stop: CancellationToken) -> Result<()> {
    let transport = Transport::new(settings)?;
    loop {
        if stop.is_cancelled() {
            break;
        }
        let pending = match store.claim(chrono::Utc::now().timestamp()).await {
            Ok(pending) => pending,
            Err(error) => {
                tracing::warn!(%error,"PostgreSQL outbox unavailable; dispatcher will retry");
                tokio::select! {_=stop.cancelled()=>break,_=tokio::time::sleep(Duration::from_secs(1))=>()}
                continue;
            }
        };
        if let Some(mut pending) = pending {
            if transport.deliver(&store, &mut pending).await.is_err() {
                if let Err(error) = store.recover_error(&pending).await {
                    tracing::warn!(%error,"Outbox lease recovery failed; the persisted lease remains recoverable");
                }
                tracing::warn!(event=%pending.metadata.event_id,"Delivery failed; persisted for retry");
            }
        } else {
            tokio::select! {_=stop.cancelled()=>break,_=tokio::time::sleep(Duration::from_millis(100))=>()}
        }
    }
    Ok(())
}

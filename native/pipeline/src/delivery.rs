use crate::{
    config::Delivery,
    outbox::{ClaimStage, Outbox, Pending},
};
use anyhow::{Context, Result, ensure};
use reqwest::{Client, Response, StatusCode, multipart};
use serde_json::{Value, json};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{
    sync::{Mutex, Notify},
    task::JoinSet,
    time::{Instant, timeout_at},
};
use tokio_util::sync::CancellationToken;

// Allow the notification API's 85 s dispatch deadline and OAuth while staying
// within the outbox's 120 s lease. Evidence keeps its independent 15 s timeout.
const NOTIFICATION_TIMEOUT: Duration = Duration::from_secs(100);

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
    async fn token(&self, rejected: Option<&str>) -> Result<Option<String>> {
        let Some(url) = &self.settings.token_url else {
            return Ok(None);
        };
        let mut token = self.token.lock().await;
        if let Some(token) = token.as_ref()
            && token.expires > Instant::now()
            && rejected != Some(token.value.as_str())
        {
            return Ok(Some(token.value.clone()));
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
        let mut token = self.token(None).await?;
        for attempt in 0..2 {
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
            if let Some(token) = &token {
                request = request.bearer_auth(token)
            }
            let response = request.send().await?;
            if response.status() != StatusCode::UNAUTHORIZED || attempt == 1 || token.is_none() {
                return Ok(response);
            }
            token = self.token(token.as_deref()).await?;
        }
        unreachable!()
    }
    async fn notify(&self, pending: &Pending, violation_id: &str) -> Result<Response> {
        let id = violation_id
            .parse::<i64>()
            .ok()
            .filter(|&id| id > 0)
            .context("Stored violation_id must be a positive integer")?;
        let e = &pending.metadata;
        let body = json!({"site":e.site,"site_id":e.site_id,"stream_name":e.stream_name,
            "body":e.warnings,"image_path":null,"violation_id":id,"type":"violation",
            "title":"Safety violation alert","deep_link":format!("/violations?violation_id={id}"),
            "metadata":{"site":e.site,"stream_name":e.stream_name,"violation_id":id}});
        let deadline = Instant::now() + NOTIFICATION_TIMEOUT;
        let mut token = timeout_at(deadline, self.token(None)).await??;
        for attempt in 0..2 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            ensure!(!remaining.is_zero(), "Notification deadline expired");
            let mut request = self
                .client
                .post(
                    self.settings
                        .notification_url
                        .as_ref()
                        .context("Missing notification endpoint")?,
                )
                .timeout(remaining)
                .header("Idempotency-Key", &e.event_id)
                .json(&body);
            if let Some(token) = &token {
                request = request.bearer_auth(token)
            }
            let response = request.send().await?;
            if response.status() != StatusCode::UNAUTHORIZED || attempt == 1 || token.is_none() {
                return Ok(response);
            }
            token = timeout_at(deadline, self.token(token.as_deref())).await??;
        }
        unreachable!()
    }
    async fn upload_receipt(&self, store: &Outbox, pending: &Pending) -> Result<Option<String>> {
        let (status, delay) = match self.upload(pending).await {
            Ok(response) if response.status().is_success() => {
                let body: Value = response.json().await?;
                let id = body["violation_id"]
                    .as_i64()
                    .filter(|&id| id > 0)
                    .context("Upload receipt requires a positive integer violation_id")?;
                return Ok(Some(id.to_string()));
            }
            Ok(response) => (response.status(), retry_after(&response)),
            Err(_) => (StatusCode::SERVICE_UNAVAILABLE, 0),
        };
        store
            .retry_with_delay(
                pending,
                chrono::Utc::now().timestamp(),
                permanent(status),
                "Evidence endpoint rejected or unavailable",
                delay,
            )
            .await?;
        Ok(None)
    }
    async fn finish_notification(&self, store: &Outbox, pending: &Pending) -> Result<()> {
        let receipt = pending
            .violation_id
            .as_deref()
            .context("Missing upload receipt")?;
        let (failed, reason, delay) = match self.notify(pending, receipt).await {
            Ok(response) if response.status().is_success() => {
                let delay = retry_after(&response);
                let body: Value = response.json().await?;
                let outcome = body["outcome"]
                    .as_str()
                    .context("Notification response requires an outcome")?;
                match outcome {
                    "completed" | "skipped" => return store.complete(pending).await,
                    "retryable" => (
                        false,
                        "Notification not completed",
                        delay.max(body["retry_after_seconds"].as_i64().unwrap_or_default()),
                    ),
                    "permanent_failure" => (true, "Notification permanently rejected", 0),
                    _ => anyhow::bail!("Unknown notification outcome"),
                }
            }
            Ok(response) => (
                permanent(response.status()),
                "Notification endpoint rejected",
                retry_after(&response),
            ),
            Err(_) => (false, "Notification transport unavailable", 0),
        };
        store
            .retry_with_delay(
                pending,
                chrono::Utc::now().timestamp(),
                failed,
                reason,
                delay.clamp(0, 86400),
            )
            .await
    }
    pub async fn deliver(&self, store: &Outbox, pending: &mut Pending) -> Result<()> {
        if pending.violation_id.is_none() {
            let Some(id) = self.upload_receipt(store, pending).await? else {
                return Ok(());
            };
            store.uploaded(pending, &id).await?;
            pending.violation_id = Some(id);
        }
        if self.settings.notification_url.is_some() {
            self.finish_notification(store, pending).await
        } else {
            store.complete(pending).await
        }
    }
}
fn retry_after(response: &Response) -> i64 {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            value.parse::<i64>().ok().or_else(|| {
                chrono::DateTime::parse_from_rfc2822(value)
                    .ok()
                    .map(|date| {
                        date.timestamp()
                            .saturating_sub(chrono::Utc::now().timestamp())
                    })
            })
        })
        .unwrap_or_default()
        .clamp(0, 86400)
}
fn permanent(status: StatusCode) -> bool {
    status.is_client_error() && !matches!(status.as_u16(), 401 | 408 | 425 | 429)
}
pub async fn dispatch(store: Outbox, settings: Delivery, stop: CancellationToken) -> Result<()> {
    settings.validate_workers()?;
    let evidence_workers = settings.evidence_workers;
    let notification_workers = settings.notification_workers;
    let send_notifications = settings.notification_url.is_some();
    let transport = Transport::new(settings)?;
    let handoff = Arc::new(Notify::new());
    if send_notifications {
        tokio::join!(
            dispatch_stage(
                &store,
                &transport,
                ClaimStage::Evidence,
                evidence_workers,
                &handoff,
                &stop
            ),
            dispatch_stage(
                &store,
                &transport,
                ClaimStage::Notification,
                notification_workers,
                &handoff,
                &stop
            ),
        );
    } else {
        dispatch_stage(
            &store,
            &transport,
            ClaimStage::Any,
            evidence_workers,
            &handoff,
            &stop,
        )
        .await;
    }
    Ok(())
}

async fn dispatch_stage(
    store: &Outbox,
    transport: &Transport,
    stage: ClaimStage,
    workers: usize,
    handoff: &Arc<Notify>,
    stop: &CancellationToken,
) {
    let mut tasks = JoinSet::new();
    let notification_stage = matches!(stage, ClaimStage::Notification);
    while !stop.is_cancelled() {
        while let Some(result) = tasks.try_join_next() {
            log_task_failure(result, stage);
        }
        if tasks.len() == workers {
            tokio::select! {
                _ = stop.cancelled() => break,
                Some(result) = tasks.join_next() => log_task_failure(result, stage),
            }
            continue;
        }
        let wait = match store
            .claim_stage(chrono::Utc::now().timestamp(), stage)
            .await
        {
            Ok(Some(pending)) => {
                let (store, transport, handoff) =
                    (store.clone(), transport.clone(), handoff.clone());
                tasks.spawn(deliver_claim(store, transport, stage, pending, handoff));
                continue;
            }
            Ok(None) => Duration::from_millis(100),
            Err(error) => {
                tracing::warn!(%error, ?stage, "PostgreSQL outbox unavailable; dispatcher will retry");
                tokio::select! {
                    _ = stop.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => (),
                }
                continue;
            }
        };
        tokio::select! {
            _ = stop.cancelled() => break,
            _ = handoff.notified(), if notification_stage => (),
            _ = tokio::time::sleep(wait) => (),
        }
    }
    while let Some(result) = tasks.join_next().await {
        log_task_failure(result, stage);
    }
}

async fn deliver_claim(
    store: Outbox,
    transport: Transport,
    stage: ClaimStage,
    mut pending: Pending,
    handoff: Arc<Notify>,
) {
    let result = match stage {
        ClaimStage::Any => transport.deliver(&store, &mut pending).await,
        ClaimStage::Evidence => handoff_evidence(&store, &transport, &pending, &handoff).await,
        ClaimStage::Notification => transport.finish_notification(&store, &pending).await,
    };
    if result.is_err() {
        if let Err(error) = store.recover_error(&pending).await {
            tracing::warn!(%error, "Outbox lease recovery failed; the persisted lease remains recoverable");
        }
        tracing::warn!(event=%pending.metadata.event_id, ?stage, "Delivery failed; persisted for retry");
    }
}

async fn handoff_evidence(
    store: &Outbox,
    transport: &Transport,
    pending: &Pending,
    handoff: &Notify,
) -> Result<()> {
    if let Some(id) = transport.upload_receipt(store, pending).await? {
        store
            .schedule_notification(pending, &id, chrono::Utc::now().timestamp())
            .await?;
        handoff.notify_one();
    }
    Ok(())
}

fn log_task_failure(result: Result<(), tokio::task::JoinError>, stage: ClaimStage) {
    if let Err(error) = result {
        tracing::warn!(%error, ?stage, "Outbox delivery task failed; the persisted lease remains recoverable");
    }
}

use std::{
    collections::{BTreeMap, HashSet},
    sync::OnceLock,
    time::{Duration, Instant},
};

use axum::{
    Extension, Json,
    extract::State,
    http::{HeaderMap, StatusCode},
};
use chrono::{DateTime, Utc};
use futures_util::{StreamExt, stream::FuturesUnordered};
use redis::aio::ConnectionManager;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sqlx::PgPool;
use uuid::Uuid;

use crate::{ApiError, AppState, auth::Grant, tenancy::database_unavailable};

use super::{Settings, fcm, inbox::Kind, temporary_failure, token_hash};

type Catalogue = BTreeMap<String, BTreeMap<String, String>>;

const TARGET_PAGE_SIZE: i64 = 256;
const SENDER_CONCURRENCY: usize = 8;
const CLAIM_TIMEOUT_SECONDS: u64 = 120;
const COMPLETED_TTL_SECONDS: u64 = 600;

pub(super) fn catalogue() -> &'static Catalogue {
    static CATALOGUE: OnceLock<Catalogue> = OnceLock::new();
    CATALOGUE.get_or_init(|| {
        serde_json::from_str(include_str!("languages.json")).expect("checked translations")
    })
}

#[derive(Deserialize)]
pub(super) struct Event {
    site: String,
    site_id: i32,
    stream_name: String,
    body: BTreeMap<String, Warning>,
    image_path: Option<String>,
    violation_id: i64,
    #[serde(rename = "type")]
    kind: Kind,
    title: String,
    deep_link: String,
    metadata: Map<String, Value>,
}

#[derive(Deserialize)]
struct Warning {
    count: u32,
}

impl Event {
    fn validate(&self) -> Result<(), ApiError> {
        let valid_text = |value: &str, max| {
            !value.trim().is_empty()
                && value.chars().count() <= max
                && !value.chars().any(char::is_control)
        };
        let english = &catalogue()["en-GB"];
        if self.site_id <= 0 || self.violation_id <= 0 {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "site_id and violation_id must be positive integers",
            ));
        }
        if !valid_text(&self.site, 100)
            || !valid_text(&self.stream_name, 80)
            || !valid_text(&self.title, 120)
            || !valid_text(&self.deep_link, 255)
            || !self.deep_link.starts_with('/')
            || self.deep_link.starts_with("//")
            || self.deep_link.contains('\\')
            || self.body.is_empty()
            || self.body.len() > 8
            || self.body.iter().any(|(key, warning)| {
                warning.count == 0
                    || !english
                        .get(key)
                        .is_some_and(|template| template.contains("{count}"))
            })
        {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Invalid notification event",
            ));
        }
        Ok(())
    }

    fn text(&self, lang: &str) -> (&'static str, String) {
        let translations = catalogue().get(lang).unwrap_or(&catalogue()["en-GB"]);
        let lines: Vec<_> = self
            .body
            .iter()
            .map(|(key, warning)| translations[key].replace("{count}", &warning.count.to_string()))
            .collect();
        (
            &translations["warning_notification"],
            format!("{} - {}\n{}", self.site, self.stream_name, lines.join("\n")),
        )
    }
}

struct Claim {
    redis: ConnectionManager,
    key: String,
    owner: String,
}

impl Claim {
    fn new(redis: ConnectionManager, event: &Event) -> Result<Self, ApiError> {
        let mut bytes = [0; 16];
        aws_lc_rs::rand::fill(&mut bytes).map_err(|_| temporary_failure())?;
        Ok(Self {
            redis,
            key: format!(
                "fcm_notification_dedupe:{}:{}",
                event.site_id, event.violation_id
            ),
            owner: format!("processing:{}", Uuid::from_bytes(bytes)),
        })
    }

    async fn acquire(&mut self) -> Result<Option<Value>, ApiError> {
        let result: Option<String> = redis::cmd("SET")
            .arg(&self.key)
            .arg(&self.owner)
            .arg("NX")
            .arg("EX")
            .arg(CLAIM_TIMEOUT_SECONDS)
            .query_async(&mut self.redis)
            .await
            .map_err(|_| temporary_failure())?;
        if result.is_some() {
            return Ok(None);
        }
        let value: Option<String> = redis::cmd("GET")
            .arg(&self.key)
            .query_async(&mut self.redis)
            .await
            .map_err(|_| temporary_failure())?;
        Ok(Some(if value.as_deref() == Some("done") {
            outcome("skipped", "Duplicate notification skipped.")
        } else {
            outcome(
                "retryable",
                "Notification delivery is in progress; retry later.",
            )
        }))
    }

    async fn confirm(&mut self) -> Result<(), ApiError> {
        let value: Option<String> = redis::cmd("GET")
            .arg(&self.key)
            .query_async(&mut self.redis)
            .await
            .map_err(|_| temporary_failure())?;
        if value.as_deref() != Some(&self.owner) {
            return Err(temporary_failure());
        }
        Ok(())
    }

    async fn finish(&mut self, done: bool) -> Result<(), ApiError> {
        let owned: bool = redis::Script::new(
            "
            if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
            if ARGV[2] == 'done' then
                redis.call('SET', KEYS[1], 'done', 'EX', ARGV[3])
            else
                redis.call('DEL', KEYS[1])
            end
            return 1
        ",
        )
        .key(&self.key)
        .arg(&self.owner)
        .arg(if done { "done" } else { "release" })
        .arg(COMPLETED_TTL_SECONDS)
        .invoke_async(&mut self.redis)
        .await
        .map_err(|_| temporary_failure())?;
        if !owned && done {
            return Err(temporary_failure());
        }
        Ok(())
    }
}

fn outcome(name: &str, message: &str) -> Value {
    json!({"success": matches!(name, "completed" | "skipped"), "outcome": name, "message": message})
}

pub(super) async fn send_site(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Json(event): Json<Event>,
) -> Result<Json<Value>, ApiError> {
    let actor = state
        .auth
        .authorize(&headers, Grant::NotificationDispatch)
        .await?;
    event.validate()?;
    if !actor.allows_site(i64::from(event.site_id), &event.site) {
        return Err(site_unavailable());
    }
    let known: bool =
        sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM sites WHERE id = $1 AND name = $2)")
            .bind(event.site_id)
            .bind(&event.site)
            .fetch_one(&state.db)
            .await
            .map_err(database_unavailable)?;
    if !known
        || !settings
            .0
            .scope
            .active(&state.db)
            .await
            .map_err(database_unavailable)?
    {
        return Err(site_unavailable());
    }
    let slot = settings
        .0
        .dispatch_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| temporary_failure())?;
    let mut claim = Claim::new(settings.0.redis.clone(), &event)?;
    if let Some(result) = claim.acquire().await? {
        return Ok(Json(result));
    }
    // Finish recording provider acknowledgements even if the HTTP caller
    // disconnects. The bounded permit stays with the delivery task.
    tokio::spawn(async move {
        let _slot = slot;
        let result = tokio::time::timeout(
            Duration::from_secs(85),
            deliver(&state.db, &settings, &mut claim, &event),
        )
        .await
        .unwrap_or_else(|_| {
            Ok(outcome(
                "retryable",
                "Notification delivery deadline reached.",
            ))
        });
        let done = result.as_ref().is_ok_and(|value| value["success"] == true);
        claim.finish(done).await?;
        result.map(Json)
    })
    .await
    .map_err(|_| temporary_failure())?
}

fn site_unavailable() -> ApiError {
    ApiError::new(StatusCode::FORBIDDEN, "Site notification unavailable")
}

const RECIPIENT_SCOPE: &str = "
    FROM site_notification_preferences p JOIN users u ON u.id = p.user_id
    JOIN sites s ON s.id = p.site_id
    WHERE p.site_id = $1 AND u.tenant_id = $2 AND s.name = $3
      AND p.is_enabled AND u.status = 'active' AND NOT u.deletion_pending
      AND (u.role = 'super_admin' OR EXISTS (
        SELECT 1 FROM user_sites us
        JOIN site_groups sg ON sg.group_id = u.group_id AND sg.site_id = us.site_id
        WHERE us.user_id = u.id AND us.site_id = p.site_id
      ))";

type Target = (i32, String, String, String, DateTime<Utc>, String);

#[derive(Default)]
struct DeliveryStats {
    delivered: usize,
    invalid: usize,
    permanent: usize,
    retryable: usize,
    retry_after_seconds: Option<u64>,
    failure_reasons: BTreeMap<&'static str, usize>,
}

impl DeliveryStats {
    fn observe(&mut self, result: fcm::SendResult) {
        let Err(failure) = result else {
            self.delivered += 1;
            return;
        };
        *self.failure_reasons.entry(failure.reason).or_default() += 1;
        match failure.kind {
            fcm::FailureKind::Registration => self.invalid += 1,
            fcm::FailureKind::Payload => self.permanent += 1,
            fcm::FailureKind::Configuration | fcm::FailureKind::Retryable => self.retryable += 1,
        }
        self.retry_after_seconds = self.retry_after_seconds.max(failure.retry_after_seconds);
    }

    fn event_response(self, notification_count: u64, more_pending: bool) -> Value {
        let outcome = if more_pending || self.retryable != 0 {
            "retryable"
        } else if self.permanent != 0 {
            "permanent_failure"
        } else {
            "completed"
        };
        let mut result = json!({"success": outcome == "completed", "outcome": outcome,
            "notification_count": notification_count, "success_count": self.delivered,
            "failure_count": self.retryable + self.permanent, "invalid_tokens": self.invalid});
        if !self.failure_reasons.is_empty() {
            result["failure_reasons"] = json!(self.failure_reasons);
        }
        if let Some(delay) = self.retry_after_seconds {
            result["retry_after_seconds"] = json!(delay);
        }
        result
    }
}

async fn send_target(
    db: &PgPool,
    settings: &Settings,
    sender: &fcm::Sender,
    target: &Target,
    event_digest: Option<&str>,
    message: impl FnOnce(&str) -> Value,
) -> Result<fcm::SendResult, ApiError> {
    let (_, hash, encrypted, _, revision, platform) = target;
    if let Some(failure) = sender.deferred_failure(hash, *revision) {
        return Ok(Err(failure));
    }
    let token = settings
        .0
        .cipher
        .decrypt(encrypted)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .filter(|token| token_hash(token) == *hash);
    let result = if let Some(token) = token {
        sender.send(&message(&token), platform).await
    } else {
        let failure = fcm::Failure::device_storage();
        sender.observe_failure(platform, failure);
        Err(failure)
    };
    record_result(db, target, event_digest, result).await?;
    if let Err(failure) = result {
        sender.defer_failure(hash, *revision, failure);
    }
    Ok(result)
}

async fn deliver(
    db: &PgPool,
    settings: &Settings,
    claim: &mut Claim,
    event: &Event,
) -> Result<Value, ApiError> {
    let sender = settings.0.fcm.as_ref().ok_or_else(temporary_failure)?;
    let key = format!("violation:{}", event.violation_id);
    let digest = token_hash(&claim.key);
    let tenant = settings.0.scope.tenant_id;
    let (_, text) = event.text("zh-TW");
    // Store the inbox independently of Firebase delivery. Subsequent attempts
    // reuse both the record key and any confirmed per-registration receipts.
    let records = format!(
        "WITH recipients AS (SELECT u.id {RECIPIENT_SCOPE})
         INSERT INTO notifications
           (user_id, event_key, type, title, body, deep_link, metadata, is_read, created_at)
         SELECT id, $4, $5, $6, $7, $8, $9, false, now() FROM recipients
         ON CONFLICT (user_id, event_key) DO NOTHING"
    );
    let n = sqlx::query(&records)
        .bind(event.site_id)
        .bind(tenant)
        .bind(&event.site)
        .bind(key)
        .bind(event.kind.as_str())
        .bind(&event.title)
        .bind(text)
        .bind(&event.deep_link)
        .bind(sqlx::types::Json(&event.metadata))
        .execute(db)
        .await
        .map_err(database_unavailable)?
        .rows_affected();

    let targets = format!(
        "WITH recipients AS (SELECT u.id {RECIPIENT_SCOPE})
         SELECT t.user_id, t.device_token_hash, t.device_token_encrypted, t.device_lang, t.last_seen_at, t.platform
         FROM recipients r JOIN fcm_device_tokens t ON t.user_id = r.id
         WHERE t.disabled_at IS NULL AND t.device_token_hash > $6 AND NOT EXISTS (
             SELECT 1 FROM notification_delivery_receipts d
             WHERE d.event_digest = $4 AND d.user_id = t.user_id AND d.token_hash = t.device_token_hash
               AND d.registration_at = t.last_seen_at AND d.expires_at > now()
         )
         ORDER BY t.device_token_hash LIMIT $5"
    );
    let data = json!({
        "navigate": "violation_list_page", "violation_id": event.violation_id.to_string(),
        "deep_link": &event.deep_link, "type": event.kind.as_str()
    });
    let image = event.image_path.as_deref().filter(|url| {
        url.len() <= 1024
            && url::Url::parse(url).is_ok_and(|url| {
                url.scheme() == "https"
                    && url.has_host()
                    && url.username().is_empty()
                    && url.password().is_none()
            })
    });
    let mut stats = DeliveryStats::default();
    let mut more_pending = false;
    let eligible_query = format!(
        "WITH recipients AS (SELECT u.id {RECIPIENT_SCOPE})
         SELECT t.device_token_hash
         FROM unnest($4::integer[], $5::text[], $6::timestamptz[]) AS selected(user_id, hash, seen_at)
         JOIN recipients r ON r.id = selected.user_id
         JOIN fcm_device_tokens t ON t.user_id = selected.user_id
              AND t.device_token_hash = selected.hash AND t.last_seen_at = selected.seen_at
         WHERE t.disabled_at IS NULL"
    );
    let deadline = Instant::now() + Duration::from_secs(35);
    let eligible_query = eligible_query.as_str();
    let digest = digest.as_str();
    let data = &data;
    let mut cursor = String::new();
    let mut translations = BTreeMap::new();
    'pages: loop {
        let page: Vec<Target> = sqlx::query_as(&targets)
            .bind(event.site_id)
            .bind(tenant)
            .bind(&event.site)
            .bind(digest)
            .bind(TARGET_PAGE_SIZE)
            .bind(&cursor)
            .fetch_all(db)
            .await
            .map_err(database_unavailable)?;
        let Some(last) = page.last() else {
            if cursor.is_empty() {
                return Ok(outcome("skipped", "No pending devices for this event."));
            }
            break;
        };
        cursor.clone_from(&last.1);
        let last_page = page.len() < TARGET_PAGE_SIZE as usize;
        // Reuse rendered text for each language across the whole dispatch.
        for (_, _, _, lang, _, _) in &page {
            if !translations.contains_key(lang) {
                translations.insert(lang.clone(), event.text(lang));
            }
        }
        for batch in page.chunks(SENDER_CONCURRENCY) {
            if Instant::now() >= deadline {
                more_pending = true;
                break 'pages;
            }
            claim.confirm().await?;
            let (mut users, mut hashes, mut registrations) = (Vec::new(), Vec::new(), Vec::new());
            for target in batch {
                users.push(target.0);
                hashes.push(target.1.as_str());
                registrations.push(target.4);
            }
            let eligible: Vec<String> = sqlx::query_scalar(eligible_query)
                .bind(event.site_id)
                .bind(tenant)
                .bind(&event.site)
                .bind(users)
                .bind(hashes)
                .bind(registrations)
                .fetch_all(db)
                .await
                .map_err(database_unavailable)?;
            let eligible: HashSet<_> = eligible.iter().map(String::as_str).collect();
            let mut results = FuturesUnordered::new();
            for target in batch {
                if eligible.contains(target.1.as_str()) {
                    let (title, body) = &translations[&target.3];
                    results.push(send_target(
                        db,
                        settings,
                        sender,
                        target,
                        Some(digest),
                        move |token| fcm::payload(token, title, body, data, image),
                    ));
                }
            }
            let mut error = None;
            while let Some(result) = results.next().await {
                match result {
                    Ok(result) => stats.observe(result),
                    Err(cause) => error = Some(cause),
                }
            }
            if let Some(error) = error {
                return Err(error);
            }
        }
        if last_page {
            break;
        }
    }
    Ok(stats.event_response(n, more_pending))
}

async fn record_result(
    db: &PgPool,
    target: &Target,
    digest: Option<&str>,
    result: fcm::SendResult,
) -> Result<(), ApiError> {
    match result {
        Ok(()) => {
            let update = sqlx::query(
                "UPDATE fcm_device_tokens SET last_success_at = now(), last_failure_at = NULL,
                     failure_reason = NULL, updated_at = now()
                 WHERE user_id = $1 AND device_token_hash = $2 AND last_seen_at = $3 AND disabled_at IS NULL"
            ).bind(target.0).bind(&target.1).bind(target.4);
            if let Some(digest) = digest {
                let mut tx = db.begin().await.map_err(database_unavailable)?;
                sqlx::query(
                    "INSERT INTO notification_delivery_receipts
                   (event_digest, user_id, token_hash, registration_at, delivered_at, expires_at)
                 VALUES ($1, $2, $3, $4, now(), now() + interval '7 days')
                 ON CONFLICT (event_digest, user_id, token_hash, registration_at) DO UPDATE
                   SET delivered_at = EXCLUDED.delivered_at, expires_at = EXCLUDED.expires_at
                   WHERE notification_delivery_receipts.expires_at <= now()",
                )
                .bind(digest)
                .bind(target.0)
                .bind(&target.1)
                .bind(target.4)
                .execute(&mut *tx)
                .await
                .map_err(database_unavailable)?;
                update
                    .execute(&mut *tx)
                    .await
                    .map_err(database_unavailable)?;
                tx.commit().await.map_err(database_unavailable)?;
            } else {
                update.execute(db).await.map_err(database_unavailable)?;
            }
        }
        Err(failure) => {
            sqlx::query(
                "UPDATE fcm_device_tokens SET
                   disabled_at = CASE WHEN $4 THEN now() ELSE disabled_at END,
                   last_failure_at = now(), failure_reason = $5, updated_at = now()
                 WHERE user_id = $1 AND device_token_hash = $2 AND last_seen_at = $3 AND disabled_at IS NULL"
            ).bind(target.0).bind(&target.1).bind(target.4)
             .bind(failure.kind == fcm::FailureKind::Registration).bind(failure.reason)
             .execute(db).await.map_err(database_unavailable)?;
        }
    }
    Ok(())
}

pub(super) async fn test_device(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let (_, owner) = super::account(&state, &settings, &headers).await?;
    let sender = settings.0.fcm.as_ref().ok_or_else(temporary_failure)?;
    let _slot = settings
        .0
        .dispatch_slots
        .try_acquire()
        .map_err(|_| temporary_failure())?;
    let mut redis = settings.0.redis.clone();
    let allowed: Option<String> = redis::cmd("SET")
        .arg(format!(
            "rust:fcm-test:{}:{}",
            settings.0.scope.deployment_id, owner.id
        ))
        .arg("1")
        .arg("NX")
        .arg("EX")
        .arg(30)
        .query_async(&mut redis)
        .await
        .map_err(|_| temporary_failure())?;
    if allowed.is_none() {
        return Err(ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "Wait before sending another test notification",
        ));
    }
    let targets: Vec<Target> = sqlx::query_as(
        "SELECT user_id, device_token_hash, device_token_encrypted, device_lang, last_seen_at, platform
         FROM fcm_device_tokens WHERE user_id = $1 AND disabled_at IS NULL ORDER BY last_seen_at DESC LIMIT 8",
    ).bind(owner.id).fetch_all(&state.db).await.map_err(database_unavailable)?;
    let data = json!({"type":"system","test":"true"});
    let mut work = FuturesUnordered::new();
    for target in &targets {
        work.push(send_target(
            &state.db,
            &settings,
            sender,
            target,
            None,
            |token| {
                fcm::payload(
                    token,
                    "Visionnaire test notification",
                    "This is a test notification.",
                    &data,
                    None,
                )
            },
        ));
    }
    let mut stats = DeliveryStats::default();
    while let Some(result) = work.next().await {
        stats.observe(result?);
    }
    let sent = stats.delivered;
    let attempted = targets.len();
    let success = sent > 0 && sent == attempted;
    let message = if attempted == 0 {
        "No FCM token registered for this user."
    } else if success {
        "Test notification sent."
    } else {
        "Test notification failed."
    };
    Ok(Json(
        json!({"success": success, "message": message, "attempted_tokens": attempted,
        "success_count": sent, "failure_count": attempted - sent, "invalid_tokens": stats.invalid}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn violation_dispatch_requires_explicit_ids_and_type() {
        let payload = json!({
            "site":"demo", "site_id":4, "stream_name":"camera", "violation_id":4294967296_i64,
            "type":"violation", "title":"Alert", "deep_link":"/violations", "metadata":{},
            "body":{"warning_no_hardhat":{"count":1}}
        });
        serde_json::from_value::<Event>(payload.clone())
            .unwrap()
            .validate()
            .unwrap();
        for field in ["site_id", "violation_id", "type"] {
            let mut missing = payload.clone();
            let value = missing.as_object_mut().unwrap().remove(field).unwrap();
            if field == "type" {
                missing["notification_type"] = value;
            }
            assert!(
                serde_json::from_value::<Event>(missing).is_err(),
                "accepted missing {field}"
            );
        }
        for field in ["site_id", "violation_id"] {
            for value in [Value::Null, json!("42")] {
                let mut malformed = payload.clone();
                malformed[field] = value;
                assert!(serde_json::from_value::<Event>(malformed).is_err());
            }
            let mut zero = payload.clone();
            zero[field] = json!(0);
            assert!(
                serde_json::from_value::<Event>(zero)
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
    }

    #[test]
    fn pipeline_payload_translates_all_configured_locales() {
        let req: Event = serde_json::from_value(json!({
            "site":"demo", "site_id":4, "stream_name":"camera", "type":"violation",
            "body":{"warning_no_hardhat":{"count":7}}, "violation_id":42,
            "title":"Safety violation alert", "deep_link":"/violations?violation_id=42",
            "metadata": {}
        }))
        .unwrap();
        req.validate().unwrap();
        assert_eq!(catalogue().len(), 8);
        for language in catalogue().keys() {
            let (_, text) = req.text(language);
            assert!(text.contains('7') && !text.contains("{count}"));
        }
        assert!(req.text("zh-TW").1.contains("未佩戴安全帽"));
    }
}

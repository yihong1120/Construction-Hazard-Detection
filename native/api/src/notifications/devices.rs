use axum::{
    Extension, Json,
    extract::State,
    http::{HeaderMap, StatusCode},
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{ApiError, AppState, tenancy::database_unavailable};

use super::{Settings, account, delivery::catalogue, token_hash};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Registration {
    device_token: String,
    device_lang: String,
    platform: Platform,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Platform {
    Android,
    Ios,
    Web,
}

impl Platform {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Android => "android",
            Self::Ios => "ios",
            Self::Web => "web",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Unregistration {
    device_token: String,
}

fn validate_token(token: &str) -> Result<(), ApiError> {
    if token.is_empty() || token.len() > 4096 || !token.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid FCM device token",
        ));
    }
    Ok(())
}

async fn invalidate_token_cache(settings: &Settings, owners: &[i32], token: &str, hash: &str) {
    let mut connection = settings.0.redis.clone();
    let mut pipe = redis::pipe();
    pipe.atomic();
    for owner in owners {
        pipe.cmd("HDEL")
            .arg(format!("fcm_tokens:{owner}"))
            .arg(token)
            .ignore()
            .cmd("SREM")
            .arg(format!("fcm_token_index:{owner}"))
            .arg(hash)
            .ignore()
            .cmd("DEL")
            .arg(format!("fcm_token_meta:{owner}:{hash}"))
            .arg(format!("fcm_tokens_ready:{owner}"))
            .ignore();
    }
    if pipe.query_async::<()>(&mut connection).await.is_err() {
        tracing::warn!("Device token cache invalidation deferred");
    }
}

pub(super) async fn register(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Json(reg): Json<Registration>,
) -> Result<Json<Value>, ApiError> {
    let (_, owner) = account(&state, &settings, &headers).await?;
    validate_token(&reg.device_token)?;
    if !catalogue().contains_key(reg.device_lang.as_str()) {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Unsupported device language",
        ));
    }
    let hash = token_hash(&reg.device_token);
    let encrypted = settings.0.cipher.encrypt(reg.device_token.as_bytes());
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let dates: Option<(DateTime<Utc>, DateTime<Utc>)> = sqlx::query_as(
        "INSERT INTO fcm_device_tokens (
            user_id, device_token_encrypted, device_token_hash, platform, device_lang,
            permission_status, last_seen_at, created_at, updated_at
         ) VALUES ($1, $2, $3, $4, $5, 'unknown', now(), now(), now())
         ON CONFLICT (device_token_hash) DO NOTHING
         RETURNING created_at, last_seen_at",
    )
    .bind(owner.id)
    .bind(&encrypted)
    .bind(&hash)
    .bind(reg.platform.as_str())
    .bind(&reg.device_lang)
    .fetch_optional(&mut *tx)
    .await
    .map_err(database_unavailable)?;
    let mut owners = vec![owner.id];
    let (created, seen) = match dates {
        Some(dates) => dates,
        None => {
            let previous_owner: i32 = sqlx::query_scalar(
                "SELECT user_id FROM fcm_device_tokens
                 WHERE device_token_hash = $1 FOR UPDATE",
            )
            .bind(&hash)
            .fetch_optional(&mut *tx)
            .await
            .map_err(database_unavailable)?
            .ok_or_else(|| {
                ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "device_registration_changed",
                )
            })?;
            if previous_owner != owner.id {
                owners.push(previous_owner);
            }
            sqlx::query_as(
                "UPDATE fcm_device_tokens SET
                    user_id = $1, device_token_encrypted = $2,
                    platform = $4, device_lang = $5, permission_status = 'unknown',
                    app_version = NULL, web_vapid_key_available = NULL,
                    web_service_worker_registered = NULL, disabled_at = NULL, failure_reason = NULL,
                    last_seen_at = GREATEST(clock_timestamp(), last_seen_at + interval '1 microsecond'),
                    updated_at = clock_timestamp()
                 WHERE device_token_hash = $3 RETURNING created_at, last_seen_at",
            )
            .bind(owner.id)
            .bind(&encrypted)
            .bind(&hash)
            .bind(reg.platform.as_str())
            .bind(&reg.device_lang)
            .fetch_one(&mut *tx)
            .await
            .map_err(database_unavailable)?
        }
    };
    tx.commit().await.map_err(database_unavailable)?;
    invalidate_token_cache(&settings, &owners, &reg.device_token, &hash).await;
    Ok(Json(
        json!({"ok": true, "updated": true, "user_id": owner.id, "device_lang": reg.device_lang,
        "registered_at": created, "last_seen_at": seen}),
    ))
}

pub(super) async fn unregister(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Json(request): Json<Unregistration>,
) -> Result<Json<Value>, ApiError> {
    let (_, owner) = account(&state, &settings, &headers).await?;
    validate_token(&request.device_token)?;
    let hash = token_hash(&request.device_token);
    let result = sqlx::query(
        "UPDATE fcm_device_tokens SET disabled_at = now(), updated_at = now(),
            failure_reason = NULL, last_failure_at = NULL
         WHERE user_id = $1 AND device_token_hash = $2 AND disabled_at IS NULL",
    )
    .bind(owner.id)
    .bind(&hash)
    .execute(&state.db)
    .await
    .map_err(database_unavailable)?;
    invalidate_token_cache(&settings, &[owner.id], &request.device_token, &hash).await;
    let message = if result.rows_affected() > 0 {
        "Token deleted."
    } else {
        "Token not found."
    };
    Ok(Json(json!({"message": message})))
}

pub(super) async fn status(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let (_, owner) = account(&state, &settings, &headers).await?;
    let devices: Vec<Value> = sqlx::query_scalar(
        "SELECT jsonb_build_object(
            'token_hash', device_token_hash, 'registration_revision', last_seen_at,
            'platform', platform, 'device_lang', device_lang, 'permission_status', permission_status,
            'registered_at', created_at, 'last_seen_at', last_seen_at, 'last_success_at', last_success_at,
            'last_failure_at', last_failure_at, 'failure_reason', failure_reason,
            'is_active', disabled_at IS NULL)
         FROM fcm_device_tokens WHERE user_id = $1
         ORDER BY disabled_at ASC NULLS FIRST, last_seen_at DESC, id DESC",
    ).bind(owner.id).fetch_all(&state.db).await.map_err(database_unavailable)?;
    let count = devices
        .iter()
        .filter(|device| device["is_active"] == true)
        .count();
    Ok(Json(
        json!({"user_id": owner.id, "has_fcm_token": count > 0, "token_count": count, "devices": devices}),
    ))
}

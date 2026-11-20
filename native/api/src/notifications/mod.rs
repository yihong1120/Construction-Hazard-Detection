mod delivery;
mod devices;
mod fcm;
mod inbox;
mod preferences;
mod receipts;

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Extension, Router,
    http::{HeaderMap, StatusCode},
    routing::{delete, get, patch},
};
use redis::aio::ConnectionManager;
use serde::Deserialize;
use sqlx::PgPool;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::{
    ApiError, AppState,
    auth::{Actor, Grant},
    tenancy::{Account, DeploymentScope, private_response},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub token_key_env: String,
    fcm: Option<fcm::Config>,
    #[serde(default)]
    receipt_maintenance: bool,
}

#[derive(Clone)]
struct Settings(Arc<Inner>);

struct Inner {
    scope: DeploymentScope,
    cipher: fernet::Fernet,
    fcm: Option<fcm::Sender>,
    redis: ConnectionManager,
    dispatch_slots: Arc<Semaphore>,
}

pub fn routes(
    config: Config,
    scope: DeploymentScope,
    redis: ConnectionManager,
    db: PgPool,
    shutdown: CancellationToken,
) -> Result<Router<AppState>> {
    let key = std::env::var(&config.token_key_env)
        .with_context(|| format!("Set {}", config.token_key_env))?;
    let cipher = fernet::Fernet::new(&key).context("Invalid device token encryption key")?;
    let fcm = config.fcm.map(fcm::Sender::new).transpose()?;
    let sends_enabled = fcm.is_some();
    if config.receipt_maintenance {
        receipts::start(db, shutdown)?;
    }
    let settings = Settings(Arc::new(Inner {
        scope,
        cipher,
        fcm,
        redis,
        dispatch_slots: Arc::new(Semaphore::new(2)),
    }));
    let router = Router::new()
        .route(
            "/devices",
            axum::routing::put(devices::register).delete(devices::unregister),
        )
        .route("/notifications", get(inbox::list))
        .route("/notifications/unread_count", get(inbox::unread_count))
        .route("/notifications/device-status", get(devices::status))
        .route("/notifications/read_all", patch(inbox::read_all))
        .route("/notifications/{id}/read", patch(inbox::mark_read))
        .route("/notifications/{id}", delete(inbox::delete))
        .route(
            "/notifications/site_preferences",
            get(preferences::list).put(preferences::update),
        );
    let router = if sends_enabled {
        router
            .route(
                "/send_fcm_notification",
                axum::routing::post(delivery::send_site),
            )
            .route(
                "/notifications/test",
                axum::routing::post(delivery::test_device),
            )
    } else {
        router
    };
    Ok(router
        .layer(Extension(settings))
        .layer(axum::middleware::map_response(private_response)))
}

async fn account(
    state: &AppState,
    settings: &Settings,
    headers: &HeaderMap,
) -> Result<(Actor, Account), ApiError> {
    let actor = state
        .auth
        .authorize(headers, Grant::NotificationRead)
        .await?;
    let owner = settings.0.scope.account(&state.db, &actor.subject).await?;
    Ok((actor, owner))
}

fn temporary_failure() -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "Notification service unavailable",
    )
}

fn token_hash(token: &str) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(token.as_bytes()))
}

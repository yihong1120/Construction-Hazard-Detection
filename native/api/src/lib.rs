mod accounts;
pub use accounts::website_access_gateway;
pub mod auth;
mod bff;
#[cfg(feature = "tensorrt")]
pub mod detection;
mod evidence;
mod native_social;
mod notifications;
pub mod registry;
pub mod streaming;
mod tenancy;
mod violations;

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use serde::Deserialize;
use serde_json::json;
use sqlx::{
    PgPool,
    postgres::{PgConnectOptions, PgPoolOptions},
};
use tokio_util::sync::CancellationToken;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

use streaming::{discovery, media_auth, metadata, playback};
use tenancy::DeploymentScope;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub listen: SocketAddr,
    pub oidc: auth::OidcSettings,
    #[serde(default)]
    pub grants: Vec<auth::ServiceGrant>,
    #[serde(default)]
    pub database_access: Option<auth::DatabaseAccessConfig>,
    #[serde(default)]
    pub cors_origins: Vec<String>,
    #[serde(default = "enabled")]
    pub evidence_api: bool,
    #[serde(default = "enabled")]
    pub streaming_api: bool,
    #[serde(default)]
    pub media_auth: bool,
    #[serde(default)]
    pub metadata_prefix: String,
    #[serde(default)]
    pub playback: Option<playback::Config>,
    #[serde(default)]
    pub violation_history: Option<violations::Config>,
    #[serde(default)]
    pub deployment_scope: Option<DeploymentScope>,
    #[serde(default)]
    pub account_management: bool,
    #[serde(default = "enabled")]
    pub account_operation_worker: bool,
    #[serde(default)]
    bff: Option<bff::Config>,
    #[serde(default)]
    native_social: Option<native_social::Config>,
    #[serde(default)]
    pub registry: Option<registry::Config>,
    #[serde(default)]
    pub notifications: Option<notifications::Config>,
    #[cfg(feature = "tensorrt")]
    #[serde(default)]
    pub detection: Option<detection::Config>,
}

fn enabled() -> bool {
    true
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.listen.ip().is_loopback(),
            "listen must use a loopback address"
        );
        self.oidc.validate()?;
        ensure!(
            self.playback.is_none()
                || self.media_auth
                || (self.bff.is_some() && self.deployment_scope.is_some()),
            "Playback requires media_auth or a BFF with deployment_scope"
        );
        ensure!(
            self.deployment_scope.is_some()
                || (!self.account_management && self.notifications.is_none()),
            "Account management and notifications require deployment_scope"
        );
        if let Some(config) = &self.bff {
            config.validate()?;
        }
        if let Some(config) = &self.native_social {
            ensure!(
                self.account_management
                    && self.database_access.is_some()
                    && self.deployment_scope.is_some(),
                "Native social authentication requires the configured account deployment"
            );
            config.validate()?;
        }
        if let Some(config) = &self.registry {
            config.validate()?;
        }
        cors_origins(&self.cors_origins)?;
        if let Some(config) = &self.database_access {
            ensure!(
                self.deployment_scope.is_some(),
                "Database access requires a deployment scope"
            );
            config.validate()?;
        }
        if let (Some(scope), Some(history)) = (self.deployment_scope, &self.violation_history) {
            ensure!(
                scope.tenant_id == history.tenant_id,
                "Violation history and deployment_scope must use the same tenant"
            );
        }
        #[cfg(feature = "tensorrt")]
        if let Some(detection) = &self.detection {
            detection.validate()?;
        }
        if self.database_access.is_none() || !self.grants.is_empty() {
            auth::validate_grants(&self.grants)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    pub auth: auth::Authorizer,
    pub media_auth: media_auth::MediaAuth,
    pub metadata: metadata::Metadata,
    evidence_api: bool,
    pub playback: Option<playback::Playback>,
    bff: Option<bff::Bff>,
    ice: Option<discovery::IceSettings>,
    pub shutdown: CancellationToken,
    pub websocket_lifetime: Duration,
    #[cfg(feature = "tensorrt")]
    pub inference: Option<std::sync::Arc<detection::Inference>>,
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    detail: &'static str,
}

impl ApiError {
    pub const fn new(status: StatusCode, detail: &'static str) -> Self {
        Self { status, detail }
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn database(error: sqlx::Error) -> Self {
        tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "database operation failed");
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "Evidence storage unavailable",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, Json(json!({"detail": self.detail}))).into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, "Bearer".parse().unwrap());
        }
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
        response
    }
}

pub async fn connect_database() -> Result<PgPool> {
    let url = std::env::var("RUST_API_DATABASE_URL").context("Set RUST_API_DATABASE_URL")?;
    let mut options = url
        .parse::<PgConnectOptions>()
        .map_err(|_| anyhow::anyhow!("Invalid Rust API database URL"))?;
    if let Ok(schema) = std::env::var("RUST_API_DATABASE_SCHEMA") {
        ensure!(
            !schema.is_empty()
                && schema.len() <= 63
                && schema.as_bytes()[0].is_ascii_lowercase()
                && schema.bytes().all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'
                }),
            "Invalid Rust API database schema"
        );
        let path = if schema == "public" {
            schema
        } else {
            format!("{schema},public")
        };
        options = options.options([("search_path", path)]);
    }
    PgPoolOptions::new()
        .max_connections(8)
        .acquire_timeout(Duration::from_secs(5))
        .connect_with(options)
        .await
        .map_err(|_| anyhow::anyhow!("Could not connect to the evidence database"))
}

pub async fn init_database(db: &PgPool) -> Result<()> {
    let mut transaction = db.begin().await?;
    // The lock serialises explicit schema initialisation across replicas.
    sqlx::query("SELECT pg_advisory_xact_lock(20188366648888338)")
        .execute(&mut *transaction)
        .await?;
    sqlx::raw_sql(include_str!("evidence.sql"))
        .execute(&mut *transaction)
        .await?;
    transaction.commit().await?;
    Ok(())
}

pub async fn app(config: Config, db: PgPool, shutdown: CancellationToken) -> Result<Router> {
    config.validate()?;
    let ice = discovery::IceSettings::from_env()?;
    if let Some(scope) = config.deployment_scope {
        scope.validate(&db).await?;
    }
    let database_access = match config.database_access {
        Some(access) => Some(
            auth::DatabaseAccess::new(
                access,
                config.deployment_scope.expect("configured scope"),
                db.clone(),
            )
            .await?,
        ),
        None => None,
    };
    let websocket_lifetime = websocket_lifetime()?;
    let url = std::env::var("RUST_API_REDIS_URL").context("Set RUST_API_REDIS_URL")?;
    let redis_client =
        redis::Client::open(url).map_err(|_| anyhow::anyhow!("Invalid Rust API Redis URL"))?;
    let redis = ConnectionManager::new_with_config(
        redis_client.clone(),
        ConnectionManagerConfig::new()
            .set_connection_timeout(Duration::from_secs(3))
            .set_response_timeout(Duration::from_secs(3)),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Could not connect to the Rust API Redis service"))?;
    #[cfg(feature = "tensorrt")]
    let inference = config.detection.map(detection::start).transpose()?;
    let playback = config
        .playback
        .map(|settings| playback::Playback::new(settings, redis.clone(), config.deployment_scope))
        .transpose()?;
    let bff = config
        .bff
        .map(|settings| bff::Bff::new(settings, redis.clone(), config.cors_origins.clone()))
        .transpose()?;
    let registry = config
        .registry
        .map(|settings| registry::routes(settings, redis.clone(), config.deployment_scope))
        .transpose()?;
    let native_social = config
        .native_social
        .map(|settings| {
            native_social::routes(
                settings,
                &config.oidc.issuer,
                config.deployment_scope.expect("configured scope"),
                redis.clone(),
            )
        })
        .transpose()?;
    let notifications = config
        .notifications
        .map(|settings| {
            notifications::routes(
                settings,
                config.deployment_scope.expect("configured scope"),
                redis.clone(),
                db.clone(),
                shutdown.clone(),
            )
        })
        .transpose()?;
    let state = AppState {
        db,
        auth: auth::Authorizer::configured(
            config.oidc,
            config.grants,
            redis.clone(),
            database_access,
        )?,
        media_auth: media_auth::MediaAuth::new(redis_client.clone()).await?,
        metadata: metadata::Metadata::new(redis_client, redis).with_prefix(config.metadata_prefix),
        evidence_api: config.evidence_api,
        playback,
        bff,
        ice,
        shutdown,
        websocket_lifetime,
        #[cfg(feature = "tensorrt")]
        inference,
    };
    let production_uploads = config
        .violation_history
        .as_ref()
        .is_some_and(|settings| settings.write_uploads);
    let router = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready));
    let router = if config.streaming_api {
        router.merge(discovery::routes()).merge(metadata::routes())
    } else {
        router
    };
    let router = if config.evidence_api {
        router.merge(evidence::routes(!production_uploads))
    } else {
        router
    };
    let router = if config.media_auth {
        router.merge(media_auth::routes())
    } else {
        router
    };
    let router = if state.playback.is_some() {
        router.merge(if config.media_auth {
            playback::routes()
        } else {
            playback::facade_routes()
        })
    } else {
        router
    };
    let router = if config.account_management {
        router.merge(
            accounts::routes(
                config.deployment_scope.expect("configured scope"),
                &state.db,
                state.shutdown.clone(),
                config.account_operation_worker,
            )
            .await?,
        )
    } else {
        router
    };
    let router = if state.bff.is_some() {
        router.merge(bff::routes())
    } else {
        router
    };
    let router = if let Some(routes) = registry {
        router.merge(routes)
    } else {
        router
    };
    let router = if let Some(routes) = native_social {
        router.merge(routes)
    } else {
        router
    };
    let router = if let Some(routes) = notifications {
        router.merge(routes)
    } else {
        router
    };
    let router = if let Some(settings) = config.violation_history {
        router.merge(violations::routes(settings, &state.db, state.shutdown.clone()).await?)
    } else {
        router
    };
    #[cfg(feature = "tensorrt")]
    let router = if state.inference.is_some() {
        router.merge(detection::routes())
    } else {
        router
    };
    let router = if config.cors_origins.is_empty() {
        router
    } else {
        router.layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(cors_origins(&config.cors_origins)?))
                .allow_credentials(true)
                .allow_methods(AllowMethods::mirror_request())
                .allow_headers(AllowHeaders::mirror_request()),
        )
    };
    Ok(router.with_state(state))
}

fn cors_origins(config: &[String]) -> Result<Vec<axum::http::HeaderValue>> {
    config
        .iter()
        .map(|origin| {
            let url = url::Url::parse(origin)?;
            let local_http = url.scheme() == "http"
                && url.host_str().is_some_and(|host| {
                    host == "localhost"
                        || host
                            .trim_matches(['[', ']'])
                            .parse::<std::net::IpAddr>()
                            .is_ok_and(|ip| ip.is_loopback())
                });
            ensure!(
                (url.scheme() == "https" || local_http)
                    && url.origin().ascii_serialization() == *origin,
                "Configure complete HTTPS or loopback browser origins"
            );
            Ok(origin.parse()?)
        })
        .collect()
}

fn websocket_lifetime() -> Result<Duration> {
    let seconds = match std::env::var("WS_MAX_SESSION_SECONDS") {
        Ok(raw) => raw.parse().context("Invalid WS_MAX_SESSION_SECONDS")?,
        Err(std::env::VarError::NotPresent) => 1800.0,
        Err(_) => anyhow::bail!("Invalid WS_MAX_SESSION_SECONDS"),
    };
    let lifetime =
        Duration::try_from_secs_f64(seconds).context("Invalid WS_MAX_SESSION_SECONDS")?;
    ensure!(
        !lifetime.is_zero(),
        "WS_MAX_SESSION_SECONDS must be positive"
    );
    Ok(lifetime)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn ready(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let query = if state.evidence_api {
        "SELECT to_regclass('rust_api.evidence') IS NOT NULL"
    } else {
        "SELECT TRUE"
    };
    if !sqlx::query_scalar::<_, bool>(query)
        .fetch_one(&state.db)
        .await
        .map_err(ApiError::database)?
    {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "Evidence schema unavailable",
        ));
    }
    state.auth.ready().await?;
    Ok(Json(json!({"status": "ready"})))
}

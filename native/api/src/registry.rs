mod management;
mod tenants;

use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use aws_lc_rs::signature::Ed25519KeyPair;
use axum::{
    Extension, Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::{
    Engine,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use hmac::{Hmac, Mac};
use redis::aio::ConnectionManager;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use url::Url;
use uuid::Uuid;

use crate::{
    AppState,
    tenancy::{DeploymentScope, private_response},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub api_base_path: String,
    pub key_id: String,
    pub signing_key_env: String,
    pub enrollment_pepper_env: String,
    pub document_ttl_seconds: i64,
    pub enrollment_rate_limit: i64,
    pub enrollment_rate_window_seconds: i64,
    #[serde(default)]
    pub invitation_policy_file: Option<PathBuf>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.api_base_path.starts_with('/')
                && self.api_base_path.len() <= 256
                && self.api_base_path[1..].split('/').all(|part| {
                    !part.is_empty()
                        && part
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                }),
            "Invalid Registry API root path"
        );
        ensure!(
            (1..=256).contains(&self.key_id.len())
                && self.key_id.as_bytes()[0].is_ascii_alphanumeric()
                && self.key_id.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                })
                && (1..=86_400).contains(&self.document_ttl_seconds)
                && (1..=1000).contains(&self.enrollment_rate_limit)
                && (1..=86_400).contains(&self.enrollment_rate_window_seconds),
            "Invalid Registry signature or enrollment policy"
        );
        Ok(())
    }
}

#[derive(Clone)]
struct Registry(Arc<Inner>);

struct Inner {
    config: Config,
    key: Ed25519KeyPair,
    pepper: Hmac<Sha256>,
    redis: ConnectionManager,
}

fn signing_key(pem: &str) -> Result<Ed25519KeyPair> {
    let pem = pem.trim().replace("\\n", "\n");
    let body = pem
        .strip_prefix("-----BEGIN PRIVATE KEY-----")
        .and_then(|body| body.strip_suffix("-----END PRIVATE KEY-----"))
        .context("Registry signing key must use PKCS#8 PEM")?;
    let der = STANDARD
        .decode(body.split_ascii_whitespace().collect::<String>())
        .map_err(|_| anyhow::anyhow!("Invalid Registry PEM encoding"))?;
    Ed25519KeyPair::from_pkcs8_maybe_unchecked(&der)
        .map_err(|_| anyhow::anyhow!("Registry signing key must be Ed25519"))
}

pub(crate) fn routes(
    config: Config,
    redis: ConnectionManager,
    scope: Option<DeploymentScope>,
) -> Result<Router<AppState>> {
    config.validate()?;
    let key = signing_key(
        &std::env::var(&config.signing_key_env).context("Set the Registry signing key")?,
    )?;
    let pepper = std::env::var(&config.enrollment_pepper_env)
        .context("Set the Registry enrollment pepper")?;
    ensure!(
        pepper.len() >= 32,
        "Registry enrollment pepper is too short"
    );
    let registry = Registry(Arc::new(Inner {
        config,
        key,
        pepper: Hmac::new_from_slice(pepper.as_bytes()).expect("HMAC accepts any key length"),
        redis,
    }));
    let routes = Router::new()
        .route("/deployment-registry/v1/deployments/{id}", get(document))
        .route(
            "/deployment-registry/v1/enrollments/exchange",
            post(exchange),
        );
    let routes = if let Some(scope) = scope {
        routes
            .merge(management::routes(scope))
            .merge(tenants::routes(scope))
    } else {
        routes
    };
    Ok(routes
        .layer(Extension(registry))
        .layer(axum::middleware::map_response(private_response)))
}

fn rejection(status: StatusCode, code: &str) -> Response {
    (status, Json(json!({"detail": {"code": code}}))).into_response()
}

fn unavailable() -> Response {
    rejection(StatusCode::SERVICE_UNAVAILABLE, "registry_unavailable")
}

fn canonical_url(value: &str, root: &str) -> bool {
    Url::parse(value).is_ok_and(|url| {
        url.scheme() == "https"
            && url.has_host()
            && !url.host_str().is_some_and(|host| host.ends_with('.'))
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && url.path() == root
            && url.as_str() == value
    })
}

async fn document(
    State(state): State<AppState>,
    Extension(registry): Extension<Registry>,
    Path(id): Path<Uuid>,
) -> Response {
    let row: Option<(Uuid, String, i32, bool)> = match sqlx::query_as(
        "SELECT d.tenant_id, d.api_base_url, d.config_revision,
                d.status = 'active' AND t.status = 'active'
         FROM public.deployments d JOIN public.tenants t ON t.id = d.tenant_id
         WHERE d.id = $1",
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await
    {
        Ok(row) => row,
        Err(_) => return unavailable(),
    };
    let Some((tenant, api_base_url, revision, active)) = row else {
        return rejection(StatusCode::NOT_FOUND, "deployment_not_found");
    };
    if !active {
        return rejection(StatusCode::GONE, "deployment_revoked");
    }
    if revision < 1 || !canonical_url(&api_base_url, &registry.0.config.api_base_path) {
        return unavailable();
    }
    let now = chrono::Utc::now().timestamp();
    let mut result = json!({
        "api_base_url": api_base_url,
        "config_revision": revision,
        "deployment_id": id,
        "expires_at": now + registry.0.config.document_ttl_seconds,
        "issued_at": now,
        "schema_version": 1,
        "tenant_id": tenant,
    });
    let signature = registry.0.key.sign(&canonical_document(&result));
    result["key_id"] = json!(registry.0.config.key_id);
    result["signature"] = json!(URL_SAFE_NO_PAD.encode(signature.as_ref()));
    Json(result).into_response()
}

fn canonical_document(document: &Value) -> Vec<u8> {
    // Explicit sorted map preserves Flutter's signature bytes independently
    // of serde_json's object ordering feature.
    let fields: std::collections::BTreeMap<_, _> = document
        .as_object()
        .expect("Registry document is an object")
        .iter()
        .collect();
    serde_json::to_vec(&fields).expect("Registry fields contain JSON values")
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Exchange {
    enrollment_code: String,
}

async fn exchange(
    State(state): State<AppState>,
    Extension(registry): Extension<Registry>,
    headers: HeaderMap,
    Json(input): Json<Exchange>,
) -> Response {
    let raw = input.enrollment_code;
    if !(16..=512).contains(&raw.len())
        || !raw
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return rejection(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_enrollment_code_format",
        );
    }
    // The loopback listener receives X-Real-IP from the Registry's Nginx
    // location, which overwrites inbound values with the remote address.
    let client = headers
        .get("x-real-ip")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<IpAddr>().ok())
        .map_or_else(|| "unknown-client".into(), |ip| ip.to_string());
    let verifier = verifier(&registry, &raw);
    match rate_limit(&registry, &client, &verifier).await {
        Ok(Some(seconds)) => {
            let mut response = rejection(StatusCode::TOO_MANY_REQUESTS, "enrollment_rate_limited");
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, seconds.to_string().parse().unwrap());
            return response;
        }
        Ok(None) => {}
        Err(_) => return unavailable(),
    }
    match redeem(&state.db, &verifier).await {
        Ok(Redemption::Redeemed(deployment)) => {
            Json(json!({"deployment_id": deployment})).into_response()
        }
        Ok(Redemption::Unknown) => rejection(StatusCode::FORBIDDEN, "invalid_enrollment_code"),
        Ok(Redemption::Terminal) => {
            rejection(StatusCode::GONE, "enrollment_unavailable_or_expired")
        }
        Err(_) => unavailable(),
    }
}

fn verifier(registry: &Registry, code: &str) -> String {
    let mut mac = registry.0.pepper.clone();
    mac.update(code.as_bytes());
    format!("{:x}", mac.finalize().into_bytes())
}

async fn rate_limit(
    registry: &Registry,
    client: &str,
    verifier: &str,
) -> Result<Option<i64>, redis::RedisError> {
    let mut pipeline = redis::pipe();
    for (scope, source) in [("ip", client), ("code", verifier)] {
        let identifier = format!("{:x}", Sha256::digest(source));
        pipeline
            .cmd("EVAL")
            .arg(
                "local current = redis.call('INCR', KEYS[1])
                 local ttl = redis.call('TTL', KEYS[1])
                 if ttl < 0 then
                    ttl = tonumber(ARGV[1])
                    redis.call('EXPIRE', KEYS[1], ttl)
                 end
                 return { current, ttl }",
            )
            .arg(1)
            .arg(format!(
                "construction-hazard-detection:deployment-enrollment:{scope}:{identifier}"
            ))
            .arg(registry.0.config.enrollment_rate_window_seconds);
    }
    let checks: Vec<(i64, i64)> = pipeline.query_async(&mut registry.0.redis.clone()).await?;
    Ok(checks.into_iter().find_map(|(attempts, ttl)| {
        (attempts > registry.0.config.enrollment_rate_limit).then_some(ttl.max(1))
    }))
}

enum Redemption {
    Unknown,
    Terminal,
    Redeemed(Uuid),
}

async fn redeem(db: &PgPool, verifier: &str) -> Result<Redemption, sqlx::Error> {
    let mut tx = db.begin().await?;
    let row: Option<(i64, Uuid, bool)> = sqlx::query_as(
        "SELECT id, deployment_id,
                expires_at > CURRENT_TIMESTAMP AND revoked_at IS NULL AND redeemed_at IS NULL
         FROM public.deployment_enrollment_codes
         WHERE code_verifier_hash = $1 FOR UPDATE",
    )
    .bind(verifier)
    .fetch_optional(&mut *tx)
    .await?;
    let Some((id, deployment, eligible)) = row else {
        return Ok(Redemption::Unknown);
    };
    if !eligible {
        return Ok(Redemption::Terminal);
    }
    let active: Option<Uuid> = sqlx::query_scalar(
        "SELECT d.id FROM public.deployments d JOIN public.tenants t ON t.id = d.tenant_id
         WHERE d.id = $1 AND d.status = 'active' AND t.status = 'active' FOR UPDATE OF d, t",
    )
    .bind(deployment)
    .fetch_optional(&mut *tx)
    .await?;
    if active.is_none() {
        return Ok(Redemption::Terminal);
    }
    sqlx::query(
        "UPDATE public.deployment_enrollment_codes SET redeemed_at = CURRENT_TIMESTAMP
         WHERE id = $1",
    )
    .bind(id)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(Redemption::Redeemed(deployment))
}

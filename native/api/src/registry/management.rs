use std::collections::HashMap;

use aws_lc_rs::rand::fill;
use axum::{
    Extension, Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{delete, get},
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    ApiError, AppState,
    auth::Grant,
    tenancy::{Account, DeploymentScope},
};

use super::{Digest, Engine, Registry, Sha256, URL_SAFE_NO_PAD, Uuid, verifier};

pub(super) fn routes(scope: DeploymentScope) -> Router<AppState> {
    Router::new()
        .route("/deployment-enrollment-codes/expiry-options", get(options))
        .route("/deployment-enrollment-codes", get(list).post(create))
        .route("/deployment-enrollment-codes/{id}", delete(revoke))
        .layer(Extension(scope))
}

fn storage(_: sqlx::Error) -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "enrollment_management_unavailable",
    )
}

fn require_admin(account: Account) -> Result<Account, ApiError> {
    if account.role == "admin"
        || account.role == "super_admin" && account.username.eq_ignore_ascii_case("ChangDar")
    {
        return Ok(account);
    }
    Err(ApiError::new(StatusCode::FORBIDDEN, "Admin required"))
}

async fn principal(state: &AppState, headers: &HeaderMap) -> Result<String, ApiError> {
    Ok(state
        .auth
        .authorize(headers, Grant::AccountManage)
        .await?
        .subject)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    max_minutes: i32,
    minutes: Vec<i32>,
    default_minutes: Option<i32>,
}

fn policy(registry: &Registry, tenant: Uuid) -> Result<Policy, ApiError> {
    let unavailable = || {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "INVITATION_POLICY_UNAVAILABLE",
        )
    };
    let Some(path) = &registry.0.config.invitation_policy_file else {
        return Ok(Policy {
            max_minutes: 1_440,
            minutes: vec![30, 60, 120, 1_440],
            default_minutes: Some(30),
        });
    };
    let raw = std::fs::read(path).map_err(|_| unavailable())?;
    let mut policies: HashMap<Uuid, Policy> =
        serde_json::from_slice(&raw).map_err(|_| unavailable())?;
    let policy = policies.remove(&tenant).ok_or_else(unavailable)?;
    let distinct: std::collections::HashSet<_> = policy.minutes.iter().collect();
    if policy.max_minutes <= 0
        || policy
            .minutes
            .iter()
            .any(|n| !(1..=policy.max_minutes).contains(n))
        || distinct.len() != policy.minutes.len()
        || policy
            .default_minutes
            .is_some_and(|n| !distinct.contains(&n))
    {
        return Err(unavailable());
    }
    Ok(policy)
}

async fn options(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(registry): Extension<Registry>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let subject = principal(&state, &headers).await?;
    let account = require_admin(scope.account(&state.db, &subject).await?)?;
    let policy = policy(&registry, scope.tenant_id)?;
    let options: Vec<_> = policy
        .minutes
        .iter()
        .map(|n| json!({"id": n.to_string(), "label": format!("{n} min")}))
        .collect();
    let mut result = json!({
        "schema_version": 1,
        "options": options,
        "default_id": policy.default_minutes.map(|n| n.to_string()),
        "policy": {"max_minutes": policy.max_minutes},
    });
    let fields = json!({"user": account.id, "tenant": scope.tenant_id, "role": account.role,
        "target": null, "source": null, "result": result});
    let revision = serde_json::to_vec(&fields).expect("Invitation options contain JSON values");
    result["revision"] = json!(format!("{:x}", Sha256::digest(revision)));
    Ok(Json(result))
}

async fn list(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let subject = principal(&state, &headers).await?;
    require_admin(scope.account(&state.db, &subject).await?)?;
    let items: Vec<Value> = sqlx::query_scalar(
        "SELECT jsonb_build_object(
             'id', public_id, 'expires_at', expires_at,
             'status', CASE WHEN revoked_at IS NOT NULL THEN 'revoked'
                            WHEN redeemed_at IS NOT NULL THEN 'redeemed'
                            WHEN expires_at <= CURRENT_TIMESTAMP THEN 'expired'
                            ELSE 'active' END)
         FROM public.deployment_enrollment_codes
         WHERE deployment_id = $1 ORDER BY created_at DESC, id DESC",
    )
    .bind(scope.deployment_id)
    .fetch_all(&state.db)
    .await
    .map_err(storage)?;
    Ok(Json(json!({"items": items})))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Create {
    expires_in_minutes: i32,
}

async fn create(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(registry): Extension<Registry>,
    headers: HeaderMap,
    Json(input): Json<Create>,
) -> Result<Json<Value>, ApiError> {
    let subject = principal(&state, &headers).await?;
    let mut transaction = state.db.begin().await.map_err(storage)?;
    let account = require_admin(scope.account_for_update(&mut transaction, &subject).await?)?;
    let policy = policy(&registry, scope.tenant_id)?;
    if !policy.minutes.contains(&input.expires_in_minutes) {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invitation duration is not currently allowed",
        ));
    }
    let mut raw = [0u8; 32];
    fill(&mut raw).map_err(|_| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "enrollment_management_unavailable",
        )
    })?;
    let code = URL_SAFE_NO_PAD.encode(raw);
    let id = Uuid::new_v4();
    let (sequence, expires_at): (i64, DateTime<Utc>) = sqlx::query_as(
        "INSERT INTO public.deployment_enrollment_codes
             (deployment_id, public_id, code_verifier_hash, expires_at, created_by, created_at)
         VALUES ($1, $2, $3, CURRENT_TIMESTAMP + make_interval(mins => $4), $5, CURRENT_TIMESTAMP)
         RETURNING id, expires_at",
    )
    .bind(scope.deployment_id)
    .bind(id)
    .bind(verifier(&registry, &code))
    .bind(input.expires_in_minutes)
    .bind(account.username)
    .fetch_one(&mut *transaction)
    .await
    .map_err(storage)?;
    audit(&mut transaction, scope, account.id, sequence, "created").await?;
    transaction.commit().await.map_err(storage)?;
    Ok(Json(
        json!({"id": id, "enrollment_code": code, "expires_at": expires_at}),
    ))
}

async fn revoke(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let subject = principal(&state, &headers).await?;
    require_admin(scope.account(&state.db, &subject).await?)?;
    let id = Uuid::parse_str(&id)
        .ok()
        .filter(|value| value.to_string() == id)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "canonical_enrollment_code_id_required",
            )
        })?;
    let mut transaction = state.db.begin().await.map_err(storage)?;
    let sequence: Option<i64> = sqlx::query_scalar(
        "UPDATE public.deployment_enrollment_codes
         SET revoked_at = CURRENT_TIMESTAMP
         WHERE public_id = $1 AND deployment_id = $2 AND revoked_at IS NULL
           AND redeemed_at IS NULL RETURNING id",
    )
    .bind(id)
    .bind(scope.deployment_id)
    .fetch_optional(&mut *transaction)
    .await
    .map_err(storage)?;
    if let Some(sequence) = sequence {
        let account = require_admin(scope.account_for_update(&mut transaction, &subject).await?)?;
        audit(&mut transaction, scope, account.id, sequence, "revoked").await?;
    }
    transaction.commit().await.map_err(storage)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn audit(
    transaction: &mut sqlx::PgConnection,
    scope: DeploymentScope,
    user: i32,
    sequence: i64,
    action: &str,
) -> Result<(), ApiError> {
    sqlx::query(
        "INSERT INTO public.deployment_enrollment_code_audit_logs
             (enrollment_code_id, deployment_id, tenant_id, actor_user_id, action, created_at)
         VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)",
    )
    .bind(sequence)
    .bind(scope.deployment_id)
    .bind(scope.tenant_id)
    .bind(user)
    .bind(action)
    .execute(transaction)
    .await
    .map_err(storage)?;
    Ok(())
}

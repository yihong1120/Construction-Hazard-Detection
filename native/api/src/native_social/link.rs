use axum::http::StatusCode;
use reqwest::Method;
use serde::Deserialize;
use serde_json::json;
use sqlx::{PgConnection, PgPool};

use crate::{
    ApiError,
    accounts::Keycloak,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{LINK_EXPIRED, LocalIdentity, Provider, provider::Identity, subject_shape};

const STORAGE_CONFLICT: ApiError =
    ApiError::new(StatusCode::CONFLICT, "provider_identity_already_linked");
const PROVIDER_CONFLICT: ApiError = ApiError::new(StatusCode::CONFLICT, "provider_already_linked");
const UNAVAILABLE: ApiError = ApiError::new(
    StatusCode::SERVICE_UNAVAILABLE,
    "keycloak_identity_service_unavailable",
);
const RECONCILE: ApiError = ApiError::new(
    StatusCode::SERVICE_UNAVAILABLE,
    "account_reconciliation_required",
);
const MAX_REMOTE_IDENTITIES_BYTES: usize = 64 * 1024;

pub(super) async fn email_candidate(
    db: &PgPool,
    scope: DeploymentScope,
    identity: &Identity,
) -> Result<Option<(i32, String)>, ApiError> {
    let Some(email) = &identity.email else {
        return Ok(None);
    };
    let candidates: Vec<(i32, String)> = sqlx::query_as(
        "SELECT u.id, i.provider_user_id
         FROM users u JOIN user_profiles p ON p.user_id = u.id
         JOIN public.user_identities i ON i.user_id = u.id AND i.provider = 'keycloak'
         JOIN public.tenants t ON t.id = u.tenant_id
         JOIN public.deployments d ON d.tenant_id = t.id
         WHERE u.tenant_id = $1 AND d.id = $2 AND t.status = 'active' AND d.status = 'active'
           AND u.status = 'active' AND NOT u.deletion_pending
           AND u.email_verified_at IS NOT NULL AND p.email = $3
           AND NOT EXISTS (
                SELECT 1 FROM public.user_identities present
                WHERE present.provider = $4 AND present.provider_user_id = $5
           )
         LIMIT 2",
    )
    .bind(scope.tenant_id)
    .bind(scope.deployment_id)
    .bind(email)
    .bind(identity.provider.name())
    .bind(&identity.subject)
    .fetch_all(db)
    .await
    .map_err(database_unavailable)?;
    Ok(match candidates.as_slice() {
        [(user, subject)] if subject_shape(subject) => Some((*user, subject.clone())),
        _ => None,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FederatedIdentity {
    identity_provider: String,
    user_id: String,
}

async fn current_remote(
    keycloak: &Keycloak,
    subject: &str,
    provider: Provider,
    provider_subject: &str,
) -> Result<bool, ApiError> {
    let mut response = keycloak
        .request(Method::GET, &["users", subject, "federated-identity"], None)
        .await
        .map_err(|_| UNAVAILABLE)?;
    if response.status() != StatusCode::OK {
        return Err(UNAVAILABLE);
    }
    let mut buffer = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| UNAVAILABLE)? {
        if buffer.len() + chunk.len() > MAX_REMOTE_IDENTITIES_BYTES {
            return Err(UNAVAILABLE);
        }
        buffer.extend_from_slice(&chunk);
    }
    let identities: Vec<FederatedIdentity> =
        serde_json::from_slice(&buffer).map_err(|_| UNAVAILABLE)?;
    for identity in identities {
        if identity.identity_provider == provider.name() {
            return if identity.user_id == provider_subject {
                Ok(true)
            } else {
                Err(PROVIDER_CONFLICT)
            };
        }
    }
    Ok(false)
}

async fn stage_mirror(
    db: &mut PgConnection,
    local_id: i32,
    provider: Provider,
    subject: &str,
) -> Result<(), ApiError> {
    if subject.chars().count() > 255 {
        return Err(super::provider::INVALID_CREDENTIAL);
    }
    let created: Option<i32> = sqlx::query_scalar(
        "INSERT INTO public.user_identities (
            user_id, provider, provider_user_id, email_verified, raw_email_is_private,
            linked_at, updated_at
         ) VALUES ($1, $2, $3, false, false, now(), now())
         ON CONFLICT DO NOTHING RETURNING id",
    )
    .bind(local_id)
    .bind(provider.name())
    .bind(subject)
    .fetch_optional(&mut *db)
    .await
    .map_err(database_unavailable)?;
    if created.is_some() {
        return Ok(());
    }
    let existing: Option<String> = sqlx::query_scalar(
        "SELECT provider_user_id FROM public.user_identities
         WHERE user_id = $1 AND provider = $2 FOR UPDATE",
    )
    .bind(local_id)
    .bind(provider.name())
    .fetch_optional(&mut *db)
    .await
    .map_err(database_unavailable)?;
    match existing {
        Some(value) if value == subject => Ok(()),
        Some(_) => Err(PROVIDER_CONFLICT),
        None => Err(STORAGE_CONFLICT),
    }
}

pub(super) async fn attach(
    db: &PgPool,
    scope: DeploymentScope,
    keycloak: &Keycloak,
    local: LocalIdentity,
    identity: Identity,
) -> Result<&'static str, ApiError> {
    let provider = identity.provider;
    let provider_subject = identity.subject.as_str();
    let subject = local.subject.as_str();
    let mut tx = db.begin().await.map_err(database_unavailable)?;
    let active = scope.account_for_update(&mut tx, subject).await?;
    if active.id != local.user_id {
        return Err(LINK_EXPIRED);
    }
    stage_mirror(&mut tx, local.user_id, provider, provider_subject).await?;
    let linked = current_remote(keycloak, subject, provider, provider_subject).await?;
    if linked {
        tx.commit().await.map_err(database_unavailable)?;
        return Ok("already_linked");
    }

    let data = json!({
        "identityProvider":provider.name(), "userId":provider_subject, "userName":provider_subject,
    });
    let response = keycloak
        .request(
            Method::POST,
            &["users", subject, "federated-identity", provider.name()],
            Some(&data),
        )
        .await
        .map_err(|_| RECONCILE)?;
    match response.status() {
        StatusCode::NO_CONTENT => {
            tx.commit().await.map_err(|_| RECONCILE)?;
            Ok("linked")
        }
        StatusCode::CONFLICT => {
            if current_remote(keycloak, subject, provider, provider_subject).await? {
                tx.commit().await.map_err(database_unavailable)?;
                Ok("already_linked")
            } else {
                Err(STORAGE_CONFLICT)
            }
        }
        _ => Err(RECONCILE),
    }
}

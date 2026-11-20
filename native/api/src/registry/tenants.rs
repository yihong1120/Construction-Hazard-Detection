use axum::{
    Extension, Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, patch},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::{PgConnection, Postgres, Transaction};
use url::{Host, Url};
use uuid::Uuid;

use crate::{
    AppState,
    auth::Grant,
    tenancy::{Account, DeploymentScope},
};

use super::{Registry, canonical_url, rejection, unavailable};

type Result<T> = std::result::Result<T, Response>;
type Tenant = (Uuid, String, Option<String>, String);
type Deployment = (Uuid, Uuid, String, i32, String);

pub(super) fn routes(scope: DeploymentScope) -> Router<AppState> {
    Router::new()
        .route("/admin/tenants", get(list_tenants).post(create_tenant))
        .route("/admin/tenants/{id}", patch(update_tenant))
        .route(
            "/admin/deployments",
            get(list_deployments).post(create_deployment),
        )
        .route("/admin/deployments/{id}", patch(update_deployment))
        .layer(Extension(scope))
}

fn operator(account: Account) -> Result<()> {
    if account.username.eq_ignore_ascii_case("ChangDar")
        && matches!(account.role.as_str(), "admin" | "super_admin")
    {
        Ok(())
    } else {
        Err((
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Super admin only"})),
        )
            .into_response())
    }
}

async fn principal(state: &AppState, headers: &HeaderMap) -> Result<String> {
    state
        .auth
        .authorize(headers, Grant::AccountManage)
        .await
        .map(|actor| actor.subject)
        .map_err(IntoResponse::into_response)
}

async fn read_access(state: &AppState, scope: DeploymentScope, headers: &HeaderMap) -> Result<()> {
    let subject = principal(state, headers).await?;
    operator(
        scope
            .account(&state.db, &subject)
            .await
            .map_err(IntoResponse::into_response)?,
    )
}

async fn write_access(
    state: &AppState,
    scope: DeploymentScope,
    headers: &HeaderMap,
) -> Result<Transaction<'static, Postgres>> {
    let subject = principal(state, headers).await?;
    let mut tx = state.db.begin().await.map_err(storage)?;
    // A tenant lifecycle change locks its deployments. Serialise these rare
    // administration writes before locking the operator's deployment as well.
    sqlx::query("SELECT pg_advisory_xact_lock(4701, 1)")
        .execute(&mut *tx)
        .await
        .map_err(storage)?;
    operator(
        scope
            .account_for_update(&mut tx, &subject)
            .await
            .map_err(IntoResponse::into_response)?,
    )?;
    Ok(tx)
}

fn storage(error: sqlx::Error) -> Response {
    tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "registry administration storage failed");
    unavailable()
}

fn conflict(error: sqlx::Error, code: &str) -> Response {
    if error
        .as_database_error()
        .is_some_and(sqlx::error::DatabaseError::is_unique_violation)
    {
        rejection(StatusCode::CONFLICT, code)
    } else {
        storage(error)
    }
}

fn not_found(resource: &str) -> Response {
    rejection(StatusCode::NOT_FOUND, &format!("{resource}_not_found"))
}

fn invalid_status(resource: &str, value: &str) -> Response {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({"detail": {"code": format!("invalid_{resource}_status"), "status": value}})),
    )
        .into_response()
}

fn tenant_view((id, name, description, status): Tenant) -> Value {
    json!({"id": id, "name": name, "description": description, "status": status})
}

fn deployment_view((id, tenant_id, api_base_url, config_revision, status): Deployment) -> Value {
    json!({
        "id": id, "tenant_id": tenant_id, "api_base_url": api_base_url,
        "config_revision": config_revision, "status": status,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NewTenant {
    name: String,
    description: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TenantChange {
    name: Option<String>,
    description: Option<String>,
    status: Option<String>,
}

fn tenant_fields(name: Option<String>, description: Option<&str>) -> Result<Option<String>> {
    if name
        .as_ref()
        .is_some_and(|name| !(1..=160).contains(&name.chars().count()) || name.contains('\0'))
        || description.is_some_and(|description| {
            description.chars().count() > 4_000 || description.contains('\0')
        })
    {
        return Err(rejection(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_tenant_fields",
        ));
    }
    Ok(name.map(|name| name.trim().to_owned()))
}

async fn list_tenants(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>> {
    read_access(&state, scope, &headers).await?;
    let tenants = sqlx::query_as(
        "SELECT id, name, description, status FROM public.tenants ORDER BY name, id",
    )
    .fetch_all(&state.db)
    .await
    .map_err(storage)?;
    Ok(Json(tenants.into_iter().map(tenant_view).collect()))
}

async fn create_tenant(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(input): Json<NewTenant>,
) -> Result<impl IntoResponse> {
    let mut tx = write_access(&state, scope, &headers).await?;
    let name = tenant_fields(Some(input.name), input.description.as_deref())?;
    let tenant = sqlx::query_as(
        "INSERT INTO public.tenants (id, name, description) VALUES ($1, $2, $3)
         RETURNING id, name, description, status",
    )
    .bind(Uuid::new_v4())
    .bind(name)
    .bind(input.description)
    .fetch_one(&mut *tx)
    .await
    .map_err(|error| conflict(error, "tenant_name_conflict"))?;
    tx.commit().await.map_err(storage)?;
    Ok((StatusCode::CREATED, Json(tenant_view(tenant))))
}

async fn update_tenant(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(input): Json<TenantChange>,
) -> Result<Json<Value>> {
    let mut tx = write_access(&state, scope, &headers).await?;
    let current: Tenant = sqlx::query_as(
        "SELECT id, name, description, status FROM public.tenants WHERE id = $1 FOR UPDATE",
    )
    .bind(id)
    .fetch_optional(&mut *tx)
    .await
    .map_err(storage)?
    .ok_or_else(|| not_found("tenant"))?;
    let name = tenant_fields(input.name, input.description.as_deref())?
        .unwrap_or_else(|| current.1.clone());
    let description = input.description.or_else(|| current.2.clone());
    let status = input.status.as_deref().unwrap_or(&current.3);
    if !matches!(status, "active" | "disabled") {
        return Err(invalid_status("tenant", status));
    }
    if name == current.1 && description == current.2 && status == current.3 {
        return Ok(Json(tenant_view(current)));
    }
    let tenant = sqlx::query_as(
        "UPDATE public.tenants SET name = $2, description = $3, status = $4, updated_at = now()
         WHERE id = $1 RETURNING id, name, description, status",
    )
    .bind(id)
    .bind(name)
    .bind(description)
    .bind(status)
    .fetch_one(&mut *tx)
    .await
    .map_err(|error| conflict(error, "tenant_name_conflict"))?;
    if status != current.3 {
        sqlx::query(
            "UPDATE public.deployments SET config_revision = config_revision + 1, updated_at = now()
             WHERE tenant_id = $1",
        )
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(storage)?;
    }
    tx.commit().await.map_err(storage)?;
    Ok(Json(tenant_view(tenant)))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NewDeployment {
    tenant_id: Uuid,
    api_base_url: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeploymentChange {
    tenant_id: Option<Uuid>,
    api_base_url: Option<String>,
    status: Option<String>,
}

fn canonical_input(input: &str, root: &str) -> Option<String> {
    if !(8..=2_048).contains(&input.chars().count()) {
        return None;
    }
    let raw = input.trim();
    if raw.chars().any(|ch| ch.is_control() || ch == '\\') {
        return None;
    }
    let (_, authority_and_path) = raw.split_once("://")?;
    let slash = authority_and_path.find('/')?;
    let (authority, raw_path) = authority_and_path.split_at(slash);
    // Inspect the raw path before the URL parser normalises dot segments.
    if authority.is_empty() || authority.contains('@') || raw_path.trim_end_matches('/') != root {
        return None;
    }
    let mut url = Url::parse(raw).ok()?;
    if let Some(Host::Domain(domain)) = url.host() {
        let normalised = domain.trim_end_matches('.').to_owned();
        if normalised.is_empty() {
            return None;
        }
        url.set_host(Some(&normalised)).ok()?;
    }
    url.set_path(root);
    canonical_url(url.as_str(), root).then(|| url.into())
}

fn deployment_url(registry: &Registry, value: &str) -> Result<String> {
    canonical_input(value, &registry.0.config.api_base_path)
        .ok_or_else(|| rejection(StatusCode::UNPROCESSABLE_ENTITY, "invalid_api_base_url"))
}

async fn require_tenant(db: &mut PgConnection, id: Uuid) -> Result<()> {
    sqlx::query_scalar::<_, Uuid>("SELECT id FROM public.tenants WHERE id = $1 FOR KEY SHARE")
        .bind(id)
        .fetch_optional(db)
        .await
        .map_err(storage)?
        .map(|_| ())
        .ok_or_else(|| not_found("tenant"))
}

async fn list_deployments(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>> {
    read_access(&state, scope, &headers).await?;
    let deployments = sqlx::query_as(
        "SELECT id, tenant_id, api_base_url, config_revision, status
         FROM public.deployments ORDER BY created_at, id",
    )
    .fetch_all(&state.db)
    .await
    .map_err(storage)?;
    Ok(Json(deployments.into_iter().map(deployment_view).collect()))
}

async fn create_deployment(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(registry): Extension<Registry>,
    headers: HeaderMap,
    Json(input): Json<NewDeployment>,
) -> Result<impl IntoResponse> {
    let mut tx = write_access(&state, scope, &headers).await?;
    require_tenant(&mut tx, input.tenant_id).await?;
    let origin = deployment_url(&registry, &input.api_base_url)?;
    let deployment = sqlx::query_as(
        "INSERT INTO public.deployments (id, tenant_id, api_base_url) VALUES ($1, $2, $3)
         RETURNING id, tenant_id, api_base_url, config_revision, status",
    )
    .bind(Uuid::new_v4())
    .bind(input.tenant_id)
    .bind(origin)
    .fetch_one(&mut *tx)
    .await
    .map_err(|error| conflict(error, "api_base_url_conflict"))?;
    tx.commit().await.map_err(storage)?;
    Ok((StatusCode::CREATED, Json(deployment_view(deployment))))
}

async fn update_deployment(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(registry): Extension<Registry>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(input): Json<DeploymentChange>,
) -> Result<Json<Value>> {
    let mut tx = write_access(&state, scope, &headers).await?;
    let current: Deployment = sqlx::query_as(
        "SELECT id, tenant_id, api_base_url, config_revision, status
         FROM public.deployments WHERE id = $1 FOR UPDATE",
    )
    .bind(id)
    .fetch_optional(&mut *tx)
    .await
    .map_err(storage)?
    .ok_or_else(|| not_found("deployment"))?;
    let tenant = input.tenant_id.unwrap_or(current.1);
    if tenant != current.1 {
        require_tenant(&mut tx, tenant).await?;
    }
    let origin = input
        .api_base_url
        .as_deref()
        .map(|origin| deployment_url(&registry, origin))
        .transpose()?
        .unwrap_or_else(|| current.2.clone());
    let status = input.status.as_deref().unwrap_or(&current.4);
    if !matches!(status, "active" | "revoked") {
        return Err(invalid_status("deployment", status));
    }
    if tenant == current.1 && origin == current.2 && status == current.4 {
        return Ok(Json(deployment_view(current)));
    }
    let deployment = sqlx::query_as(
        "UPDATE public.deployments SET tenant_id = $2, api_base_url = $3, status = $4,
            config_revision = config_revision + 1, updated_at = now()
         WHERE id = $1 RETURNING id, tenant_id, api_base_url, config_revision, status",
    )
    .bind(id)
    .bind(tenant)
    .bind(origin)
    .bind(status)
    .fetch_one(&mut *tx)
    .await
    .map_err(|error| conflict(error, "api_base_url_conflict"))?;
    tx.commit().await.map_err(storage)?;
    Ok(Json(deployment_view(deployment)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalises_exact_root_while_rejecting_alternate_paths_and_credentials() {
        let root = "/hazard/api";
        assert_eq!(
            canonical_input(" HTTPS://Example.COM.:443/hazard/api/// ", root).as_deref(),
            Some("https://example.com/hazard/api"),
        );
        assert_eq!(
            canonical_input("https://[2001:db8::1]:8443/hazard/api", root).as_deref(),
            Some("https://[2001:db8::1]:8443/hazard/api"),
        );
        for invalid in [
            "http://example.com/hazard/api",
            "https://example.com/../hazard/api",
            "https://example.com/hazard/api?foo=bar",
            "https://example.com/hazard/api#fragment",
            "https://user@example.com/hazard/api",
            "https://example.com/hazard/%61pi",
        ] {
            assert!(canonical_input(invalid, root).is_none(), "{invalid}");
        }
    }

    #[test]
    fn requires_the_named_operator_and_either_admin_role() {
        for (username, role, allowed) in [
            ("changdar", "admin", true),
            ("ChangDar", "super_admin", true),
            ("another-admin", "super_admin", false),
            ("ChangDar", "user", false),
        ] {
            assert_eq!(
                operator(Account {
                    id: 1,
                    username: username.into(),
                    group_id: None,
                    role: role.into(),
                })
                .is_ok(),
                allowed
            );
        }
    }
}

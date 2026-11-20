use axum::{
    Extension, Json, Router,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::get,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    ApiError, AppState,
    auth::Grant,
    tenancy::{Account, DeploymentScope, database_unavailable, private_response},
};

mod features;
mod groups;
mod lifecycle;
mod models;
mod site_media;
mod sites;
mod streams;
mod users_admin;
mod website_access;

pub(crate) use lifecycle::keycloak::Keycloak;
pub(crate) use site_media::start as start_media_cleanup;
pub use website_access::controller as website_access_gateway;
pub(crate) use website_access::website_client_ip;

const SUPER_ADMIN_USERNAME: &str = "ChangDar";

pub async fn routes(
    scope: DeploymentScope,
    db: &PgPool,
    shutdown: CancellationToken,
    run_worker: bool,
) -> anyhow::Result<Router<AppState>> {
    let lifecycle = lifecycle::Lifecycle::start(db, scope, shutdown, run_worker).await?;
    let router = Router::new()
        .merge(streams::routes())
        .merge(sites::routes())
        .merge(groups::routes())
        .merge(features::routes())
        .merge(users_admin::routes())
        .merge(website_access::routes())
        .route("/me", get(current_profile))
        .route("/admin/users", get(list_users_page))
        .route("/admin/capabilities", get(capabilities))
        .route("/list_users", get(list_users));
    let router = if lifecycle.available() {
        router.merge(lifecycle::routes())
    } else {
        router
    };
    Ok(router
        .layer(Extension(scope))
        .layer(Extension(lifecycle))
        .layer(axum::middleware::map_response(private_response)))
}

pub(crate) fn is_super_admin(account: &Account) -> bool {
    account.username.eq_ignore_ascii_case(SUPER_ADMIN_USERNAME)
        && matches!(account.role.as_str(), "admin" | "super_admin")
}

fn admin_group(account: &Account) -> Result<Option<i32>, ApiError> {
    if is_super_admin(account) {
        return Ok(None);
    }
    account
        .group_id
        .filter(|_| account.role == "admin")
        .map(Some)
        .ok_or_else(|| ApiError::new(StatusCode::FORBIDDEN, "Administrator group required"))
}

async fn admin(
    state: &AppState,
    scope: DeploymentScope,
    headers: &HeaderMap,
    permission: Grant,
) -> Result<Option<i32>, ApiError> {
    let actor = state.auth.authorize(headers, permission).await?;
    let account = scope.account(&state.db, &actor.subject).await?;
    admin_group(&account)
}

async fn current_profile(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountRead).await?;
    let account = scope.account(&state.db, &actor.subject).await?;
    let profile = sqlx::query_scalar(
        "SELECT jsonb_build_object(
            'id', u.id, 'username', u.username, 'tenant_id', u.tenant_id,
            'role', u.role, 'group_id', u.group_id, 'status', u.status,
            'display_name', COALESCE(NULLIF(trim(concat_ws(' ', p.given_name, p.family_name)), ''), u.username),
            'feature_names', ARRAY(SELECT f.feature_name FROM group_features gf JOIN features f
                ON gf.feature_id = f.id WHERE gf.group_id = u.group_id ORDER BY f.feature_name))
         FROM users u LEFT JOIN user_profiles p ON p.user_id = u.id
         WHERE u.id = $1 AND u.tenant_id = $2 AND u.status = 'active' AND NOT u.deletion_pending",
    )
    .bind(account.id)
    .bind(scope.tenant_id)
    .fetch_optional(&state.db)
    .await
    .map_err(database_unavailable)?
    .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, "invalid_user"))?;
    Ok(Json(profile))
}

const USER_JSON: &str = "jsonb_build_object(
        'id', u.id, 'username', u.username, 'role', u.role, 'status', u.status,
        'email_verified_at', u.email_verified_at, 'group_id', u.group_id,
        'created_at', u.created_at, 'updated_at', u.updated_at,
        'group', CASE WHEN g.id IS NOT NULL THEN jsonb_build_object('id', g.id, 'name', g.name, 'uniform_number', g.uniform_number) END,
        'group_name', g.name,
        'profile', CASE WHEN p.user_id IS NOT NULL THEN jsonb_build_object(
            'family_name', p.family_name, 'middle_name', p.middle_name, 'given_name', p.given_name,
            'email', p.email, 'mobile_number', p.mobile_number,
            'created_at', p.created_at, 'updated_at', p.updated_at) END
    )";

async fn users(
    state: &AppState,
    tenant_id: Uuid,
    group_id: Option<i32>,
    cursor: i32,
    limit: Option<i64>,
) -> Result<Vec<Value>, ApiError> {
    let query = format!("SELECT {USER_JSON} FROM users u LEFT JOIN group_info g ON g.id=u.group_id
        LEFT JOIN user_profiles p ON p.user_id=u.id
        WHERE u.tenant_id=$1 AND ($2::int IS NULL OR u.group_id=$2) AND u.id>$3 ORDER BY u.id LIMIT $4");
    sqlx::query_scalar(&query)
        .bind(tenant_id)
        .bind(group_id)
        .bind(cursor)
        .bind(limit)
        .fetch_all(&state.db)
        .await
        .map_err(database_unavailable)
}

#[derive(Deserialize)]
struct UserPage {
    cursor: Option<i32>,
    #[serde(default = "default_page_size")]
    page_size: usize,
}

fn default_page_size() -> usize {
    50
}

async fn list_users_page(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Query(page): Query<UserPage>,
) -> Result<Json<Value>, ApiError> {
    let cursor = page.cursor.unwrap_or_default();
    if cursor < 0 || !(1..=100).contains(&page.page_size) {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid user page",
        ));
    }
    let group_id = admin(&state, scope, &headers, Grant::AccountRead).await?;
    let mut items = users(
        &state,
        scope.tenant_id,
        group_id,
        cursor,
        Some(page.page_size as i64 + 1),
    )
    .await?;
    let next_cursor = if items.len() > page.page_size {
        items.pop();
        items.last().and_then(|item| item["id"].as_i64())
    } else {
        None
    };
    Ok(Json(json!({"items": items, "next_cursor": next_cursor})))
}

async fn list_users(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>, ApiError> {
    let group_id = admin(&state, scope, &headers, Grant::AccountRead).await?;
    Ok(Json(
        users(&state, scope.tenant_id, group_id, 0, None).await?,
    ))
}

async fn capabilities(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    Extension(lifecycle): Extension<lifecycle::Lifecycle>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let group_id = admin(&state, scope, &headers, Grant::AccountRead).await?;
    let platform_admin = group_id.is_none();
    Ok(Json(json!({
        "scope": if platform_admin {"all_groups"} else {"own_group"},
        "managed_group_id": group_id,
        "can_create_users": lifecycle.available(),
        "can_reset_passwords": lifecycle.available(),
        "can_suspend_users": lifecycle.available(),
        "can_delete_users": lifecycle.available(),
        "can_manage_groups": platform_admin,
        "can_manage_group_features": platform_admin,
        "can_assign_group_admins": platform_admin,
        "can_manage_website_access": platform_admin
    })))
}

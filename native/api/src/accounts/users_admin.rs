use axum::{
    Extension, Json, Router,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    routing::{get, patch, put},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::PgConnection;

use crate::{
    ApiError, AppState,
    auth::Grant,
    tenancy::{Account, DeploymentScope, database_unavailable},
};

use super::{
    SUPER_ADMIN_USERNAME, USER_JSON, UserPage, admin_group, models::resource_options,
    sites::require_group,
};

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/users/assignable-roles", get(assignable_roles))
        .route("/list_pending_users", get(list_pending))
        .route("/approve_user_signup", put(approve))
        .route("/admin/users/{user_id}/approval", patch(review))
        .route("/update_user_role", put(update_role))
        .route("/update_user_group", put(update_group))
}

const USER_NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "User not found");
const GROUP_DENIED: ApiError = ApiError::new(
    StatusCode::FORBIDDEN,
    "Cannot manage users outside your group.",
);
const DELETION_PENDING: ApiError = ApiError::new(StatusCode::CONFLICT, "account_deletion_pending");
const NOT_APPROVABLE: ApiError = ApiError::new(
    StatusCode::BAD_REQUEST,
    "User is not awaiting signup approval.",
);
const TENANT_DENIED: ApiError =
    ApiError::new(StatusCode::FORBIDDEN, "Cannot manage another tenant");

pub(super) struct Target {
    pub(super) username: String,
    pub(super) role: String,
    pub(super) group_id: Option<i32>,
    pub(super) tenant_id: uuid::Uuid,
    pub(super) deletion_pending: bool,
}

pub(super) async fn user_target(
    db: &mut PgConnection,
    id: i32,
    lock: bool,
) -> Result<Target, ApiError> {
    find_user_target(db, id, lock).await?.ok_or(USER_NOT_FOUND)
}

pub(super) async fn find_user_target(
    db: &mut PgConnection,
    id: i32,
    lock: bool,
) -> Result<Option<Target>, ApiError> {
    let sql = if lock {
        "SELECT username,role,group_id,tenant_id,deletion_pending FROM users WHERE id=$1 FOR UPDATE"
    } else {
        "SELECT username,role,group_id,tenant_id,deletion_pending FROM users WHERE id=$1"
    };
    let row = sqlx::query_as(sql)
        .bind(id)
        .fetch_optional(db)
        .await
        .map_err(database_unavailable)?;
    Ok(row.map(
        |(username, role, group_id, tenant_id, deletion_pending)| Target {
            username,
            role,
            group_id,
            tenant_id,
            deletion_pending,
        },
    ))
}

pub(super) fn roles_for(group: Option<i32>, target: Option<&Target>) -> &'static [&'static str] {
    if target.is_some_and(|target| {
        target.username.eq_ignore_ascii_case(SUPER_ADMIN_USERNAME)
            || (group.is_some() && matches!(target.role.as_str(), "admin" | "super_admin"))
    }) {
        return &[];
    }
    if group.is_none() {
        &["admin", "user", "guest"]
    } else {
        &["user", "guest"]
    }
}

fn target_in_group(
    group: Option<i32>,
    scope: DeploymentScope,
    target: &Target,
) -> Result<(), ApiError> {
    if target.tenant_id != scope.tenant_id {
        return Err(TENANT_DENIED);
    }
    if group.is_some() && group != target.group_id {
        return Err(GROUP_DENIED);
    }
    Ok(())
}

fn target_mutable(target: &Target) -> Result<(), ApiError> {
    if target.deletion_pending {
        return Err(DELETION_PENDING);
    }
    if target.username.eq_ignore_ascii_case(SUPER_ADMIN_USERNAME) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Cannot operate on super admin",
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
struct RoleQuery {
    target_user_id: Option<i32>,
    #[serde(default = "default_locale")]
    locale: String,
}

fn default_locale() -> String {
    "zh-TW".to_owned()
}

async fn assignable_roles(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Query(query): Query<RoleQuery>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountRead).await?;
    let mut db = state.db.acquire().await.map_err(database_unavailable)?;
    let account = scope.account(&mut *db, &actor.subject).await?;
    let group = admin_group(&account)?;
    let target = if let Some(id) = query.target_user_id {
        if id <= 0 {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Invalid user identifier",
            ));
        }
        let target = user_target(&mut db, id, false).await?;
        target_in_group(group, scope, &target)?;
        Some(target)
    } else {
        None
    };
    let chinese = query.locale.to_ascii_lowercase().starts_with("zh");
    let options = roles_for(group, target.as_ref())
        .iter()
        .map(|&role| {
            let label = match (chinese, role) {
                (true, "admin") => "管理員",
                (true, "user") => "使用者",
                (true, "guest") => "訪客",
                _ => role,
            };
            json!({"id": role, "label": label})
        })
        .collect();
    Ok(Json(resource_options(
        &scope.tenant_id.to_string(),
        &account,
        json!(query.target_user_id),
        options,
        Value::Null,
        Value::Null,
    )))
}

#[derive(Deserialize)]
struct UserRole {
    user_id: i32,
    new_role: String,
}

async fn update_role(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<UserRole>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let target = user_target(&mut tx, body.user_id, true).await?;
    target_in_group(group, scope, &target)?;
    target_mutable(&target)?;
    if !roles_for(group, Some(&target)).contains(&body.new_role.as_str()) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Role assignment is not allowed",
        ));
    }
    sqlx::query("UPDATE users SET role=$2,updated_at=now() WHERE id=$1")
        .bind(body.user_id)
        .bind(body.new_role)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(json!({"message": "User role updated successfully."})))
}

#[derive(Deserialize)]
struct UserGroup {
    user_id: i32,
    new_group_id: i32,
}

pub(super) fn can_manage_target(
    group: Option<i32>,
    scope: DeploymentScope,
    target: &Target,
) -> Result<(), ApiError> {
    can_manage_target_scope(group, scope, target)?;
    if target.deletion_pending {
        return Err(DELETION_PENDING);
    }
    Ok(())
}

pub(super) fn can_manage_target_scope(
    group: Option<i32>,
    scope: DeploymentScope,
    target: &Target,
) -> Result<(), ApiError> {
    target_in_group(group, scope, target)?;
    if target.username.eq_ignore_ascii_case(SUPER_ADMIN_USERNAME) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Cannot operate on super admin",
        ));
    }
    if group.is_some() && matches!(target.role.as_str(), "admin" | "super_admin") {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Only super admin can manage administrator accounts.",
        ));
    }
    Ok(())
}

async fn update_group(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<UserGroup>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let target = user_target(&mut tx, body.user_id, true).await?;
    can_manage_target(group, scope, &target)?;
    if group.is_some_and(|id| id != body.new_group_id) {
        return Err(GROUP_DENIED);
    }
    require_group(&mut tx, body.new_group_id).await?;
    sqlx::query("UPDATE users SET group_id=$2,updated_at=now() WHERE id=$1")
        .bind(body.user_id)
        .bind(body.new_group_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(json!({"message": "User group updated successfully."})))
}

async fn list_pending(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Query(page): Query<UserPage>,
) -> Result<Json<Value>, ApiError> {
    super::admin(&state, scope, &headers, Grant::AccountRead).await?;
    let cursor = page.cursor.unwrap_or_default();
    if cursor < 0 || !(1..=100).contains(&page.page_size) {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid user page",
        ));
    }
    let sql = format!(
        "SELECT {USER_JSON} || jsonb_build_object('email',p.email,'terms_version',c.terms_version,
           'privacy_version',c.privacy_version,'ai_terms_version',c.ai_terms_version,
           'notification_consent',c.notification_consent,'provider',COALESCE((
             SELECT string_agg(DISTINCT provider, ',' ORDER BY provider) FROM user_identities WHERE user_id=u.id
           ),'password'))
         FROM users u LEFT JOIN group_info g ON g.id=u.group_id LEFT JOIN user_profiles p ON p.user_id=u.id
         LEFT JOIN LATERAL (SELECT terms_version,privacy_version,ai_terms_version,notification_consent FROM user_consents
             WHERE user_id=u.id ORDER BY accepted_at DESC,id DESC LIMIT 1) c ON true
         WHERE u.tenant_id=$1 AND u.role='user' AND u.status='pending_admin_approval'
            AND u.email_verified_at IS NOT NULL AND u.group_id IS NULL AND NOT u.deletion_pending AND u.id>$2
         ORDER BY u.id LIMIT $3"
    );
    let mut items: Vec<Value> = sqlx::query_scalar(&sql)
        .bind(scope.tenant_id)
        .bind(cursor)
        .bind(page.page_size as i64 + 1)
        .fetch_all(&state.db)
        .await
        .map_err(database_unavailable)?;
    let next_cursor = if items.len() > page.page_size {
        items.pop();
        items.last().and_then(|item| item["id"].as_i64())
    } else {
        None
    };
    Ok(Json(json!({"items": items, "next_cursor": next_cursor})))
}

#[derive(Deserialize)]
struct Approval {
    user_id: i32,
    group_id: Option<i32>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Decision {
    #[default]
    Approved,
    Rejected,
}

#[derive(Deserialize)]
struct Review {
    #[serde(default)]
    decision: Decision,
    group_id: Option<i32>,
}

async fn approve(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Approval>,
) -> Result<Json<Value>, ApiError> {
    decide(
        &state,
        scope,
        &headers,
        body.user_id,
        body.group_id,
        Decision::Approved,
    )
    .await
}

async fn review(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Path(user): Path<i32>,
    Json(body): Json<Review>,
) -> Result<Json<Value>, ApiError> {
    decide(&state, scope, &headers, user, body.group_id, body.decision).await
}

async fn decide(
    state: &AppState,
    scope: DeploymentScope,
    headers: &HeaderMap,
    user: i32,
    requested_group: Option<i32>,
    decision: Decision,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account: Account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let target = user_target(&mut tx, user, true).await?;
    target_in_group(None, scope, &target)?;
    target_mutable(&target)?;
    let (status, target_group) = match decision {
        Decision::Approved => {
            if group.is_some() && requested_group.is_some() && group != requested_group {
                return Err(GROUP_DENIED);
            }
            let id = requested_group
                .or(group)
                .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "group_id is required."))?;
            require_group(&mut tx, id).await?;
            ("active", Some(id))
        }
        Decision::Rejected => ("rejected", None),
    };
    let changed = sqlx::query(
        "UPDATE users SET status=$2,group_id=$3,updated_at=now() WHERE id=$1 AND role='user'
         AND status='pending_admin_approval' AND email_verified_at IS NOT NULL AND group_id IS NULL",
    ).bind(user).bind(status).bind(target_group).execute(&mut *tx).await.map_err(database_unavailable)?;
    if changed.rows_affected() == 0 {
        return Err(NOT_APPROVABLE);
    }
    if let Some(id) = target_group {
        sqlx::query(
            "INSERT INTO site_notification_preferences (user_id,site_id,is_enabled,updated_at)
             SELECT $1,site_id,true,now() FROM site_groups WHERE group_id=$2 ON CONFLICT DO NOTHING",
        ).bind(user).bind(id).execute(&mut *tx).await.map_err(database_unavailable)?;
    }
    let sql = format!(
        "SELECT {USER_JSON} FROM users u LEFT JOIN group_info g ON g.id=u.group_id
        LEFT JOIN user_profiles p ON p.user_id=u.id WHERE u.id=$1"
    );
    let result = sqlx::query_scalar(&sql)
        .bind(user)
        .fetch_one(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(result))
}

use axum::{
    Extension, Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{delete, get, post, put},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::PgConnection;

use crate::{
    ApiError, AppState,
    auth::Grant,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{SUPER_ADMIN_USERNAME, admin, admin_group};

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/list_sites", get(list_sites))
        .route("/create_site", post(create_site))
        .route("/update_site", put(update_site))
        .route("/delete_site", delete(delete_site))
        .route("/add_group_to_site", post(add_group))
        .route("/remove_group_from_site", post(remove_group))
        .route("/add_user_to_site", post(add_user))
        .route("/remove_user_from_site", post(remove_user))
}

#[derive(Deserialize)]
struct Create {
    name: String,
    #[serde(default)]
    group_ids: Vec<i32>,
}

#[derive(Deserialize)]
struct Update {
    site_id: i32,
    new_name: String,
}

#[derive(Deserialize)]
struct Delete {
    site_id: i32,
}

#[derive(Deserialize)]
struct GroupMember {
    site_id: i32,
    group_id: i32,
}

#[derive(Deserialize)]
struct UserMember {
    site_id: i32,
    user_id: i32,
}

const SITE_NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "Site not found.");
const GROUP_NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "Group not found.");
const USER_NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "User not found.");
const WRONG_GROUP: ApiError = ApiError::new(StatusCode::FORBIDDEN, "Cannot operate on other group");

fn validate_name(name: &str) -> Result<(), ApiError> {
    if !(1..=80).contains(&name.chars().count()) || name.trim().is_empty() || name.contains('\0') {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid site name.",
        ));
    }
    Ok(())
}

fn site_constraint(error: sqlx::Error) -> ApiError {
    if error
        .as_database_error()
        .is_some_and(|error| error.is_unique_violation())
    {
        ApiError::new(StatusCode::CONFLICT, "Site name already exists.")
    } else {
        database_unavailable(error)
    }
}

async fn permitted_site(
    db: &mut PgConnection,
    site_id: i32,
    group: Option<i32>,
) -> Result<(String, Vec<i32>), ApiError> {
    let name = lock_site(db, site_id).await?;
    let groups: Vec<i32> =
        sqlx::query_scalar("SELECT group_id FROM site_groups WHERE site_id=$1 ORDER BY group_id")
            .bind(site_id)
            .fetch_all(db)
            .await
            .map_err(database_unavailable)?;
    if group.is_some_and(|group| !groups.contains(&group)) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Cannot manage other group's site",
        ));
    }
    Ok((name, groups))
}

pub(super) async fn lock_site(db: &mut PgConnection, site_id: i32) -> Result<String, ApiError> {
    sqlx::query_scalar("SELECT name FROM sites WHERE id=$1 FOR UPDATE")
        .bind(site_id)
        .fetch_optional(db)
        .await
        .map_err(database_unavailable)?
        .ok_or(SITE_NOT_FOUND)
}

pub(super) async fn require_group(db: &mut PgConnection, group: i32) -> Result<(), ApiError> {
    sqlx::query_scalar::<_, i32>("SELECT id FROM group_info WHERE id=$1 FOR KEY SHARE")
        .bind(group)
        .fetch_optional(db)
        .await
        .map_err(database_unavailable)?
        .ok_or(GROUP_NOT_FOUND)?;
    Ok(())
}

async fn site_read(
    db: &mut PgConnection,
    scope: DeploymentScope,
    group: Option<i32>,
    site: Option<i32>,
) -> Result<Vec<Value>, ApiError> {
    sqlx::query_scalar(
        "SELECT jsonb_build_object('id', s.id, 'name', s.name,
            'group_ids', ARRAY(SELECT g.id FROM site_groups sg JOIN group_info g ON g.id=sg.group_id
                WHERE sg.site_id=s.id AND ($2::int IS NULL OR g.id=$2) ORDER BY g.id),
            'group_names', ARRAY(SELECT g.name FROM site_groups sg JOIN group_info g ON g.id=sg.group_id
                WHERE sg.site_id=s.id AND ($2::int IS NULL OR g.id=$2) ORDER BY g.id),
            'user_ids', ARRAY(SELECT u.id FROM user_sites us JOIN users u ON u.id=us.user_id
                WHERE us.site_id=s.id AND u.tenant_id=$1 AND ($2::int IS NULL OR u.group_id=$2) ORDER BY u.id))
         FROM sites s WHERE ($3::int IS NULL OR s.id=$3)
           AND ($2::int IS NULL OR EXISTS (SELECT 1 FROM site_groups sg WHERE sg.site_id=s.id AND sg.group_id=$2))
         ORDER BY s.id",
    ).bind(scope.tenant_id).bind(group).bind(site).fetch_all(db).await.map_err(database_unavailable)
}

async fn list_sites(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>, ApiError> {
    let group = admin(&state, scope, &headers, Grant::AccountRead).await?;
    let mut db = state.db.acquire().await.map_err(database_unavailable)?;
    Ok(Json(site_read(&mut db, scope, group, None).await?))
}

async fn seed_group_preferences(
    db: &mut PgConnection,
    scope: DeploymentScope,
    site_id: i32,
    groups: &[i32],
) -> Result<(), ApiError> {
    sqlx::query(
        "INSERT INTO site_notification_preferences (user_id,site_id,is_enabled,updated_at)
         SELECT id,$1,true,now() FROM users WHERE tenant_id=$2 AND group_id=ANY($3)
         ON CONFLICT (user_id,site_id) DO NOTHING",
    )
    .bind(site_id)
    .bind(scope.tenant_id)
    .bind(groups)
    .execute(db)
    .await
    .map_err(database_unavailable)?;
    Ok(())
}

async fn member_preference(
    db: &mut PgConnection,
    site_id: i32,
    user_id: i32,
) -> Result<(), ApiError> {
    sqlx::query(
        "INSERT INTO site_notification_preferences (user_id,site_id,is_enabled,updated_at)
         VALUES ($1,$2,true,now()) ON CONFLICT (user_id,site_id) DO NOTHING",
    )
    .bind(user_id)
    .bind(site_id)
    .execute(db)
    .await
    .map_err(database_unavailable)?;
    Ok(())
}

async fn create_site(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Create>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    validate_name(&body.name)?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let mut groups = body.group_ids;
    if let Some(group) = group {
        if groups.iter().any(|&id| id != group) {
            return Err(WRONG_GROUP);
        }
        groups = vec![group];
    }
    if groups.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "group_id is required for new site",
        ));
    }
    groups.sort_unstable();
    groups.dedup();
    for &group_id in &groups {
        require_group(&mut tx, group_id).await?;
    }
    let id: i32 = sqlx::query_scalar(
        "INSERT INTO sites (name,created_at,updated_at) VALUES ($1,now(),now()) RETURNING id",
    )
    .bind(body.name)
    .fetch_one(&mut *tx)
    .await
    .map_err(site_constraint)?;
    sqlx::query(
        "INSERT INTO site_groups (site_id,group_id) SELECT $1,id FROM unnest($2::int[]) AS ids(id)",
    )
    .bind(id)
    .bind(&groups)
    .execute(&mut *tx)
    .await
    .map_err(database_unavailable)?;
    seed_group_preferences(&mut tx, scope, id, &groups).await?;
    let platform_admin: Option<i32> = sqlx::query_scalar(
        "SELECT id FROM users WHERE tenant_id=$1 AND lower(username)=lower($2) AND role IN ('admin','super_admin')
         AND status='active' AND NOT deletion_pending ORDER BY id LIMIT 1",
    ).bind(scope.tenant_id).bind(SUPER_ADMIN_USERNAME).fetch_optional(&mut *tx).await.map_err(database_unavailable)?;
    if let Some(user_id) = platform_admin {
        sqlx::query(
            "INSERT INTO user_sites (user_id,site_id) VALUES ($1,$2) ON CONFLICT DO NOTHING",
        )
        .bind(user_id)
        .bind(id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
        member_preference(&mut tx, id, user_id).await?;
    }
    let response = site_read(&mut tx, scope, group, Some(id))
        .await?
        .pop()
        .ok_or(SITE_NOT_FOUND)?;
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(response))
}

async fn update_site(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Update>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    validate_name(&body.new_name)?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let (old_name, _) = permitted_site(&mut tx, body.site_id, admin_group(&account)?).await?;
    sqlx::query("UPDATE sites SET name=$2,updated_at=now() WHERE id=$1")
        .bind(body.site_id)
        .bind(&body.new_name)
        .execute(&mut *tx)
        .await
        .map_err(site_constraint)?;
    if old_name != body.new_name {
        state.metadata.invalidate_site_generation(&old_name).await?;
    }
    tx.commit().await.map_err(database_unavailable)?;
    state
        .metadata
        .invalidate_site_notification_cache(&[&old_name, &body.new_name])
        .await;
    Ok(Json(json!({"message": "Site updated successfully."})))
}

async fn delete_site(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Delete>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let (name, _) = permitted_site(&mut tx, body.site_id, admin_group(&account)?).await?;
    sqlx::query(
        "INSERT INTO site_media_cleanup_jobs (path) SELECT DISTINCT image_path FROM violations
         WHERE image_path IS NOT NULL AND (site_id=$1 OR (site_id IS NULL AND site=$2))
         ON CONFLICT (path) DO UPDATE SET attempt_count=0,last_error=NULL,completed_at=NULL,lease_token=NULL,lease_expires_at=NULL",
    ).bind(body.site_id).bind(&name).execute(&mut *tx).await.map_err(database_unavailable)?;
    // The installed schema has historical associations without cascades.
    sqlx::query("DELETE FROM user_sites WHERE site_id=$1")
        .bind(body.site_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    sqlx::query("DELETE FROM stream_configs WHERE site_id=$1")
        .bind(body.site_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    sqlx::query("DELETE FROM sites WHERE id=$1")
        .bind(body.site_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    state.metadata.invalidate_site_generation(&name).await?;
    tx.commit().await.map_err(database_unavailable)?;
    state
        .metadata
        .invalidate_site_notification_cache(&[&name])
        .await;
    Ok(Json(
        json!({"message": "Site and related data deleted successfully."}),
    ))
}

async fn add_group(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<GroupMember>,
) -> Result<Json<Value>, ApiError> {
    change_group(&state, scope, &headers, body, true).await
}

async fn remove_group(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<GroupMember>,
) -> Result<Json<Value>, ApiError> {
    change_group(&state, scope, &headers, body, false).await
}

async fn change_group(
    state: &AppState,
    scope: DeploymentScope,
    headers: &HeaderMap,
    member: GroupMember,
    add: bool,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let (name, _) = permitted_site(&mut tx, member.site_id, group).await?;
    if group.is_some_and(|expected| expected != member.group_id) {
        return Err(WRONG_GROUP);
    }
    let message = if add {
        require_group(&mut tx, member.group_id).await?;
        sqlx::query(
            "INSERT INTO site_groups (site_id,group_id) VALUES ($1,$2) ON CONFLICT DO NOTHING",
        )
        .bind(member.site_id)
        .bind(member.group_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
        seed_group_preferences(&mut tx, scope, member.site_id, &[member.group_id]).await?;
        "Group linked to site successfully."
    } else {
        sqlx::query("DELETE FROM site_groups WHERE site_id=$1 AND group_id=$2")
            .bind(member.site_id)
            .bind(member.group_id)
            .execute(&mut *tx)
            .await
            .map_err(database_unavailable)?;
        sqlx::query(
            "DELETE FROM site_notification_preferences p USING users u
             WHERE p.user_id=u.id AND u.tenant_id=$3 AND p.site_id=$1 AND u.group_id=$2
                AND NOT EXISTS (SELECT 1 FROM user_sites us WHERE us.site_id=$1 AND us.user_id=p.user_id)",
        ).bind(member.site_id).bind(member.group_id).bind(scope.tenant_id).execute(&mut *tx).await.map_err(database_unavailable)?;
        "Group unlinked from site successfully."
    };
    tx.commit().await.map_err(database_unavailable)?;
    state
        .metadata
        .invalidate_site_notification_cache(&[&name])
        .await;
    Ok(Json(json!({"message": message})))
}

async fn add_user(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<UserMember>,
) -> Result<Json<Value>, ApiError> {
    change_user(&state, scope, &headers, body, true).await
}

async fn remove_user(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<UserMember>,
) -> Result<Json<Value>, ApiError> {
    change_user(&state, scope, &headers, body, false).await
}

async fn change_user(
    state: &AppState,
    scope: DeploymentScope,
    headers: &HeaderMap,
    member: UserMember,
    add: bool,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let (target_group, username): (Option<i32>, String) = sqlx::query_as(
        "SELECT group_id,username FROM users WHERE tenant_id=$1 AND id=$2 AND NOT deletion_pending FOR SHARE",
    ).bind(scope.tenant_id).bind(member.user_id).fetch_optional(&mut *tx).await.map_err(database_unavailable)?.ok_or(USER_NOT_FOUND)?;
    let (site, groups) = permitted_site(&mut tx, member.site_id, group).await?;
    if username.eq_ignore_ascii_case(SUPER_ADMIN_USERNAME) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Cannot modify super admin's site membership.",
        ));
    }
    if group.is_some_and(|id| Some(id) != target_group) {
        return Err(WRONG_GROUP);
    }
    let message = if add {
        if !target_group.is_some_and(|id| groups.contains(&id)) {
            return Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "User and site must belong to the same group.",
            ));
        }
        sqlx::query(
            "INSERT INTO user_sites (user_id,site_id) VALUES ($1,$2) ON CONFLICT DO NOTHING",
        )
        .bind(member.user_id)
        .bind(member.site_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
        member_preference(&mut tx, member.site_id, member.user_id).await?;
        "User linked to site successfully."
    } else {
        sqlx::query("DELETE FROM user_sites WHERE user_id=$1 AND site_id=$2")
            .bind(member.user_id)
            .bind(member.site_id)
            .execute(&mut *tx)
            .await
            .map_err(database_unavailable)?;
        sqlx::query(
            "DELETE FROM site_notification_preferences WHERE user_id=$1 AND site_id=$2
                AND NOT EXISTS (SELECT 1 FROM site_groups WHERE site_id=$2 AND group_id=$3)",
        )
        .bind(member.user_id)
        .bind(member.site_id)
        .bind(target_group)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
        "User unlinked from site successfully."
    };
    tx.commit().await.map_err(database_unavailable)?;
    state
        .metadata
        .invalidate_site_notification_cache(&[&site])
        .await;
    Ok(Json(json!({"message": message})))
}

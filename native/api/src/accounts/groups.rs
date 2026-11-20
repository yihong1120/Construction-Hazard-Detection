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

use super::{admin, is_super_admin};

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/list_groups", get(list_groups))
        .route("/create_group", post(create_group))
        .route("/update_group", put(update_group))
        .route("/delete_group", delete(delete_group))
}

#[derive(Deserialize)]
struct Create {
    name: String,
    uniform_number: String,
}

#[derive(Deserialize)]
struct Update {
    group_id: i32,
    new_name: Option<String>,
    new_uniform_number: Option<String>,
}

#[derive(Deserialize)]
struct Delete {
    group_id: i32,
}

const NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "Group not found.");
const INVALID_NUMBER: ApiError = ApiError::new(
    StatusCode::BAD_REQUEST,
    "Uniform number must be unique and exactly 8 digits.",
);

fn check_name(name: &str) -> Result<(), ApiError> {
    if name.trim().is_empty() || name.chars().count() > 100 || name.contains('\0') {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Invalid group name.",
        ));
    }
    Ok(())
}

fn check_number(number: &str) -> Result<(), ApiError> {
    if number.len() != 8 || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(INVALID_NUMBER);
    }
    Ok(())
}

fn constraint(error: sqlx::Error) -> ApiError {
    if error
        .as_database_error()
        .is_some_and(|error| error.is_unique_violation())
    {
        INVALID_NUMBER
    } else {
        database_unavailable(error)
    }
}

pub(super) async fn require_super(
    db: &mut PgConnection,
    scope: DeploymentScope,
    subject: &str,
) -> Result<i32, ApiError> {
    let operator = scope.account_for_update(db, subject).await?;
    if !is_super_admin(&operator) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Super administrator required.",
        ));
    }
    Ok(operator.id)
}

async fn list_groups(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>, ApiError> {
    let group = admin(&state, scope, &headers, Grant::AccountRead).await?;
    let output = sqlx::query_scalar(
        "SELECT jsonb_build_object('id', id, 'name', name, 'uniform_number', uniform_number)
         FROM group_info WHERE $1::int IS NULL OR id=$1 ORDER BY id",
    )
    .bind(group)
    .fetch_all(&state.db)
    .await
    .map_err(database_unavailable)?;
    Ok(Json(output))
}

async fn create_group(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Create>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    check_name(&body.name)?;
    check_number(&body.uniform_number)?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    require_super(&mut tx, scope, &actor.subject).await?;
    let created = sqlx::query_scalar(
        "INSERT INTO group_info (name,uniform_number,max_allowed_streams,updated_at) VALUES ($1,$2,8,now())
         RETURNING jsonb_build_object('id', id, 'name', name, 'uniform_number', uniform_number)",
    ).bind(body.name).bind(body.uniform_number).fetch_one(&mut *tx).await.map_err(constraint)?;
    tx.commit().await.map_err(constraint)?;
    Ok(Json(created))
}

async fn update_group(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Update>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let name = body
        .new_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let number = body
        .new_uniform_number
        .as_deref()
        .filter(|number| !number.is_empty());
    if name.is_none() && number.is_none() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "Nothing to update."));
    }
    if let Some(name) = name {
        check_name(name)?;
    }
    if let Some(number) = number {
        check_number(number)?;
    }
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    require_super(&mut tx, scope, &actor.subject).await?;
    let changed = sqlx::query(
        "UPDATE group_info SET name=COALESCE($2,name),uniform_number=COALESCE($3,uniform_number),updated_at=now()
         WHERE id=$1",
    ).bind(body.group_id).bind(name).bind(number).execute(&mut *tx).await.map_err(constraint)?;
    if changed.rows_affected() == 0 {
        return Err(NOT_FOUND);
    }
    tx.commit().await.map_err(constraint)?;
    Ok(Json(json!({"message": "Group updated successfully."})))
}

async fn delete_group(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Delete>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    require_super(&mut tx, scope, &actor.subject).await?;
    sqlx::query_scalar::<_, i32>("SELECT id FROM group_info WHERE id=$1 FOR UPDATE")
        .bind(body.group_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(database_unavailable)?
        .ok_or(NOT_FOUND)?;
    sqlx::query("DELETE FROM stream_configs WHERE group_id=$1")
        .bind(body.group_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    let changed = sqlx::query("DELETE FROM group_info WHERE id=$1")
        .bind(body.group_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    if changed.rows_affected() == 0 {
        return Err(NOT_FOUND);
    }
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(json!({"message": "Group deleted successfully."})))
}

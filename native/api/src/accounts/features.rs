use axum::{
    Extension, Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{delete, get, post, put},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    ApiError, AppState,
    auth::Grant,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{admin, groups::require_super};

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/list_features", get(list_features))
        .route("/create_feature", post(create_feature))
        .route("/update_feature", put(update_feature))
        .route("/delete_feature", delete(delete_feature))
        .route("/update_group_feature", post(update_group_features))
        .route("/list_group_features", get(list_group_features))
}

const NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "Feature not found");
const SUPER_REQUIRED: ApiError =
    ApiError::new(StatusCode::FORBIDDEN, "Super administrator required.");

fn feature_name(name: &str) -> Result<(), ApiError> {
    if name.trim().is_empty() || name.chars().count() > 50 || name.contains('\0') {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Invalid feature name.",
        ));
    }
    Ok(())
}

fn feature_constraint(error: sqlx::Error) -> ApiError {
    if error
        .as_database_error()
        .is_some_and(|error| error.is_unique_violation())
    {
        ApiError::new(StatusCode::CONFLICT, "Feature name already exists.")
    } else {
        database_unavailable(error)
    }
}

async fn list_features(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>, ApiError> {
    if admin(&state, scope, &headers, Grant::AccountRead)
        .await?
        .is_some()
    {
        return Err(SUPER_REQUIRED);
    }
    let rows = sqlx::query_scalar(
        "SELECT jsonb_build_object('id',id,'feature_name',feature_name,'description',description) FROM features ORDER BY id",
    ).fetch_all(&state.db).await.map_err(database_unavailable)?;
    Ok(Json(rows))
}

#[derive(Deserialize)]
struct Create {
    feature_name: String,
    description: Option<String>,
}

async fn create_feature(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Create>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    require_super(&mut tx, scope, &actor.subject).await?;
    feature_name(&body.feature_name)?;
    let row = sqlx::query_scalar(
        "INSERT INTO features (feature_name,description,updated_at) VALUES ($1,$2,now())
         RETURNING jsonb_build_object('id',id,'feature_name',feature_name,'description',description)",
    ).bind(body.feature_name).bind(body.description).fetch_one(&mut *tx).await.map_err(feature_constraint)?;
    tx.commit().await.map_err(feature_constraint)?;
    Ok(Json(row))
}

#[derive(Deserialize)]
struct Update {
    feature_id: i32,
    new_name: Option<String>,
    new_description: Option<String>,
}

async fn update_feature(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Update>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    require_super(&mut tx, scope, &actor.subject).await?;
    if let Some(name) = &body.new_name {
        feature_name(name)?;
    }
    if body.new_name.is_none() && body.new_description.is_none() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "Nothing to update"));
    }
    let result = sqlx::query(
        "UPDATE features SET feature_name=COALESCE($2,feature_name), description=COALESCE($3,description),updated_at=now() WHERE id=$1",
    ).bind(body.feature_id).bind(body.new_name).bind(body.new_description).execute(&mut *tx).await.map_err(feature_constraint)?;
    if result.rows_affected() == 0 {
        return Err(NOT_FOUND);
    }
    tx.commit().await.map_err(feature_constraint)?;
    Ok(Json(json!({"message":"Feature updated successfully."})))
}

#[derive(Deserialize)]
struct Delete {
    feature_id: i32,
}

async fn delete_feature(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Delete>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    require_super(&mut tx, scope, &actor.subject).await?;
    let result = sqlx::query("DELETE FROM features WHERE id=$1")
        .bind(body.feature_id)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    if result.rows_affected() == 0 {
        return Err(NOT_FOUND);
    }
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(json!({"message":"Feature deleted successfully."})))
}

#[derive(Deserialize)]
struct Assign {
    group_id: i32,
    feature_ids: Vec<i32>,
}

async fn update_group_features(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(body): Json<Assign>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    require_super(&mut tx, scope, &actor.subject).await?;
    sqlx::query_scalar::<_, i32>("SELECT id FROM group_info WHERE id=$1 FOR UPDATE")
        .bind(body.group_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(database_unavailable)?
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "Group not found"))?;
    sqlx::query("DELETE FROM group_features WHERE group_id=$1 AND NOT(feature_id=ANY($2))")
        .bind(body.group_id)
        .bind(&body.feature_ids)
        .execute(&mut *tx)
        .await
        .map_err(database_unavailable)?;
    sqlx::query(
        "INSERT INTO group_features (group_id,feature_id)
         SELECT $1,id FROM features WHERE id=ANY($2) ON CONFLICT DO NOTHING",
    )
    .bind(body.group_id)
    .bind(body.feature_ids)
    .execute(&mut *tx)
    .await
    .map_err(database_unavailable)?;
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(
        json!({"message":"Group features updated successfully."}),
    ))
}

async fn list_group_features(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>, ApiError> {
    if admin(&state, scope, &headers, Grant::AccountRead)
        .await?
        .is_some()
    {
        return Err(SUPER_REQUIRED);
    }
    let rows = sqlx::query_scalar(
        "SELECT jsonb_build_object('group_id',g.id,'group_name',g.name,'feature_ids',
           ARRAY(SELECT feature_id FROM group_features WHERE group_id=g.id ORDER BY feature_id))
         FROM group_info g ORDER BY g.id",
    )
    .fetch_all(&state.db)
    .await
    .map_err(database_unavailable)?;
    Ok(Json(rows))
}

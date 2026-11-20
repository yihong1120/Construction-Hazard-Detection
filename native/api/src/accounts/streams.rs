use std::collections::{HashMap, HashSet};

use axum::{
    Extension, Json, Router,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    routing::{delete, get, post, put},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::PgConnection;

use crate::{
    ApiError, AppState,
    auth::Grant,
    tenancy::{Account, DeploymentScope, database_unavailable},
};

use super::{
    admin_group,
    models::{Catalog, ResourceError},
    sites::lock_site,
};

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/sites/{site_id}/stream-config",
            get(list_streams).put(upsert_streams),
        )
        .route("/sites/{site_id}/stream-models", get(list_models))
        .route("/create_stream_config", post(create_stream))
        .route("/stream_config/update/{cfg_id}", put(update_stream))
        .route("/delete_stream_config/{cfg_id}", delete(delete_stream))
        .route("/group_stream_limit", get(group_limit))
}

#[derive(Deserialize, Serialize)]
struct Settings {
    stream_name: String,
    video_url: String,
    #[serde(default = "default_model")]
    model_key: String,
    #[serde(default = "enabled")]
    recognition_enabled: bool,
    #[serde(default = "default_start")]
    work_start_hour: i32,
    #[serde(default = "default_end")]
    work_end_hour: i32,
    #[serde(default)]
    detect_no_safety_vest_or_helmet: bool,
    #[serde(default)]
    detect_near_machinery_or_vehicle: bool,
    #[serde(default)]
    detect_in_restricted_area: bool,
    #[serde(default)]
    detect_in_utility_pole_restricted_area: bool,
    #[serde(default)]
    detect_machinery_close_to_pole: bool,
    expire_date: Option<DateTime<Utc>>,
}

fn default_model() -> String {
    "yolo26n".to_owned()
}

fn enabled() -> bool {
    true
}

fn default_start() -> i32 {
    7
}

fn default_end() -> i32 {
    18
}

impl Settings {
    fn validate(&self) -> Result<(), ApiError> {
        if !(1..=80).contains(&self.stream_name.chars().count())
            || !(1..=255).contains(&self.video_url.chars().count())
            || !(1..=80).contains(&self.model_key.chars().count())
            || !(0..=24).contains(&self.work_start_hour)
            || !(0..=24).contains(&self.work_end_hour)
            || [&self.stream_name, &self.video_url, &self.model_key]
                .into_iter()
                .any(|field| field.trim().is_empty() || field.contains('\0'))
        {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Invalid stream configuration",
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct Create {
    site_id: i32,
    group_id: Option<i32>,
    #[serde(flatten)]
    settings: Settings,
}

#[derive(Deserialize)]
struct Item {
    id: Option<i32>,
    #[serde(flatten)]
    settings: Settings,
}

#[derive(Deserialize)]
struct Upsert {
    streams: Vec<Item>,
}

#[derive(Deserialize)]
struct GroupQuery {
    group_id: i32,
}

const SITE_NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "Site not found.");
const STREAM_NOT_FOUND: ApiError =
    ApiError::new(StatusCode::NOT_FOUND, "Stream configuration not found.");
const GROUP_NOT_FOUND: ApiError = ApiError::new(StatusCode::NOT_FOUND, "Group not found");
const GROUP_DENIED: ApiError =
    ApiError::new(StatusCode::FORBIDDEN, "Cannot operate on other group");
const NAME_EXISTS: ApiError = ApiError::new(
    StatusCode::BAD_REQUEST,
    "Stream name already exists in site.",
);

async fn site_groups(
    db: &mut PgConnection,
    account: &Account,
    group: Option<i32>,
    site_id: i32,
    membership_required: bool,
    lock: bool,
) -> Result<Vec<i32>, ApiError> {
    if lock {
        lock_site(db, site_id).await?;
    }
    let (groups, member): (Vec<i32>, bool) = sqlx::query_as(
        "SELECT ARRAY(SELECT sg.group_id FROM site_groups sg WHERE sg.site_id=s.id ORDER BY sg.group_id),
                EXISTS(SELECT 1 FROM user_sites us WHERE us.site_id=s.id AND us.user_id=$2)
         FROM sites s WHERE s.id=$1",
    )
        .bind(site_id)
        .bind(account.id)
        .fetch_optional(db)
        .await
        .map_err(database_unavailable)?
        .ok_or(SITE_NOT_FOUND)?;
    if group.is_some_and(|id| !groups.contains(&id)) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Cannot manage other group's site",
        ));
    }
    if membership_required && !member {
        return Err(ApiError::new(StatusCode::FORBIDDEN, "No access to site"));
    }
    Ok(groups)
}

fn owner_group(
    groups: &[i32],
    operator: Option<i32>,
    requested: Option<i32>,
) -> Result<i32, ApiError> {
    let owner = operator
        .or(requested)
        .or_else(|| groups.first().copied())
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "Site must have a group before configuring streams.",
            )
        })?;
    if !groups.contains(&owner) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Group is not associated with this site.",
        ));
    }
    Ok(owner)
}

const SETTINGS_JSON: &str = "jsonb_build_object(
    'stream_name',sc.stream_name,'video_url',sc.video_url,'model_key',sc.model_key,
    'recognition_enabled',sc.recognition_enabled,
    'work_start_hour',COALESCE(sc.work_start_hour,0),'work_end_hour',COALESCE(sc.work_end_hour,0),
    'detect_no_safety_vest_or_helmet',COALESCE(sc.detect_no_safety_vest_or_helmet,false),
    'detect_near_machinery_or_vehicle',COALESCE(sc.detect_near_machinery_or_vehicle,false),
    'detect_in_restricted_area',COALESCE(sc.detect_in_restricted_area,false),
    'detect_in_utility_pole_restricted_area',COALESCE(sc.detect_in_utility_pole_restricted_area,false),
    'detect_machinery_close_to_pole',COALESCE(sc.detect_machinery_close_to_pole,false),
    'expire_date',sc.expire_date)";

async fn stream_list(
    db: &mut PgConnection,
    site_id: i32,
    group: Option<i32>,
) -> Result<Vec<Value>, ApiError> {
    let query = format!(
        "WITH visible AS (
            SELECT * FROM stream_configs WHERE site_id=$1 AND ($2::int IS NULL OR group_id=$2)
         ), counts AS (
            SELECT group_id, count(*) AS current FROM stream_configs
            WHERE group_id IN (SELECT group_id FROM visible) GROUP BY group_id
         )
         SELECT {SETTINGS_JSON} || jsonb_build_object('id',sc.id,'updated_at',sc.updated_at,
            'total_stream_in_group', c.current, 'max_allowed_streams', g.max_allowed_streams)
         FROM visible sc JOIN counts c USING(group_id) JOIN group_info g ON sc.group_id=g.id ORDER BY sc.id",
    );
    sqlx::query_scalar(&query)
        .bind(site_id)
        .bind(group)
        .fetch_all(db)
        .await
        .map_err(database_unavailable)
}

async fn list_streams(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Path(site_id): Path<i32>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountRead).await?;
    let mut db = state.db.acquire().await.map_err(database_unavailable)?;
    let account = scope.account(&mut *db, &actor.subject).await?;
    let group = admin_group(&account)?;
    site_groups(&mut db, &account, group, site_id, false, false).await?;
    Ok(Json(stream_list(&mut db, site_id, group).await?))
}

async fn list_models(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Path(site_id): Path<i32>,
) -> Result<Json<Value>, ResourceError> {
    let actor = state.auth.authorize(&headers, Grant::AccountRead).await?;
    let mut db = state.db.acquire().await.map_err(database_unavailable)?;
    let account = scope.account(&mut *db, &actor.subject).await?;
    let group = admin_group(&account)?;
    site_groups(&mut db, &account, group, site_id, true, false).await?;
    let catalog = Catalog::load(&mut db).await?;
    Ok(Json(catalog.options(
        &scope.tenant_id.to_string(),
        &account,
        site_id,
    )))
}

async fn quota(
    db: &mut PgConnection,
    group: i32,
    additional: Option<usize>,
) -> Result<(i64, i32), ApiError> {
    let statement = if additional.is_some() {
        "SELECT max_allowed_streams FROM group_info WHERE id=$1 FOR UPDATE"
    } else {
        "SELECT max_allowed_streams FROM group_info WHERE id=$1"
    };
    let limit: i32 = sqlx::query_scalar(statement)
        .bind(group)
        .fetch_optional(&mut *db)
        .await
        .map_err(database_unavailable)?
        .ok_or(GROUP_NOT_FOUND)?;
    let current: i64 = sqlx::query_scalar("SELECT count(*) FROM stream_configs WHERE group_id=$1")
        .bind(group)
        .fetch_one(db)
        .await
        .map_err(database_unavailable)?;
    if additional.is_some_and(|count| current + count as i64 > i64::from(limit)) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Stream limit reached for group.",
        ));
    }
    Ok((current, limit))
}

async fn group_limit(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Query(query): Query<GroupQuery>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountRead).await?;
    let mut db = state.db.acquire().await.map_err(database_unavailable)?;
    let account = scope.account(&mut *db, &actor.subject).await?;
    if admin_group(&account)?.is_some_and(|id| id != query.group_id) {
        return Err(GROUP_DENIED);
    }
    let (current, limit) = quota(&mut db, query.group_id, None).await?;
    Ok(Json(
        json!({"group_id": query.group_id, "current_streams_count": current, "max_allowed_streams": limit}),
    ))
}

async fn owner_of_name(
    db: &mut PgConnection,
    site_id: i32,
    name: &str,
) -> Result<Option<i32>, ApiError> {
    sqlx::query_scalar("SELECT id FROM stream_configs WHERE site_id=$1 AND stream_name=$2")
        .bind(site_id)
        .bind(name)
        .fetch_optional(db)
        .await
        .map_err(database_unavailable)
}

const DECODE: &str = "jsonb_to_record($3::jsonb) AS v(
    stream_name text, video_url text, model_key text, recognition_enabled bool,
    work_start_hour int, work_end_hour int, detect_no_safety_vest_or_helmet bool,
    detect_near_machinery_or_vehicle bool, detect_in_restricted_area bool,
    detect_in_utility_pole_restricted_area bool, detect_machinery_close_to_pole bool, expire_date timestamptz)";

async fn insert_stream(
    db: &mut PgConnection,
    site: i32,
    group: i32,
    settings: &Settings,
) -> Result<i32, ApiError> {
    let statement = format!(
        "INSERT INTO stream_configs (site_id, group_id, stream_name, video_url, model_key,
         recognition_enabled, work_start_hour, work_end_hour, detect_no_safety_vest_or_helmet,
         detect_near_machinery_or_vehicle, detect_in_restricted_area,
         detect_in_utility_pole_restricted_area, detect_machinery_close_to_pole, expire_date, updated_at)
         SELECT $1,$2,v.stream_name,v.video_url,v.model_key,v.recognition_enabled,v.work_start_hour,
          v.work_end_hour,v.detect_no_safety_vest_or_helmet,v.detect_near_machinery_or_vehicle,
          v.detect_in_restricted_area,v.detect_in_utility_pole_restricted_area,v.detect_machinery_close_to_pole,
          v.expire_date,now() FROM {DECODE} RETURNING id"
    );
    sqlx::query_scalar(&statement)
        .bind(site)
        .bind(group)
        .bind(sqlx::types::Json(settings))
        .fetch_one(db)
        .await
        .map_err(conflict)
}

async fn replace_stream(
    db: &mut PgConnection,
    cfg: i32,
    site: i32,
    settings: &Settings,
) -> Result<(), ApiError> {
    let statement = format!(
        "UPDATE stream_configs sc SET stream_name=v.stream_name,video_url=v.video_url,model_key=v.model_key,
         recognition_enabled=v.recognition_enabled,work_start_hour=v.work_start_hour,work_end_hour=v.work_end_hour,
         detect_no_safety_vest_or_helmet=v.detect_no_safety_vest_or_helmet,
         detect_near_machinery_or_vehicle=v.detect_near_machinery_or_vehicle,
         detect_in_restricted_area=v.detect_in_restricted_area,
         detect_in_utility_pole_restricted_area=v.detect_in_utility_pole_restricted_area,
         detect_machinery_close_to_pole=v.detect_machinery_close_to_pole,expire_date=v.expire_date,updated_at=now()
         FROM {DECODE} WHERE sc.id=$1 AND sc.site_id=$2"
    );
    let update = sqlx::query(&statement)
        .bind(cfg)
        .bind(site)
        .bind(sqlx::types::Json(settings))
        .execute(db)
        .await
        .map_err(conflict)?;
    if update.rows_affected() != 1 {
        return Err(STREAM_NOT_FOUND);
    }
    Ok(())
}

fn conflict(error: sqlx::Error) -> ApiError {
    if error
        .as_database_error()
        .is_some_and(|error| error.is_unique_violation() || error.is_foreign_key_violation())
    {
        ApiError::new(
            StatusCode::CONFLICT,
            "Stream configuration update conflicted.",
        )
    } else {
        database_unavailable(error)
    }
}

async fn create_stream(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Json(payload): Json<Create>,
) -> Result<Json<Value>, ResourceError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    payload.settings.validate()?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let groups = site_groups(&mut tx, &account, group, payload.site_id, true, true).await?;
    let owner = owner_group(&groups, group, payload.group_id)?;
    Catalog::load(&mut tx)
        .await?
        .validate_models(
            &scope.tenant_id.to_string(),
            &account,
            payload.site_id,
            [payload.settings.model_key.as_str()],
        )
        .await?;
    quota(&mut tx, owner, Some(1)).await?;
    if owner_of_name(&mut tx, payload.site_id, &payload.settings.stream_name)
        .await?
        .is_some()
    {
        return Err(NAME_EXISTS.into());
    }
    let id = insert_stream(&mut tx, payload.site_id, owner, &payload.settings).await?;
    tx.commit().await.map_err(conflict)?;
    Ok(Json(
        json!({"id": id, "message": "Stream configuration created successfully."}),
    ))
}

async fn upsert_streams(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Path(site_id): Path<i32>,
    Json(body): Json<Upsert>,
) -> Result<Json<Vec<Value>>, ResourceError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut names = HashSet::new();
    let mut ids = HashSet::new();
    let mut new_count = 0;
    for item in &body.streams {
        item.settings.validate()?;
        if !names.insert(item.settings.stream_name.as_str()) {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "Duplicate stream names are not allowed.",
            )
            .into());
        }
        if let Some(id) = item.id {
            if !ids.insert(id) {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "Duplicate stream IDs are not allowed.",
                )
                .into());
            }
        } else {
            new_count += 1;
        }
    }

    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let groups = site_groups(&mut tx, &account, group, site_id, true, true).await?;
    let owner = owner_group(&groups, group, None)?;
    if !body.streams.is_empty() {
        Catalog::load(&mut tx)
            .await?
            .validate_models(
                &scope.tenant_id.to_string(),
                &account,
                site_id,
                body.streams
                    .iter()
                    .map(|item| item.settings.model_key.as_str()),
            )
            .await?;
    }
    let current: Vec<(i32, i32, String)> = sqlx::query_as(
        "SELECT id,group_id,stream_name FROM stream_configs WHERE site_id=$1 FOR UPDATE",
    )
    .bind(site_id)
    .fetch_all(&mut *tx)
    .await
    .map_err(database_unavailable)?;
    let by_id: HashSet<_> = current
        .iter()
        .filter(|row| group.is_none_or(|id| id == row.1))
        .map(|row| row.0)
        .collect();
    let by_name: HashMap<_, _> = current.iter().map(|row| (row.2.as_str(), row.0)).collect();
    for item in &body.streams {
        if item.id.is_some_and(|id| !by_id.contains(&id)) {
            return Err(STREAM_NOT_FOUND.into());
        }
        if by_name
            .get(item.settings.stream_name.as_str())
            .is_some_and(|id| Some(*id) != item.id)
        {
            return Err(NAME_EXISTS.into());
        }
    }
    if new_count > 0 {
        quota(&mut tx, owner, Some(new_count)).await?;
    }
    for item in body.streams {
        if let Some(id) = item.id {
            replace_stream(&mut tx, id, site_id, &item.settings).await?;
        } else {
            insert_stream(&mut tx, site_id, owner, &item.settings).await?;
        }
    }
    let rows = stream_list(&mut tx, site_id, group).await?;
    tx.commit().await.map_err(conflict)?;
    Ok(Json(rows))
}

async fn stream_site(db: &mut PgConnection, cfg: i32) -> Result<i32, ApiError> {
    sqlx::query_scalar("SELECT site_id FROM stream_configs WHERE id=$1")
        .bind(cfg)
        .fetch_optional(db)
        .await
        .map_err(database_unavailable)?
        .ok_or(STREAM_NOT_FOUND)
}

async fn lock_stream(
    db: &mut PgConnection,
    cfg: i32,
    site: i32,
    group: Option<i32>,
) -> Result<Value, ApiError> {
    let query = format!(
        "SELECT group_id, {SETTINGS_JSON}
         FROM stream_configs sc WHERE id=$1 AND site_id=$2 FOR UPDATE",
    );
    let (owner, settings): (i32, Value) = sqlx::query_as(&query)
        .bind(cfg)
        .bind(site)
        .fetch_optional(db)
        .await
        .map_err(database_unavailable)?
        .ok_or(STREAM_NOT_FOUND)?;
    if group.is_some_and(|expected| expected != owner) {
        return Err(GROUP_DENIED);
    }
    Ok(settings)
}

async fn update_stream(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Path(cfg): Path<i32>,
    Json(patch): Json<serde_json::Map<String, Value>>,
) -> Result<Json<Value>, ResourceError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let site = stream_site(&mut tx, cfg).await?;
    site_groups(&mut tx, &account, group, site, true, true).await?;
    let mut current = lock_stream(&mut tx, cfg, site, group).await?;
    if let Some(settings) = current.as_object_mut() {
        settings.extend(patch);
    }
    let settings: Settings = serde_json::from_value(current).map_err(|_| {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid stream configuration",
        )
    })?;
    settings.validate()?;
    if owner_of_name(&mut tx, site, &settings.stream_name)
        .await?
        .is_some_and(|id| id != cfg)
    {
        return Err(NAME_EXISTS.into());
    }
    Catalog::load(&mut tx)
        .await?
        .validate_models(
            &scope.tenant_id.to_string(),
            &account,
            site,
            [settings.model_key.as_str()],
        )
        .await?;
    replace_stream(&mut tx, cfg, site, &settings).await?;
    tx.commit().await.map_err(conflict)?;
    Ok(Json(
        json!({"message": "Stream configuration updated successfully."}),
    ))
}

async fn delete_stream(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    Path(cfg): Path<i32>,
) -> Result<Json<Value>, ApiError> {
    let actor = state.auth.authorize(&headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account = scope.account_for_update(&mut tx, &actor.subject).await?;
    let group = admin_group(&account)?;
    let site = stream_site(&mut tx, cfg).await?;
    site_groups(&mut tx, &account, group, site, false, true).await?;
    lock_stream(&mut tx, cfg, site, group).await?;
    sqlx::query("DELETE FROM stream_configs WHERE id=$1")
        .bind(cfg)
        .execute(&mut *tx)
        .await
        .map_err(conflict)?;
    tx.commit().await.map_err(conflict)?;
    Ok(Json(
        json!({"message": "Stream configuration deleted successfully."}),
    ))
}

use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::{PgConnection, Row};

use super::{
    Scope, Settings, database_error, inaccessible, invalid, private, projection,
    query::{SITE_CONDITION, load_record},
    scope,
};
use crate::{ApiError, AppState, auth::Grant};

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum ReviewStatus {
    Pending,
    Resolved,
    Dismissed,
}

impl ReviewStatus {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Resolved => "resolved",
            Self::Dismissed => "dismissed",
        }
    }
}

fn actions(record: &Value) -> &'static [ReviewStatus] {
    use ReviewStatus::{Dismissed, Pending, Resolved};
    const COMPLETED: &[ReviewStatus] = &[Pending];
    const QUEUED: &[ReviewStatus] = &[Resolved, Dismissed];
    if record["is_flagged"] != true {
        return &[];
    }
    match record["review_status"].as_str() {
        Some("pending") => QUEUED,
        Some("resolved" | "dismissed") => COMPLETED,
        _ => &[],
    }
}

#[derive(Deserialize)]
pub(super) struct Queue {
    review_status: Option<ReviewStatus>,
    site_id: Option<i32>,
    current_id: Option<i32>,
}

pub(super) async fn next(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Query(query): Query<Queue>,
) -> Result<Response, ApiError> {
    let (mut db, scope) = scope(&state, &settings, &headers, Grant::ViolationReview).await?;
    let (ids, names) = scope.select_sites(query.site_id)?;
    let statement = format!(
        "SELECT to_jsonb(v) FROM violations AS v WHERE {SITE_CONDITION}
         AND v.is_flagged IS TRUE AND v.review_status = $3 AND ($4::int IS NULL OR v.id <> $4)
         ORDER BY v.flagged_at ASC NULLS LAST, v.detection_time ASC, v.id ASC LIMIT 1",
    );
    let row = sqlx::query_scalar(&statement)
        .bind(ids)
        .bind(names)
        .bind(
            query
                .review_status
                .unwrap_or(ReviewStatus::Pending)
                .as_str(),
        )
        .bind(query.current_id)
        .fetch_optional(&mut *db)
        .await
        .map_err(database_error)?;
    let detail = match row {
        Some(record) => projection::detail(&mut db, &settings, record).await?,
        None => Value::Null,
    };
    Ok(private(Json(detail).into_response()))
}

pub(super) async fn audit_log(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Path(id): Path<i32>,
) -> Result<Response, ApiError> {
    let (mut db, scope) = scope(&state, &settings, &headers, Grant::ViolationReview).await?;
    let record = load_record(&mut db, &scope, id, false).await?;
    if record["is_flagged"] != true {
        return Err(inaccessible());
    }
    Ok(private(Json(audit(&mut db, id).await?).into_response()))
}

pub(super) async fn audit(db: &mut PgConnection, id: i32) -> Result<Vec<Value>, ApiError> {
    sqlx::query_scalar(
        "SELECT jsonb_build_object(
             'id', id, 'violation_id', violation_id, 'actor_user_id', reviewed_by,
             'action', action, 'old_status', old_status, 'new_status', new_status,
             'note', review_note, 'flagged_reason', flagged_reason, 'created_at', reviewed_at
         ) FROM public.violation_review_audit_logs WHERE violation_id = $1
         ORDER BY reviewed_at DESC, id DESC",
    )
    .bind(id)
    .fetch_all(db)
    .await
    .map_err(database_error)
}

#[derive(Deserialize)]
pub(super) struct Locale {
    locale: Option<String>,
}

pub(super) async fn options(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Path(id): Path<i32>,
    Query(locale): Query<Locale>,
) -> Result<Response, ApiError> {
    let (mut db, scope) = scope(&state, &settings, &headers, Grant::ViolationReview).await?;
    let record = load_record(&mut db, &scope, id, false).await?;
    explicit_site(&mut db, &scope, &record, false).await?;
    let chinese = locale
        .locale
        .as_deref()
        .unwrap_or("zh-TW")
        .to_ascii_lowercase()
        .starts_with("zh");
    let options: Vec<_> = actions(&record)
        .iter()
        .map(|action| {
            let label = match (chinese, action) {
                (true, ReviewStatus::Pending) => "待審核",
                (true, ReviewStatus::Resolved) => "已處理",
                (true, ReviewStatus::Dismissed) => "已駁回",
                (false, action) => action.as_str(),
            };
            json!({"id": action.as_str(), "label": label})
        })
        .collect();
    Ok(option_response(
        &settings,
        &scope,
        id,
        "review",
        &record["review_status"],
        options,
    ))
}

pub(super) async fn feedback_options(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Path(id): Path<i32>,
) -> Result<Response, ApiError> {
    let (mut db, scope) = scope(&state, &settings, &headers, Grant::ViolationRead).await?;
    let record = load_record(&mut db, &scope, id, false).await?;
    explicit_site(&mut db, &scope, &record, false).await?;
    let options = projection::classes(&record["model_classes"])?
        .iter()
        .map(|class| json!({"id": class.code, "label": class.display_name, "color": class.color}))
        .collect();
    Ok(option_response(
        &settings,
        &scope,
        id,
        "feedback",
        &record["model_version"],
        options,
    ))
}

fn option_response(
    settings: &Settings,
    scope: &Scope,
    id: i32,
    kind: &str,
    source: &Value,
    options: Vec<Value>,
) -> Response {
    let key = json!({"user": scope.user, "tenant": settings.tenant, "role": scope.role,
        "violation": id, "kind": kind, "source": source, "options": options});
    let revision = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&key).expect("valid review options"))
    );
    private(Json(json!({
        "schema_version": 1, "revision": revision, "options": options, "default_id": null, "policy": {}
    })).into_response())
}

pub(super) async fn explicit_site(
    db: &mut PgConnection,
    scope: &Scope,
    record: &Value,
    lock: bool,
) -> Result<(), ApiError> {
    let site = scope
        .sites
        .iter()
        .find(|site| {
            if let Some(id) = record["site_id"].as_i64() {
                i64::from(site.id) == id
            } else {
                record["site"] == site.name
            }
        })
        .ok_or_else(inaccessible)?;
    let mut query = "SELECT s.id FROM user_sites AS m JOIN sites AS s ON m.site_id = s.id
          WHERE m.user_id = $1 AND s.id = $2 AND s.name = $3"
        .to_owned();
    if lock {
        query.push_str(" FOR SHARE OF s, m");
    }
    sqlx::query_scalar::<_, i32>(&query)
        .bind(scope.user)
        .bind(site.id)
        .bind(&site.name)
        .fetch_optional(&mut *db)
        .await
        .map_err(database_error)?
        .ok_or_else(inaccessible)?;
    if lock && scope.role != "super_admin" {
        sqlx::query_scalar::<_, i32>(
            "SELECT s.id FROM sites AS s JOIN site_groups AS g ON g.site_id = s.id
             WHERE s.id = $1 AND s.name = $2 AND g.group_id = $3 FOR SHARE OF s, g",
        )
        .bind(site.id)
        .bind(&site.name)
        .bind(scope.group)
        .fetch_optional(db)
        .await
        .map_err(database_error)?
        .ok_or_else(inaccessible)?;
    }
    Ok(())
}

#[derive(Deserialize)]
pub(super) struct Update {
    review_status: ReviewStatus,
    review_note: Option<String>,
}

pub(super) async fn transition(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Path(id): Path<i32>,
    Json(update): Json<Update>,
) -> Result<Response, ApiError> {
    let actor = state
        .auth
        .authorize(&headers, Grant::ViolationReview)
        .await?;
    if update
        .review_note
        .as_ref()
        .is_some_and(|note| note.chars().count() > 1000 || note.contains('\0'))
    {
        return Err(invalid("Invalid review_note"));
    }
    let mut tx = state.db.begin().await.map_err(database_error)?;
    let scope = Scope::load(&mut tx, &actor, settings.tenant, true, true).await?;
    let record = load_record(&mut tx, &scope, id, true).await?;
    if record["is_flagged"] != true {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "Flagged violation not found",
        ));
    }
    explicit_site(&mut tx, &scope, &record, true).await?;
    if !actions(&record).contains(&update.review_status) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Review transition is not currently allowed",
        ));
    }

    let now = Utc::now();
    let row = sqlx::query(
        "UPDATE violations AS v
         SET review_status = $2, review_note = $3, reviewed_by = $4, reviewed_at = $5
         WHERE v.id = $1 RETURNING to_jsonb(v) AS record",
    )
    .bind(id)
    .bind(update.review_status.as_str())
    .bind(&update.review_note)
    .bind(scope.user)
    .bind(now)
    .fetch_one(&mut *tx)
    .await
    .map_err(database_error)?;
    let updated = projection::base_detail(row.try_get("record").map_err(database_error)?)?;
    sqlx::query(
        "INSERT INTO public.violation_review_audit_logs
           (violation_id, action, old_status, new_status, review_note, flagged_reason, reviewed_by, reviewed_at)
         VALUES ($1, 'review_status_changed', $2, $3, $4, $5, $6, $7)",
    ).bind(id).bind(record["review_status"].as_str()).bind(update.review_status.as_str())
        .bind(update.review_note).bind(record["flag_reason"].as_str()).bind(scope.user).bind(now)
        .execute(&mut *tx).await.map_err(database_error)?;
    tx.commit().await.map_err(database_error)?;
    Ok(private(Json(updated).into_response()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_transitions_require_a_flag_and_current_state() {
        assert!(actions(&json!({"is_flagged": false, "review_status": "pending"})).is_empty());
        assert!(
            actions(&json!({"is_flagged": true, "review_status": "pending"}))
                .contains(&ReviewStatus::Resolved)
        );
        assert!(
            !actions(&json!({"is_flagged": true, "review_status": "resolved"}))
                .contains(&ReviewStatus::Dismissed)
        );
        assert!(
            actions(&json!({"is_flagged": true, "review_status": "resolved"}))
                .contains(&ReviewStatus::Pending)
        );
    }
}

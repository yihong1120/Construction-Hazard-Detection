use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{ApiError, AppState, tenancy::database_unavailable};

use super::{Settings, account};

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReadStatus {
    Read,
    Unread,
    All,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind {
    Signature,
    Violation,
    Document,
    SiteAlert,
    System,
}

impl Kind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Signature => "signature",
            Self::Violation => "violation",
            Self::Document => "document",
            Self::SiteAlert => "site_alert",
            Self::System => "system",
        }
    }
}

#[derive(Deserialize)]
pub(super) struct Page {
    cursor: Option<String>,
    #[serde(default = "page_size")]
    page_size: usize,
    status: Option<ReadStatus>,
    #[serde(rename = "type")]
    kind: Option<Kind>,
}

const fn page_size() -> usize {
    20
}

#[derive(Deserialize, Serialize)]
struct Cursor {
    created_at: DateTime<Utc>,
    id: i32,
}

fn decode_cursor(value: &str) -> Result<Cursor, ApiError> {
    let cursor = (value.len() <= 192)
        .then(|| URL_SAFE_NO_PAD.decode(value.trim_end_matches('=')).ok())
        .flatten()
        .and_then(|bytes| serde_json::from_slice::<Cursor>(&bytes).ok())
        .filter(|cursor| cursor.id > 0);
    cursor.ok_or_else(|| {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_notification_cursor",
        )
    })
}

const ROW: &str = "jsonb_build_object(
    'id', id, 'type', type, 'title', title, 'body', body, 'deep_link', deep_link,
    'is_read', is_read, 'created_at', created_at, 'metadata', metadata)";

pub(super) async fn list(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    Query(page): Query<Page>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    if !(1..=100).contains(&page.page_size) {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid notification page size",
        ));
    }
    let (_, owner) = account(&state, &settings, &headers).await?;
    let cursor = page.cursor.as_deref().map(decode_cursor).transpose()?;
    let read = match page.status {
        Some(ReadStatus::Read) => Some(true),
        Some(ReadStatus::Unread) => Some(false),
        _ => None,
    };
    let sql = format!(
        "SELECT created_at, id, {ROW} FROM notifications
         WHERE user_id = $1 AND ($2::boolean IS NULL OR is_read = $2)
           AND ($3::text IS NULL OR type = $3)
           AND ($4::timestamptz IS NULL OR (created_at, id) < ($4, $5))
         ORDER BY created_at DESC, id DESC LIMIT $6"
    );
    let mut rows: Vec<(DateTime<Utc>, i32, Value)> = sqlx::query_as(&sql)
        .bind(owner.id)
        .bind(read)
        .bind(page.kind.map(Kind::as_str))
        .bind(cursor.as_ref().map(|cursor| cursor.created_at))
        .bind(cursor.as_ref().map(|cursor| cursor.id))
        .bind(page.page_size as i64 + 1)
        .fetch_all(&state.db)
        .await
        .map_err(database_unavailable)?;
    let next = if rows.len() > page.page_size {
        rows.pop();
        rows.last().map(|(created_at, id, _)| {
            let cursor = Cursor {
                created_at: *created_at,
                id: *id,
            };
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&cursor).expect("cursor is serialisable"))
        })
    } else {
        None
    };
    let items: Vec<_> = rows.into_iter().map(|(_, _, item)| item).collect();
    Ok(Json(json!({"items": items, "next_cursor": next})))
}

pub(super) async fn unread_count(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let (_, owner) = account(&state, &settings, &headers).await?;
    let count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM notifications WHERE user_id = $1 AND NOT is_read")
            .bind(owner.id)
            .fetch_one(&state.db)
            .await
            .map_err(database_unavailable)?;
    Ok(Json(json!({"unread_count": count})))
}

pub(super) async fn mark_read(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    Path(id): Path<i32>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let (_, owner) = account(&state, &settings, &headers).await?;
    let sql = format!(
        "UPDATE notifications SET is_read = true WHERE id = $1 AND user_id = $2 RETURNING {ROW}"
    );
    let item = sqlx::query_scalar(&sql)
        .bind(id)
        .bind(owner.id)
        .fetch_optional(&state.db)
        .await
        .map_err(database_unavailable)?
        .ok_or_else(not_found)?;
    Ok(Json(item))
}

pub(super) async fn read_all(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let (_, owner) = account(&state, &settings, &headers).await?;
    let result =
        sqlx::query("UPDATE notifications SET is_read = true WHERE user_id = $1 AND NOT is_read")
            .bind(owner.id)
            .execute(&state.db)
            .await
            .map_err(database_unavailable)?;
    Ok(Json(json!({"updated_count": result.rows_affected()})))
}

pub(super) async fn delete(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    Path(id): Path<i32>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let (_, owner) = account(&state, &settings, &headers).await?;
    let result = sqlx::query("DELETE FROM notifications WHERE id = $1 AND user_id = $2")
        .bind(id)
        .bind(owner.id)
        .execute(&state.db)
        .await
        .map_err(database_unavailable)?;
    if result.rows_affected() == 0 {
        return Err(not_found());
    }
    Ok(Json(json!({"message": "Notification deleted."})))
}

fn not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "Notification not found.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_legacy_isoformat_cursor_and_rejects_bad_ids() {
        let payload = br#"{"created_at":"2026-11-19T00:00:00+00:00","id":987}"#;
        assert_eq!(
            decode_cursor(&URL_SAFE_NO_PAD.encode(payload)).unwrap().id,
            987
        );
        assert!(decode_cursor("invalid").is_err());
        let payload = br#"{"created_at":"2026-11-19T00:00:00+00:00","id":0}"#;
        assert!(decode_cursor(&URL_SAFE_NO_PAD.encode(payload)).is_err());
    }
}

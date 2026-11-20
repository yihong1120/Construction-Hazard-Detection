use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{PgConnection, Postgres, QueryBuilder};

use super::{
    Scope, Settings, database_error, inaccessible, invalid, private, projection,
    review::ReviewStatus, scope, search,
};
use crate::{ApiError, AppState, auth::Grant};

pub(super) const SITE_CONDITION: &str =
    "(v.site_id = ANY($1) OR (v.site_id IS NULL AND v.site = ANY($2)))";

pub(super) const CLASS_TYPES: [(&str, &str); 7] = [
    ("no_safety_helmet", "未戴安全帽"),
    ("no_safety_vest", "未穿安全背心"),
    ("near_vehicle", "人員靠近車輛"),
    ("near_machinery", "人員靠近機具"),
    ("restricted_area", "進入管制區"),
    ("utility_pole_restricted_area", "進入電桿管制區"),
    ("machinery_close_to_pole", "機具靠近電桿"),
];

#[derive(Deserialize)]
pub(super) struct SiteFilter {
    site_id: Option<i32>,
    group_id: Option<i32>,
}

pub(super) async fn sites(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let (_, scope) = scope(&state, &settings, &headers, Grant::ViolationRead).await?;
    Ok(private(Json(scope.sites).into_response()))
}

pub(super) async fn filters(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Query(filter): Query<SiteFilter>,
) -> Result<Response, ApiError> {
    let (mut db, scope) = scope(&state, &settings, &headers, Grant::ViolationRead).await?;
    let (ids, _) = scope.select_sites(filter.site_id)?;
    if filter.group_id.is_some() && scope.role != "super_admin" && filter.group_id != scope.group {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "No access to group_id",
        ));
    }
    let group = if scope.role == "super_admin" {
        filter.group_id
    } else {
        scope.group
    };
    let cameras = sqlx::query_as::<_, (String, String)>(
        "SELECT id::text, stream_name FROM stream_configs
         WHERE site_id = ANY($1) AND ($2::int IS NULL OR group_id = $2)
         ORDER BY stream_name, id",
    )
    .bind(ids)
    .bind(group)
    .fetch_all(&mut *db)
    .await
    .map_err(database_error)?
    .into_iter()
    .map(|(id, name)| json!({"stream_id": id, "name": name}))
    .collect::<Vec<_>>();
    let types: Vec<_> = CLASS_TYPES
        .into_iter()
        .map(|(code, label)| json!({"code": code, "label": label}))
        .collect();
    Ok(private(
        Json(json!({"cameras": cameras, "violation_types": types})).into_response(),
    ))
}

#[derive(Deserialize)]
pub(super) struct Filters {
    site_id: Option<i32>,
    stream_id: Option<String>,
    violation_type: Option<String>,
    keyword: Option<String>,
    start_time: Option<String>,
    end_time: Option<String>,
    #[serde(default = "default_limit")]
    limit: u16,
    flagged: Option<bool>,
    review_status: Option<ReviewStatus>,
    cursor: Option<String>,
}

const fn default_limit() -> u16 {
    20
}

#[derive(Deserialize, Serialize)]
struct Cursor {
    detection_time: DateTime<Utc>,
    id: i32,
}

impl Cursor {
    fn decode(encoded: &str) -> Result<Self, ApiError> {
        if encoded.len() > 240 {
            return Err(invalid("Invalid cursor"));
        }
        let bytes = URL_SAFE_NO_PAD
            .decode(encoded.trim_end_matches('='))
            .map_err(|_| invalid("Invalid cursor"))?;
        let cursor: Self = serde_json::from_slice(&bytes).map_err(|_| invalid("Invalid cursor"))?;
        if cursor.id <= 0 {
            return Err(invalid("Invalid cursor"));
        }
        Ok(cursor)
    }

    fn encode(&self) -> String {
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(self).expect("valid pagination cursor"))
    }
}

pub(super) fn time(value: &str) -> Result<DateTime<Utc>, ApiError> {
    if let Ok(stamp) = DateTime::parse_from_rfc3339(value) {
        return Ok(stamp.to_utc());
    }
    value
        .parse::<NaiveDateTime>()
        .map(|stamp| stamp.and_utc())
        .map_err(|_| invalid("Invalid time filter"))
}

pub(super) async fn stream_filter(
    db: &mut PgConnection,
    scope: &Scope,
    id: &str,
    ids: &[i32],
) -> Result<(i32, i32), ApiError> {
    let id: i32 = id
        .trim()
        .parse()
        .ok()
        .filter(|id| *id > 0)
        .ok_or_else(|| invalid("stream_id must be a positive stream configuration ID"))?;
    let row = sqlx::query_as::<_, (i32, i32)>(
        "SELECT site_id, group_id FROM stream_configs WHERE id = $1",
    )
    .bind(id)
    .fetch_optional(db)
    .await
    .map_err(database_error)?;
    if let Some((site_id, group_id)) = row
        && ids.contains(&site_id)
        && (scope.role == "super_admin" || scope.group == Some(group_id))
    {
        return Ok((id, site_id));
    }
    Err(ApiError::new(
        StatusCode::FORBIDDEN,
        "No access to stream_id",
    ))
}

pub(super) async fn history(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Query(f): Query<Filters>,
) -> Result<Response, ApiError> {
    let flagged = f.flagged == Some(true) || f.review_status.is_some();
    let grant = if flagged {
        Grant::ViolationReview
    } else {
        Grant::ViolationRead
    };
    let (mut db, scope) = scope(&state, &settings, &headers, grant).await?;
    if f.limit == 0 || f.limit > 100 {
        return Err(invalid("limit must be between 1 and 100"));
    }
    let mut site = f.site_id;
    let (ids, _) = scope.select_sites(site)?;
    let stream = if let Some(id) = f.stream_id.as_deref().filter(|id| !id.is_empty()) {
        let (stream_id, site_id) = stream_filter(&mut db, &scope, id, &ids).await?;
        site = Some(site_id);
        Some(stream_id)
    } else {
        None
    };

    let (ids, names) = scope.select_sites(site)?;
    let mut query = QueryBuilder::<Postgres>::new(
        "SELECT v.id, v.site, v.stream_name, v.detection_time, v.image_path, v.warnings_json,
                v.is_flagged, v.review_status, v.model_version, v.model_classes,
                (SELECT f.note FROM public.violation_feedback AS f
                 WHERE f.violation_id = v.id AND f.note IS NOT NULL
                 ORDER BY f.created_at DESC, f.id DESC LIMIT 1)
         FROM violations AS v WHERE (v.site_id = ANY(",
    );
    query
        .push_bind(ids)
        .push(") OR (v.site_id IS NULL AND v.site = ANY(")
        .push_bind(names)
        .push(")))");
    if flagged {
        query.push(" AND v.is_flagged IS TRUE");
    }
    if let Some(status) = f.review_status {
        query
            .push(" AND v.review_status = ")
            .push_bind(status.as_str());
    }
    if let Some(stream_id) = stream {
        query
            .push(" AND v.stream_config_id = ")
            .push_bind(stream_id);
    }
    if let Some(kind) = f.violation_type.as_deref().filter(|kind| !kind.is_empty()) {
        if !CLASS_TYPES.iter().any(|(code, _)| *code == kind.trim()) {
            return Err(invalid("Unsupported violation_type"));
        }
        query
            .push(" AND v.violation_type_codes::jsonb @> ")
            .push_bind(json!([kind.trim()]));
    }
    if let Some(keyword) = f
        .keyword
        .as_deref()
        .map(str::trim)
        .filter(|word| !word.is_empty())
    {
        if keyword.chars().count() > 128 {
            return Err(invalid("Keyword is too long"));
        }
        let terms = search::patterns(keyword);
        if !terms.is_empty() {
            query
                .push(" AND (v.stream_name ILIKE ANY(")
                .push_bind(terms.clone())
                .push(") OR v.warnings_json ILIKE ANY(")
                .push_bind(terms)
                .push("))");
        }
    }
    if let Some(stamp) = f.start_time.as_deref() {
        query
            .push(" AND v.detection_time >= ")
            .push_bind(time(stamp)?);
    }
    if let Some(stamp) = f.end_time.as_deref() {
        query
            .push(" AND v.detection_time <= ")
            .push_bind(time(stamp)?);
    }
    if let Some(encoded) = f.cursor.as_deref().filter(|item| !item.is_empty()) {
        let cursor = Cursor::decode(encoded)?;
        query
            .push(" AND (v.detection_time, v.id) < (")
            .push_bind(cursor.detection_time)
            .push(", ")
            .push_bind(cursor.id)
            .push(")");
    }
    query
        .push(" ORDER BY v.detection_time DESC, v.id DESC LIMIT ")
        .push_bind(i64::from(f.limit) + 1);

    let mut rows = query
        .build_query_as::<projection::ListRow>()
        .fetch_all(&mut *db)
        .await
        .map_err(database_error)?;
    let has_more = rows.len() > usize::from(f.limit);
    if has_more {
        rows.pop();
    }
    let next_cursor = rows.last().filter(|_| has_more).map(|row| {
        Cursor {
            detection_time: row.3,
            id: row.0,
        }
        .encode()
    });
    let items: Vec<_> = rows
        .into_iter()
        .map(projection::list_item)
        .collect::<Result<_, _>>()?;
    Ok(private(
        Json(json!({"items": items, "has_more": has_more, "next_cursor": next_cursor}))
            .into_response(),
    ))
}

pub(super) async fn load_record(
    db: &mut PgConnection,
    scope: &Scope,
    id: i32,
    lock: bool,
) -> Result<Value, ApiError> {
    let (ids, names) = scope.select_sites(None)?;
    let lock = if lock { " FOR UPDATE OF v" } else { "" };
    let statement = format!(
        "SELECT to_jsonb(v) FROM violations AS v WHERE {SITE_CONDITION} AND v.id = $3{lock}"
    );
    sqlx::query_scalar(&statement)
        .bind(ids)
        .bind(names)
        .bind(id)
        .fetch_optional(db)
        .await
        .map_err(database_error)?
        .ok_or_else(inaccessible)
}

pub(super) async fn detail(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Path(id): Path<i32>,
) -> Result<Response, ApiError> {
    let (mut db, scope) = scope(&state, &settings, &headers, Grant::ViolationRead).await?;
    let record = load_record(&mut db, &scope, id, false).await?;
    let detail = projection::detail(&mut db, &settings, record).await?;
    Ok(private(Json(detail).into_response()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_legacy_cursor_and_preserves_time_ties() {
        let token =
            URL_SAFE_NO_PAD.encode(br#"{"detection_time":"2026-11-19T23:10:05+08:00","id":47}"#);
        let cursor = Cursor::decode(&token).unwrap();
        assert_eq!(
            cursor.detection_time.to_rfc3339(),
            "2026-11-19T15:10:05+00:00"
        );
        assert_eq!(Cursor::decode(&cursor.encode()).unwrap().id, 47);
        assert!(Cursor::decode("%%%").is_err());
    }
}

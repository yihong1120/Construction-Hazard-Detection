use axum::{
    Extension, Json,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::Datelike;
use serde::Deserialize;
use serde_json::Value;

use super::{
    Settings, database_error, invalid, private,
    query::{CLASS_TYPES, SITE_CONDITION, stream_filter, time},
    scope,
};
use crate::{ApiError, AppState, auth::Grant};

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Bucket {
    #[default]
    Day,
    Hour,
    Week,
}

impl Bucket {
    const fn template(&self) -> &'static str {
        match self {
            Self::Day => "YYYY-MM-DD",
            Self::Hour => "YYYY-MM-DD\"T\"HH24:00:00\"Z\"",
            Self::Week => "IYYY-\"W\"IW",
        }
    }
}

#[derive(Deserialize)]
pub(super) struct Filters {
    start: String,
    end: String,
    site_id: Option<i32>,
    stream_id: Option<String>,
    violation_type: Option<String>,
    #[serde(default)]
    bucket: Bucket,
}

pub(super) async fn get(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Query(filter): Query<Filters>,
) -> Result<Response, ApiError> {
    let (mut db, scope) = scope(&state, &settings, &headers, Grant::ViolationRead).await?;
    if !matches!(scope.role.as_str(), "admin" | "super_admin") {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "violation_analytics_forbidden",
        ));
    }
    let start = time(&filter.start)?;
    let end = time(&filter.end)?;
    if end <= start {
        return Err(invalid("start must be before end"));
    }
    let limit = start
        .with_year(start.year() + 5)
        .or_else(|| {
            start
                .with_day(28)
                .and_then(|date| date.with_year(start.year() + 5))
        })
        .ok_or_else(|| invalid("Invalid analytics time range"))?;
    if end > limit {
        return Err(invalid("Query range must not exceed 5 years"));
    }

    let mut site = filter.site_id;
    let (ids, _) = scope.select_sites(site)?;
    let stream = if let Some(id) = filter.stream_id.as_deref().filter(|id| !id.is_empty()) {
        let (stream, site_id) = stream_filter(&mut db, &scope, id, &ids).await?;
        site = Some(site_id);
        Some(stream)
    } else {
        None
    };
    let kind = filter
        .violation_type
        .as_deref()
        .map(str::trim)
        .filter(|kind| !kind.is_empty());
    if kind.is_some_and(|kind| !CLASS_TYPES.iter().any(|(code, _)| *code == kind)) {
        return Err(invalid("Unsupported violation_type"));
    }
    let (ids, names) = scope.select_sites(site)?;
    let codes: Vec<_> = CLASS_TYPES.iter().map(|(code, _)| *code).collect();
    let labels: Vec<_> = CLASS_TYPES.iter().map(|(_, label)| *label).collect();

    // Materialise the scoped record IDs and dimensions once. Aggregate the
    // same filtered set for every chart, including multi-label events.
    let statement = format!(
        "WITH filtered AS MATERIALIZED (
            SELECT v.site_id, v.site, v.detection_time AT TIME ZONE 'UTC' AS stamp,
                   v.violation_type_codes::jsonb AS types
            FROM violations AS v WHERE {SITE_CONDITION}
            AND v.detection_time BETWEEN $3 AND $4
            AND ($5::int IS NULL OR v.stream_config_id = $5)
            AND ($6::text IS NULL OR v.violation_type_codes::jsonb @> jsonb_build_array($6))
        ), totals AS (
            SELECT count(*) AS total, count(*) FILTER (
                WHERE stamp::date = (now() AT TIME ZONE 'UTC')::date
            ) AS today FROM filtered
        ), trend AS (
            SELECT to_char(stamp, $7) AS bucket, count(*) AS count FROM filtered GROUP BY 1
        ), sites AS (
            SELECT s.id AS site_id, s.name AS site_name, count(*) AS count
            FROM filtered f JOIN sites s ON
              (f.site_id = s.id OR (f.site_id IS NULL AND f.site = s.name))
            GROUP BY s.id, s.name
        ), hours AS (
            SELECT extract(hour FROM stamp)::int AS hour, count(*) AS count FROM filtered GROUP BY 1
        ), types AS (
            SELECT c.type, c.label, count(*) AS count
            FROM filtered f JOIN unnest($8::text[], $9::text[]) c(type, label)
              ON f.types @> jsonb_build_array(c.type)
            WHERE ($6 IS NULL OR c.type = $6) GROUP BY c.type, c.label
        )
        SELECT jsonb_build_object(
            'summary', jsonb_build_object(
                'total', t.total, 'today', t.today,
                'top_site', (SELECT to_jsonb(s) FROM sites s ORDER BY count DESC, site_id LIMIT 1),
                'top_type', (SELECT to_jsonb(ty) FROM types ty ORDER BY count DESC, type LIMIT 1)
            ), 'trend', coalesce((SELECT jsonb_agg(tr ORDER BY bucket) FROM trend tr), '[]'::jsonb),
            'by_site', coalesce((SELECT jsonb_agg(s ORDER BY count DESC, site_id) FROM sites s), '[]'::jsonb),
            'by_type', coalesce((SELECT jsonb_agg(ty ORDER BY count DESC, type) FROM types ty), '[]'::jsonb),
            'by_hour', coalesce((SELECT jsonb_agg(h ORDER BY hour) FROM hours h), '[]'::jsonb)
        ) FROM totals t",
    );
    let result: Value = sqlx::query_scalar(&statement)
        .bind(ids)
        .bind(names)
        .bind(start)
        .bind(end)
        .bind(stream)
        .bind(kind)
        .bind(filter.bucket.template())
        .bind(codes)
        .bind(labels)
        .fetch_one(&mut *db)
        .await
        .map_err(database_error)?;
    Ok(private(Json(result).into_response()))
}

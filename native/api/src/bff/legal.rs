use axum::{
    Json,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{ApiError, AppState, tenancy::database_unavailable};

const DEFAULT_LOCALE: &str = "zh-TW";

#[derive(Deserialize)]
pub(super) struct Locale {
    locale: Option<String>,
}

pub(super) async fn documents(
    State(state): State<AppState>,
    Query(query): Query<Locale>,
) -> Result<Response, ApiError> {
    let requested = query.locale.as_deref().filter(|locale| !locale.is_empty());
    let documents: Value = sqlx::query_scalar(
        "SELECT COALESCE(jsonb_object_agg(type, jsonb_build_object(
                    'version', version, 'title', title, 'content', content)), '{}')
         FROM (
           SELECT DISTINCT ON (type) type, version, title, content
           FROM public.legal_documents
           WHERE locale IN ($1,$2) AND is_active AND effective_at <= CURRENT_TIMESTAMP
           ORDER BY type, (locale = $1) DESC, effective_at DESC, created_at DESC, id DESC
         ) AS documents",
    )
    .bind(requested.unwrap_or(DEFAULT_LOCALE))
    .bind(DEFAULT_LOCALE)
    .fetch_one(&state.db)
    .await
    .map_err(database_unavailable)?;
    let missing: Vec<_> = ["terms", "privacy", "ai_terms"]
        .into_iter()
        .filter(|kind| documents.get(kind).is_none())
        .collect();
    if !missing.is_empty() {
        return Ok(super::no_store((
            StatusCode::NOT_FOUND,
            Json(json!({"detail": {"code": "legal_documents_not_found", "missing": missing}})),
        )));
    }
    Ok(super::no_store(Json(documents).into_response()))
}

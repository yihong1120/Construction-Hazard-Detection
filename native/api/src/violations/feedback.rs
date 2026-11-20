use axum::{
    Extension, Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    Scope, Settings, database_error, invalid, private, projection, query::load_record,
    review::explicit_site,
};
use crate::{ApiError, AppState, auth::Grant};

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeedbackType {
    FalsePositive,
    FalseNegative,
    WrongClass,
    BadBbox,
}

impl FeedbackType {
    const fn as_str(self) -> &'static str {
        match self {
            Self::FalsePositive => "false_positive",
            Self::FalseNegative => "false_negative",
            Self::WrongClass => "wrong_class",
            Self::BadBbox => "bad_bbox",
        }
    }
}

#[derive(Deserialize)]
pub(super) struct Feedback {
    #[serde(rename = "type")]
    kind: FeedbackType,
    anonymous_id: Option<String>,
    target_detection_id: Option<String>,
    original_label: Option<String>,
    corrected_label: Option<String>,
    original_bbox: Option<[f64; 4]>,
    corrected_bbox: Option<[f64; 4]>,
    confidence: Option<f64>,
    note: Option<String>,
}

impl Feedback {
    fn validate(&self) -> Result<(), ApiError> {
        for (value, max) in [
            (&self.anonymous_id, 80),
            (&self.target_detection_id, 120),
            (&self.original_label, 120),
            (&self.corrected_label, 120),
            (&self.note, 1000),
        ] {
            if value
                .as_ref()
                .is_some_and(|item| item.chars().count() > max || item.contains('\0'))
            {
                return Err(invalid("Invalid feedback field"));
            }
        }
        for bbox in [self.original_bbox, self.corrected_bbox]
            .into_iter()
            .flatten()
        {
            if bbox
                .into_iter()
                .any(|value| !value.is_finite() || value < 0.0)
                || bbox[0] > bbox[2]
                || bbox[1] > bbox[3]
            {
                return Err(invalid("Invalid feedback bbox"));
            }
        }
        if self
            .confidence
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        {
            return Err(invalid("Invalid feedback confidence"));
        }
        let has_label = self
            .corrected_label
            .as_ref()
            .is_some_and(|label| !label.is_empty());
        let targeted = self.original_bbox.is_some()
            || self
                .target_detection_id
                .as_ref()
                .is_some_and(|id| !id.is_empty());
        let valid = match self.kind {
            FeedbackType::FalsePositive => true,
            FeedbackType::FalseNegative => has_label && self.corrected_bbox.is_some(),
            FeedbackType::WrongClass => targeted && has_label,
            FeedbackType::BadBbox => targeted && self.corrected_bbox.is_some(),
        };
        if !valid {
            return Err(invalid("Missing required feedback fields"));
        }
        Ok(())
    }

    fn validate_record(&self, record: &Value) -> Result<(), ApiError> {
        let categories = projection::classes(&record["model_classes"])?;
        if let Some(label) = self.corrected_label.as_deref()
            && !categories.iter().any(|class| class.code == label)
        {
            return Err(invalid("Unknown feedback category"));
        }
        if let Some(target) = self
            .target_detection_id
            .as_deref()
            .filter(|id| !id.is_empty())
            && !projection::detection_is_present(record["detections_json"].as_str(), target)
        {
            return Err(invalid(
                "target_detection_id does not belong to this violation",
            ));
        }
        Ok(())
    }
}

pub(super) async fn submit(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Path(id): Path<i32>,
    Json(feedback): Json<Feedback>,
) -> Result<Response, ApiError> {
    let actor = state
        .auth
        .authorize(&headers, Grant::ViolationFeedback)
        .await?;
    feedback.validate()?;
    let mut tx = state.db.begin().await.map_err(database_error)?;
    let scope = Scope::load(&mut tx, &actor, settings.tenant, false, true).await?;
    let record = load_record(&mut tx, &scope, id, true).await?;
    explicit_site(&mut tx, &scope, &record, true).await?;
    feedback.validate_record(&record)?;
    let now = Utc::now();
    let kind = feedback.kind.as_str();
    let created: i32 = sqlx::query_scalar(
        "INSERT INTO public.violation_feedback (violation_id, user_id, anonymous_id, target_detection_id,
            feedback_type, original_label, corrected_label, original_bbox, corrected_bbox, confidence,
            note, status, created_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'pending',$12) RETURNING id",
    )
    .bind(id).bind(scope.user).bind(&feedback.anonymous_id).bind(&feedback.target_detection_id)
    .bind(kind).bind(&feedback.original_label).bind(&feedback.corrected_label)
    .bind(feedback.original_bbox.map(|bbox| json!(bbox)))
    .bind(feedback.corrected_bbox.map(|bbox| json!(bbox)))
    .bind(feedback.confidence).bind(&feedback.note).bind(now)
    .fetch_one(&mut *tx).await.map_err(database_error)?;
    sqlx::query(
        "UPDATE violations SET is_flagged = TRUE, flag_reason = $2, flagged_by = $3,
            flagged_at = $4, review_status = 'pending' WHERE id = $1",
    )
    .bind(id)
    .bind(kind)
    .bind(scope.user)
    .bind(now)
    .execute(&mut *tx)
    .await
    .map_err(database_error)?;
    tx.commit().await.map_err(database_error)?;
    Ok(private((StatusCode::CREATED, Json(json!({
        "id": created, "violation_id": id, "type": kind, "status": "pending", "created_at": now,
        "target_detection_id": feedback.target_detection_id,
        "original_label": feedback.original_label, "corrected_label": feedback.corrected_label,
        "original_bbox": feedback.original_bbox, "corrected_bbox": feedback.corrected_bbox,
        "confidence": feedback.confidence, "note": feedback.note
    }))).into_response()))
}

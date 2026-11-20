mod fingerprint;
mod journal;

use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result, ensure};
use axum::{
    Extension, Json,
    extract::{Multipart, State, multipart::MultipartRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use sqlx::{PgConnection, PgPool, types::Json as PgJson};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    Scope, Settings, database_error, inaccessible, invalid, media::Store, private, projection,
    review::explicit_site,
};
use crate::{
    ApiError, AppState,
    evidence::{AcceptedUpload, Evidence, Receipt, UploadError, UploadMode, validate_upload},
};

pub(super) async fn start(db: &PgPool, media: Arc<Store>, stop: CancellationToken) -> Result<()> {
    let ready = sqlx::query_scalar::<_, bool>(
        "SELECT to_regclass('detection_model_catalog') IS NOT NULL
          AND to_regclass('public.detection_model_versions') IS NOT NULL
          AND to_regclass('public.evidence_write_operations') IS NOT NULL
          AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_attribute
            WHERE attrelid = 'violations'::regclass
              AND attname = 'event_fingerprint' AND NOT attisdropped
          )
          AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_index AS i
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
            WHERE i.indrelid = 'violations'::regclass
              AND a.attname = 'event_id' AND i.indisunique AND i.indisvalid
              AND i.indnkeyatts = 1 AND i.indpred IS NULL AND i.indimmediate
        )",
    )
    .fetch_one(db)
    .await
    .context("Check violation upload schema")?;
    ensure!(
        ready,
        "Violation uploads require model and evidence-operation tables with unique event IDs"
    );
    journal::start(db.clone(), media, stop);
    Ok(())
}

pub(super) async fn upload(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, UploadError> {
    let upload = validate_upload(&state, &headers, multipart, UploadMode::Production).await?;
    // Complete the journal transaction even if the caller disconnects after
    // sending its image. The upload permit remains owned until it is safe.
    let writer = tokio::spawn(store(state.db, settings, upload));
    let id = tokio::time::timeout(Duration::from_secs(12), writer)
        .await
        .map_err(|_| storage_error())?
        .map_err(|_| storage_error())??;
    Ok(private(
        Json(Receipt {
            message: "Violation uploaded successfully.",
            violation_id: i64::from(id),
        })
        .into_response(),
    ))
}

type WarningMap = Map<String, Value>;

fn normalise_warnings(raw: impl IntoIterator<Item = (String, Value)>) -> WarningMap {
    raw.into_iter()
        .filter_map(|(key, entry)| {
            let count = entry.get("count")?;
            if count.as_u64().is_none() && count.as_i64().is_none() {
                return None;
            }
            Some((key, json!({"count": count})))
        })
        .collect()
}

fn active_types(warnings: Option<&WarningMap>) -> PgJson<Vec<&'static str>> {
    let types = [
        ("warning_no_hardhat", "no_safety_helmet"),
        ("warning_no_safety_vest", "no_safety_vest"),
        ("warning_close_to_vehicle", "near_vehicle"),
        ("warning_close_to_machinery", "near_machinery"),
        ("warning_people_in_controlled_area", "restricted_area"),
        (
            "warning_people_in_utility_pole_controlled_area",
            "utility_pole_restricted_area",
        ),
        ("detect_machinery_close_to_pole", "machinery_close_to_pole"),
    ];
    PgJson(
        types
            .into_iter()
            .filter(|(key, _)| {
                warnings
                    .and_then(|warnings| warnings.get(*key))
                    .and_then(|warning| warning["count"].as_u64())
                    .is_some_and(|count| count > 0)
            })
            .map(|(_, kind)| kind)
            .collect(),
    )
}

async fn store(db: PgPool, settings: Settings, input: AcceptedUpload) -> Result<i32, ApiError> {
    let evidence = &input.evidence;
    let site = i32::try_from(evidence.site_id).map_err(|_| inaccessible())?;
    let mut tx = db.begin().await.map_err(database_error)?;
    sqlx::query("SET LOCAL statement_timeout = '4s'")
        .execute(&mut *tx)
        .await
        .map_err(database_error)?;
    sqlx::query("SET LOCAL lock_timeout = '2s'")
        .execute(&mut *tx)
        .await
        .map_err(database_error)?;

    let scope = Scope::load(&mut tx, &input.actor, settings.tenant, false, true).await?;
    // A platform administrator's wider read scope does not grant writes.
    explicit_site(&mut tx, &scope, &json!({"site_id": site}), true).await?;
    if !scope
        .sites
        .iter()
        .any(|entry| entry.id == site && entry.name == evidence.site)
    {
        return Err(inaccessible());
    }
    let model = model_snapshot(&mut tx, evidence, &scope, settings.tenant, site).await?;
    let event = evidence
        .event_id
        .map(|event| event.hyphenated().to_string());
    let warnings = evidence.warnings.clone().map(normalise_warnings);
    let image_sha: [u8; 32] = Sha256::digest(&input.image).into();
    let digest = fingerprint::calculate(
        evidence,
        evidence.detection_time,
        warnings.as_ref(),
        image_sha,
    )?;
    let previous_digest = || {
        evidence
            .prior_local_detection_time
            .map(|time| fingerprint::calculate(evidence, Some(time), warnings.as_ref(), image_sha))
            .transpose()
    };
    if let Some(event) = &event {
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 4104))")
            .bind(event)
            .execute(&mut *tx)
            .await
            .map_err(database_error)?;
        if let Some(record) = existing_event(&mut tx, event).await? {
            let prior_digest = previous_digest()?;
            let verified = verify_repeat(
                &settings.media,
                record,
                evidence,
                &digest,
                prior_digest.as_deref(),
            )
            .await?;
            if let Some(backfill) = verified.backfill {
                sqlx::query("UPDATE violations SET event_fingerprint = $1 WHERE id = $2 AND event_fingerprint IS NULL")
                    .bind(backfill).bind(verified.id).execute(&mut *tx).await.map_err(database_error)?;
            }
            tx.commit().await.map_err(database_error)?;
            return Ok(verified.id);
        }
    }

    let stream: Option<i32> = sqlx::query_scalar(
        "SELECT id FROM stream_configs
          WHERE site_id = $1 AND stream_name = $2
          ORDER BY id LIMIT 1 FOR SHARE",
    )
    .bind(site)
    .bind(&evidence.stream_name)
    .fetch_optional(&mut *tx)
    .await
    .map_err(database_error)?;
    let detection_time = evidence.detection_time.unwrap_or_else(Utc::now);
    let operation = journal::Write::new(detection_time, input.format);
    operation.begin(&mut tx, &db).await?;
    operation
        .persist(settings.media.clone(), input.image.clone())
        .await?;

    let id: Option<i32> = sqlx::query_scalar(
        "INSERT INTO violations (stream_name, detection_time, image_path, model_id,
            model_version, model_classes, detections_json, cone_polygon_json, pole_polygon_json,
            warnings_json, stream_config_id, violation_type_codes, is_flagged, created_at, event_id, site_id, site,
            event_fingerprint)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,FALSE,now(),$13,$14,$15,$16)
         ON CONFLICT (event_id) DO NOTHING RETURNING id",
    ).bind(&evidence.stream_name).bind(detection_time).bind(&operation.path)
        .bind(&evidence.model_id).bind(&evidence.model_version).bind(PgJson(model))
        .bind(render(evidence.detections.as_ref())?).bind(render(evidence.cone_polygon.as_ref())?)
        .bind(render(evidence.pole_polygon.as_ref())?).bind(render(warnings.as_ref())?)
        .bind(stream).bind(active_types(warnings.as_ref())).bind(&event).bind(site).bind(&evidence.site)
        .bind(&digest)
        .fetch_optional(&mut *tx).await.map_err(database_error)?;

    if let Some(id) = id {
        // Removing the journal and inserting the violation in one transaction
        // gives recovery an unambiguous owner after a lost reply or crash.
        operation.complete(&mut tx).await?;
        tx.commit().await.map_err(database_error)?;
        return Ok(id);
    }
    // An older writer may insert without the event advisory lock. Its unique
    // row still wins. Leave our unreferenced file with its recovery journal.
    let record = existing_event(&mut tx, event.as_deref().ok_or_else(storage_error)?)
        .await?
        .ok_or_else(storage_error)?;
    let prior_digest = previous_digest()?;
    Ok(verify_repeat(
        &settings.media,
        record,
        evidence,
        &digest,
        prior_digest.as_deref(),
    )
    .await?
    .id)
}

fn render(value: Option<&impl serde::Serialize>) -> Result<Option<String>, ApiError> {
    value
        .map(serde_json::to_string)
        .transpose()
        .map_err(|_| invalid("Invalid evidence metadata"))
}

async fn existing_event(db: &mut PgConnection, event: &str) -> Result<Option<Value>, ApiError> {
    sqlx::query_scalar("SELECT to_jsonb(v) FROM violations v WHERE event_id = $1 FOR UPDATE")
        .bind(event)
        .fetch_optional(db)
        .await
        .map_err(database_error)
}

fn stored_json<T: DeserializeOwned>(raw: &Value) -> Result<Option<T>, ApiError> {
    match raw.as_str() {
        Some(raw) => serde_json::from_str(raw).map(Some).map_err(|_| conflict()),
        None if raw.is_null() => Ok(None),
        _ => Err(conflict()),
    }
}

struct VerifiedRepeat {
    id: i32,
    backfill: Option<String>,
}

async fn verify_repeat(
    media: &Arc<Store>,
    record: Value,
    evidence: &Evidence,
    digest: &str,
    prior_digest: Option<&str>,
) -> Result<VerifiedRepeat, ApiError> {
    if record["site_id"].as_i64() != Some(evidence.site_id)
        || record["stream_name"] != evidence.stream_name
    {
        return Err(conflict());
    }
    let id = record["id"]
        .as_i64()
        .and_then(|id| i32::try_from(id).ok())
        .ok_or_else(storage_error)?;
    if let Some(stored) = record["event_fingerprint"].as_str() {
        return if stored == digest || prior_digest == Some(stored) {
            Ok(VerifiedRepeat { id, backfill: None })
        } else {
            Err(conflict())
        };
    }
    let text = |key| record[key].as_str().ok_or_else(conflict);
    let stored_evidence = Evidence {
        event_id: evidence.event_id,
        site_id: evidence.site_id,
        site: evidence.site.clone(),
        stream_name: evidence.stream_name.clone(),
        detection_time: evidence
            .detection_time
            .map(|_| {
                text("detection_time")?
                    .parse::<DateTime<Utc>>()
                    .map_err(|_| conflict())
            })
            .transpose()?,
        prior_local_detection_time: None,
        model_id: text("model_id")?.to_owned(),
        model_version: text("model_version")?.to_owned(),
        warnings: stored_json(&record["warnings_json"])?,
        detections: stored_json(&record["detections_json"])?,
        cone_polygon: stored_json(&record["cone_polygon_json"])?,
        pole_polygon: stored_json(&record["pole_polygon_json"])?,
    };
    let stored_warnings = stored_evidence
        .warnings
        .as_ref()
        .map(|warnings| normalise_warnings(warnings.clone()));
    let path = record["image_path"]
        .as_str()
        .ok_or_else(conflict)?
        .to_owned();
    let media = media.clone();
    let existing_sha = tokio::task::spawn_blocking(move || media.file_digest(&path))
        .await
        .map_err(|_| storage_error())?
        .map_err(|_| conflict())?;
    let stored_digest = fingerprint::calculate(
        &stored_evidence,
        stored_evidence.detection_time,
        stored_warnings.as_ref(),
        existing_sha,
    )?;
    if stored_digest != digest && prior_digest != Some(stored_digest.as_str()) {
        return Err(conflict());
    }
    Ok(VerifiedRepeat {
        id,
        backfill: Some(stored_digest),
    })
}

#[derive(Deserialize)]
struct CurrentModel {
    version: String,
    #[serde(default = "enabled")]
    enabled: bool,
    capabilities: Vec<String>,
    #[serde(default)]
    tenant_ids: Vec<Uuid>,
    #[serde(default)]
    site_ids: Vec<i32>,
    #[serde(default = "legacy_roles")]
    roles: Vec<String>,
    classes: Value,
}

const fn enabled() -> bool {
    true
}
fn legacy_roles() -> Vec<String> {
    ["admin", "user", "guest", "super_admin"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

async fn model_snapshot(
    db: &mut PgConnection,
    evidence: &Evidence,
    scope: &Scope,
    tenant: Uuid,
    site: i32,
) -> Result<Vec<projection::Class>, ApiError> {
    let model: PgJson<CurrentModel> = sqlx::query_scalar(
        "SELECT definition FROM detection_model_catalog WHERE model_key = $1 FOR SHARE",
    )
    .bind(&evidence.model_id)
    .fetch_optional(&mut *db)
    .await
    .map_err(database_error)?
    .ok_or_else(model_denied)?;
    let model = model.0;
    if !model.enabled
        || !model.capabilities.iter().any(|item| item == "stream")
        || !model.roles.contains(&scope.role)
        || !model.tenant_ids.contains(&tenant)
        || !model.site_ids.contains(&site)
    {
        return Err(model_denied());
    }
    let snapshot = if model.version == evidence.model_version {
        model.classes
    } else {
        sqlx::query_scalar(
            "SELECT classes FROM public.detection_model_versions WHERE model_key = $1 AND version = $2",
        ).bind(&evidence.model_id).bind(&evidence.model_version).fetch_optional(db).await
            .map_err(database_error)?.ok_or_else(|| ApiError::new(StatusCode::CONFLICT, "MODEL_VERSION_UNKNOWN"))?
    };
    let classes = projection::classes(&snapshot)?;
    if evidence
        .detections
        .iter()
        .flatten()
        .any(|row| !classes.iter().any(|class| class.id as f64 == row[5]))
    {
        return Err(invalid("Detection class is not in model version"));
    }
    Ok(classes)
}

fn model_denied() -> ApiError {
    ApiError::new(StatusCode::FORBIDDEN, "Model unavailable or access denied")
}

fn conflict() -> ApiError {
    ApiError::new(StatusCode::CONFLICT, "event_id_conflict")
}

fn storage_error() -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "Evidence storage unavailable",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn retries_accept_an_exact_prior_local_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let image = b"historical evidence";
        std::fs::write(dir.path().join("evidence.jpg"), image).unwrap();
        let media = Arc::new(Store::new(dir.path().canonicalize().unwrap()));
        let evidence = Evidence {
            event_id: Some(Uuid::from_u128(1)),
            site_id: 42,
            site: "Site".into(),
            stream_name: "Camera".into(),
            detection_time: Some("2026-11-19T09:30:22.554Z".parse().unwrap()),
            prior_local_detection_time: Some("2026-11-19T01:30:22.554Z".parse().unwrap()),
            model_id: "model".into(),
            model_version: "version".into(),
            warnings: None,
            detections: None,
            cone_polygon: None,
            pole_polygon: None,
        };
        let image_sha = Sha256::digest(image).into();
        let digest =
            fingerprint::calculate(&evidence, evidence.detection_time, None, image_sha).unwrap();
        let prior = fingerprint::calculate(
            &evidence,
            evidence.prior_local_detection_time,
            None,
            image_sha,
        )
        .unwrap();
        let mut row = json!({
            "id": 10,
            "site_id": evidence.site_id,
            "stream_name": evidence.stream_name,
            "event_fingerprint": prior,
            "image_path": "evidence.jpg",
            "model_id": evidence.model_id,
            "model_version": evidence.model_version,
            "detection_time": evidence.prior_local_detection_time,
        });
        let verified = verify_repeat(&media, row.clone(), &evidence, &digest, Some(&prior))
            .await
            .unwrap();
        assert_eq!(verified.id, 10);
        assert!(verified.backfill.is_none());
        assert!(
            verify_repeat(&media, row.clone(), &evidence, &digest, None)
                .await
                .is_err()
        );
        assert!(
            verify_repeat(&media, row.clone(), &evidence, &digest, Some("different"))
                .await
                .is_err()
        );
        row["event_fingerprint"] = Value::Null;
        let verified = verify_repeat(&media, row, &evidence, &digest, Some(&prior))
            .await
            .unwrap();
        assert_eq!(verified.backfill, Some(prior));
    }
}

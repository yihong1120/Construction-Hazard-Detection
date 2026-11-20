use std::collections::HashSet;

use axum::http::StatusCode;
use chrono::{DateTime, Utc};
use construction_stream_contract::overlay_palette;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use sqlx::PgConnection;

use super::{Settings, database_error, review};
use crate::ApiError;

const UNKNOWN_COLOR: &str = "#9E9E9E";
const CONE_COLOR: &str = "#FF4081";
const POLE_COLOR: &str = "#448AFF";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Class {
    pub(super) id: i64,
    pub(super) code: String,
    pub(super) display_name: String,
    #[serde(default)]
    pub(super) color: String,
}

pub(super) fn classes(snapshot: &Value) -> Result<Vec<Class>, ApiError> {
    if snapshot.is_null() {
        return Ok(Vec::new());
    }
    let unavailable = || {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "HISTORICAL_MODEL_METADATA_UNAVAILABLE",
        )
    };
    let classes: Vec<Value> =
        serde_json::from_value(snapshot.clone()).map_err(|_| unavailable())?;
    let mut ids = HashSet::new();
    let mut codes = HashSet::new();
    classes
        .into_iter()
        .map(|mut item| {
            if let Some(item) = item.as_object_mut()
                && !item.get("color").is_some_and(Value::is_string)
            {
                item.remove("color");
            }
            let mut class: Class = serde_json::from_value(item).map_err(|_| unavailable())?;
            class.code = class.code.trim().to_owned();
            class.display_name = class.display_name.trim().to_owned();
            if class.id < 0
                || class.code.is_empty()
                || class.display_name.is_empty()
                || !ids.insert(class.id)
                || !codes.insert(class.code.clone())
                || class
                    .code
                    .trim_start_matches(['+', '-'])
                    .parse::<u64>()
                    .is_ok()
            {
                return Err(unavailable());
            }
            if class.color.len() == 7
                && class.color.starts_with('#')
                && class.color[1..]
                    .bytes()
                    .all(|value| value.is_ascii_hexdigit())
            {
                class.color.make_ascii_uppercase();
            } else if let Some(color) = overlay_palette().get(&class.code).and_then(Value::as_str) {
                class.color = color.to_owned();
            } else {
                let digest = format!("{:X}", Sha256::digest(class.code.as_bytes()));
                class.color = format!("#{}", &digest[..6]);
            }
            Ok(class)
        })
        .collect()
}

fn style(classes: &[Class]) -> Value {
    let mut palette = overlay_palette().clone();
    palette.extend(
        classes
            .iter()
            .map(|class| (class.code.clone(), json!(class.color))),
    );
    json!({
        "schema_version": 1, "class_colors": palette, "unknown_class_color": UNKNOWN_COLOR,
        "cone_polygon_color": CONE_COLOR, "pole_polygon_color": POLE_COLOR, "warning_color": "#F44336"
    })
}

fn media_url(image_path: &str, endpoint: &str) -> String {
    let query: String = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("image_path", image_path)
        .finish();
    format!("/{endpoint}?{query}")
}

fn warnings(raw: Option<&str>) -> (Map<String, Value>, Option<String>) {
    let stored: Map<String, Value> = raw
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default();
    let active: Map<String, Value> = stored
        .into_iter()
        .filter_map(|(key, warning)| {
            let count = warning["count"].as_i64().filter(|count| *count > 0)?;
            Some((key, json!({"count": count})))
        })
        .collect();
    let text = if active.is_empty() {
        None
    } else {
        Some(
            active
                .iter()
                .map(|(key, warning)| format!("{key}: {}", warning["count"]))
                .collect::<Vec<_>>()
                .join(", ")
                .chars()
                .take(200)
                .collect(),
        )
    };
    (active, text)
}

pub(super) type ListRow = (
    i32,
    String,
    String,
    DateTime<Utc>,
    String,
    Option<String>,
    bool,
    Option<String>,
    Option<String>,
    Option<Value>,
    Option<String>,
);

pub(super) fn list_item(row: ListRow) -> Result<Value, ApiError> {
    let (id, site, stream, stamp, path, raw_warning, flagged, status, version, snapshot, note) =
        row;
    let classes = classes(snapshot.as_ref().unwrap_or(&Value::Null))?;
    let (warnings, warning_text) = warnings(raw_warning.as_deref());
    Ok(json!({
        "id": id, "site_name": site, "stream_name": stream, "detection_time": stamp,
        "thumbnail_url": media_url(&path, "get_violation_thumbnail"), "warnings": warnings,
        "warning_text": warning_text, "is_flagged": flagged,
        "review_status": if flagged { status } else { None }, "feedback_note": note,
        "model_version": version, "class_metadata": classes, "overlay_style": style(&classes)
    }))
}

#[derive(Serialize)]
struct Detection {
    id: String,
    label: String,
    confidence: f64,
    bbox: [f64; 4],
}

fn valid_rows(raw: Option<&str>) -> Option<Vec<(usize, Vec<f64>)>> {
    let rows: Vec<Value> = serde_json::from_str(raw?).ok()?;
    let empty = rows.is_empty();
    let valid: Vec<_> = rows
        .into_iter()
        .enumerate()
        .filter_map(|(index, row)| {
            let fields: Vec<f64> = serde_json::from_value(row).ok()?;
            if fields.len() >= 7
                && fields[2] > fields[0]
                && fields[3] > fields[1]
                && (0.0..=1.0).contains(&fields[4])
                && fields[5] >= 0.0
                && fields[5].fract() == 0.0
                && fields[6] >= -1.0
                && fields[6].fract() == 0.0
            {
                Some((index, fields))
            } else {
                None
            }
        })
        .collect();
    (empty || !valid.is_empty()).then_some(valid)
}

pub(super) fn detection_is_present(raw: Option<&str>, target: &str) -> bool {
    valid_rows(raw).is_none_or(|rows| {
        rows.into_iter().any(|(index, row)| {
            target == format!("det_{index}")
                || (row[6] != -1.0 && target == format!("{:.0}", row[6]))
        })
    })
}

fn detections(raw: Option<&str>, classes: &[Class]) -> Option<Vec<Detection>> {
    valid_rows(raw).map(|rows| {
        rows.into_iter()
            .map(|(index, row)| {
                let class_id = row[5] as i64;
                let label = classes
                    .iter()
                    .find(|class| class.id == class_id)
                    .map(|class| class.code.clone())
                    .unwrap_or_else(|| format!("class-{class_id}"));
                Detection {
                    id: format!("det_{index}"),
                    label,
                    confidence: row[4],
                    bbox: [row[0], row[1], row[2], row[3]],
                }
            })
            .collect()
    })
}

fn normalized_box(bbox: [f64; 4], image_size: Option<(u32, u32)>) -> Option<Value> {
    let (width, height) = if bbox.into_iter().all(|value| (0.0..=1.0).contains(&value)) {
        (1.0, 1.0)
    } else {
        let (width, height) = image_size?;
        (f64::from(width), f64::from(height))
    };
    let [x1, y1, x2, y2] = bbox;
    let (x1, x2) = (x1.clamp(0.0, width), x2.clamp(0.0, width));
    let (y1, y2) = (y1.clamp(0.0, height), y2.clamp(0.0, height));
    if x1 >= x2 || y1 >= y2 {
        return None;
    }
    Some(
        json!({"x": x1 / width, "y": y1 / height, "w": (x2 - x1) / width, "h": (y2 - y1) / height}),
    )
}

fn box_from_json(value: &Value) -> Option<[f64; 4]> {
    let bbox: [f64; 4] = serde_json::from_value(value.clone()).ok()?;
    Some(bbox)
}

fn overlay_object(
    id: &str,
    label: Option<&str>,
    confidence: Option<f64>,
    bbox: Value,
    feedback: Option<&Value>,
    classes: &[Class],
) -> Value {
    let class = label.and_then(|key| classes.iter().find(|class| class.code == key));
    let semantics = class.map(|value| value.code.as_str());
    let display_name = class.map(|value| value.display_name.as_str()).or(label);
    let color = class
        .map(|value| value.color.as_str())
        .or_else(|| {
            label
                .and_then(|key| overlay_palette().get(key))
                .and_then(Value::as_str)
        })
        .unwrap_or(UNKNOWN_COLOR);
    let show = semantics != Some("safety_cone");
    json!({
        "object_id": id, "label": label, "display_name": display_name, "code": semantics,
        "color": color, "confidence": confidence, "bbox": bbox,
        "show_box": show, "show_label": show, "feedback_show_box": true, "feedback_show_label": true,
        "is_flagged": feedback.is_some(), "flag_reason": feedback.map(|item| &item["type"]),
        "flag_note": feedback.map(|item| &item["note"])
    })
}

fn objects(
    detections: Option<&[Detection]>,
    feedback: &[Value],
    size: Option<(u32, u32)>,
    classes: &[Class],
) -> Vec<Value> {
    let mut objects = Vec::new();
    for detection in detections.unwrap_or_default() {
        let Some(bbox) = normalized_box(detection.bbox, size) else {
            continue;
        };
        let matched = feedback
            .iter()
            .find(|entry| entry["target_detection_id"] == detection.id)
            .or_else(|| {
                feedback.iter().find(|entry| {
                    entry["target_detection_id"].is_null()
                        && box_from_json(&entry["original_bbox"]).is_some_and(|original| {
                            original
                                .into_iter()
                                .zip(detection.bbox)
                                .all(|(a, b)| (a - b).abs() <= 1e-6)
                        })
                })
            });
        objects.push(overlay_object(
            &detection.id,
            Some(&detection.label),
            Some(detection.confidence),
            bbox,
            matched,
            classes,
        ));
    }
    for entry in feedback
        .iter()
        .filter(|entry| entry["type"] == "false_negative")
    {
        let Some(bbox) =
            box_from_json(&entry["corrected_bbox"]).and_then(|bbox| normalized_box(bbox, size))
        else {
            continue;
        };
        objects.push(overlay_object(
            &format!("feedback_{}", entry["id"]),
            entry["corrected_label"].as_str(),
            None,
            bbox,
            Some(entry),
            classes,
        ));
    }
    objects
}

fn regions(record: &Value, size: Option<(u32, u32)>) -> (Vec<Value>, Value) {
    let mut regions = Vec::new();
    let mut status = Map::new();
    for (kind, source, label, color) in [
        ("cone", "cone_polygon_json", "安全錐管制區域", CONE_COLOR),
        ("pole", "pole_polygon_json", "電桿管制區域", POLE_COLOR),
    ] {
        let state = if let Some(raw) = record[source].as_str() {
            match serde_json::from_str::<Option<Vec<Vec<[f64; 2]>>>>(raw) {
                Ok(None) => "not_recorded",
                Ok(Some(rings)) if rings.is_empty() => "empty",
                Ok(Some(rings)) if rings.iter().all(nonzero_ring) => {
                    if let Some((width, height)) = size {
                        for (index, ring) in rings.into_iter().enumerate() {
                            let points: Vec<_> = ring
                                .into_iter()
                                .map(|[x, y]| [x / f64::from(width), y / f64::from(height)])
                                .collect();
                            regions.push(json!({"id": format!("{kind}_{index}"), "kind": kind, "display_name": label,
                                "points": points, "color": color, "fill_opacity": 0.4, "closed": true}));
                        }
                        "available"
                    } else {
                        "image_unavailable"
                    }
                }
                _ => "invalid",
            }
        } else {
            "not_recorded"
        };
        status.insert(kind.to_owned(), json!(state));
    }
    (regions, Value::Object(status))
}

fn nonzero_ring(ring: &Vec<[f64; 2]>) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let area: f64 = ring
        .iter()
        .zip(ring.iter().cycle().skip(1))
        .map(|([ax, ay], [bx, by])| ax * by - bx * ay)
        .sum();
    area.is_finite() && area.abs() >= 1e-9
}

pub(super) fn base_detail(record: Value) -> Result<Value, ApiError> {
    build_detail(&record, None, None, None)
}

pub(super) async fn detail(
    db: &mut PgConnection,
    settings: &Settings,
    record: Value,
) -> Result<Value, ApiError> {
    let id = record["id"]
        .as_i64()
        .and_then(|id| i32::try_from(id).ok())
        .ok_or_else(|| {
            ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "Invalid violation record")
        })?;
    let feedback: Vec<Value> = sqlx::query_scalar(
        "SELECT jsonb_build_object(
            'id', id, 'type', feedback_type, 'note', note,
            'target_detection_id', target_detection_id, 'original_label', original_label,
            'corrected_label', corrected_label, 'original_bbox', original_bbox,
            'corrected_bbox', corrected_bbox, 'confidence', confidence,
            'status', status, 'submitted_by', user_id, 'submitted_at', created_at
         ) FROM public.violation_feedback WHERE violation_id = $1 ORDER BY created_at DESC, id DESC",
    ).bind(id).fetch_all(&mut *db).await.map_err(database_error)?;
    let audit = if record["is_flagged"] == true {
        Some(review::audit(db, id).await?)
    } else {
        None
    };
    let size = if let Some(path) = record["image_path"].as_str() {
        settings.media.dimensions(path.to_owned()).await
    } else {
        None
    };
    build_detail(&record, Some(&feedback), audit, size)
}

fn build_detail(
    record: &Value,
    feedback: Option<&[Value]>,
    audit: Option<Vec<Value>>,
    size: Option<(u32, u32)>,
) -> Result<Value, ApiError> {
    let path = record["image_path"].as_str().ok_or_else(|| {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "Invalid violation record")
    })?;
    let (warnings, warning_text) = warnings(record["warnings_json"].as_str());
    drop(warnings);
    let classes = classes(&record["model_classes"])?;
    let detections = detections(record["detections_json"].as_str(), &classes);
    let note = feedback.and_then(|rows| {
        rows.iter()
            .find_map(|value| value["note"].as_str().filter(|item| !item.is_empty()))
    });
    let objects = feedback.map(|feedback| objects(detections.as_deref(), feedback, size, &classes));
    let (regions, status) = if feedback.is_some() {
        regions(record, size)
    } else {
        (vec![], json!({}))
    };
    Ok(json!({
        "id": record["id"], "site_name": record["site"], "stream_name": record["stream_name"],
        "detection_time": record["detection_time"], "created_at": record["created_at"],
        "image_path": path, "image_url": media_url(path, "get_violation_image"),
        "thumbnail_url": media_url(path, "get_violation_thumbnail"),
        "detection_items": record["detections_json"], "warnings": record["warnings_json"],
        "warning_text": warning_text, "cone_polygons": record["cone_polygon_json"],
        "pole_polygons": record["pole_polygon_json"], "detections": detections,
        "is_flagged": record["is_flagged"], "flag_reason": record["flag_reason"],
        "flagged_by": record["flagged_by"], "flagged_at": record["flagged_at"],
        "review_status": if record["is_flagged"] == true { &record["review_status"] } else { &Value::Null },
        "review_note": record["review_note"], "reviewed_by": record["reviewed_by"],
        "reviewed_at": record["reviewed_at"], "feedback_note": note, "feedbacks": feedback,
        "review_audit_logs": audit, "model_id": record["model_id"], "model_version": record["model_version"],
        "model_classes": record["model_classes"], "class_metadata": classes, "overlay_style": style(&classes),
        "overlay_schema_version": 1, "image_size": size.map(|(w, h)| json!({"width": w, "height": h})),
        "overlay_coordinates": {"space": "normalized", "bbox_format": "xywh", "polygon_format": "xy",
            "origin": "top_left", "reference": "original_image"},
        "overlay_objects": objects, "overlay_regions": regions, "region_status": status
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_boxes_keep_indices_after_invalid_rows() {
        let detections =
            detections(Some("[null,[0,2,20,32,0.5,1,7],[0,0,0,1,1,1,9]]"), &[]).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].id, "det_1");
        assert_eq!(detections[0].label, "class-1");
        assert_eq!(
            normalized_box(detections[0].bbox, Some((100, 100))).unwrap(),
            json!({"x": 0.0, "y": 0.02, "w": 0.2, "h": 0.3})
        );
    }

    #[test]
    fn historical_colours_use_snapshot_and_reject_duplicates() {
        let rows = json!([{"id": 1, "code": "hat", "display_name": "帽子", "color": "#37aaff"}]);
        let color = classes(&rows).unwrap().first().unwrap().color.clone();
        assert_eq!(color, "#37AAFF");
        let duplicate = json!([rows[0], rows[0]]);
        assert!(classes(&duplicate).is_err());
    }
}

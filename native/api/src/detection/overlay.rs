use axum::http::StatusCode;
use construction_postprocess_core::geometry::{Polygons, cone_regions, coordinates, pole_regions};
use serde_json::{Value, json};

use super::{
    Model,
    runtime::{Detection, Fault},
};
use crate::ApiError;

const CONE_COLOR: &str = "#FF4081";
const POLE_COLOR: &str = "#448AFF";
const UNKNOWN_COLOR: &str = "#9E9E9E";

#[derive(Debug, PartialEq)]
pub(super) struct Options {
    view: &'static str,
    compute: bool,
}

impl Options {
    pub(super) fn parse(view: Option<&str>, compute: Option<&str>) -> Result<Self, Fault> {
        let invalid = |detail| ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, detail);
        let view = match view {
            None | Some("regions") => "regions",
            Some("objects") => "objects",
            _ => return Err(invalid("Invalid detection view").into()),
        };
        let compute = match compute.map(str::to_ascii_lowercase).as_deref() {
            None | Some("1" | "true" | "t" | "on" | "y" | "yes") => true,
            Some("0" | "false" | "f" | "off" | "n" | "no") => false,
            _ => return Err(invalid("Invalid compute_regions flag").into()),
        };
        if view == "regions" && !compute {
            return Err(invalid("Region view requires compute_regions=true").into());
        }
        Ok(Self { view, compute })
    }

    pub(super) fn render(&self, detection: &Detection, model: &Model) -> Result<Value, Fault> {
        let mut colors = construction_stream_contract::overlay_palette().clone();
        colors.extend(
            model
                .classes
                .iter()
                .map(|class| (class.code.clone(), json!(class.color))),
        );
        let w = f64::from(detection.width);
        let h = f64::from(detection.height);
        let mut cones = Vec::new();
        let mut poles = Vec::new();
        let mut objects = Vec::with_capacity(detection.rows.len());
        for (index, &(x1, y1, x2, y2, confidence, class_id)) in detection.rows.iter().enumerate() {
            let (left, right) = ((x1 as f64).clamp(0., w), (x2 as f64).clamp(0., w));
            let (top, bottom) = ((y1 as f64).clamp(0., h), (y2 as f64).clamp(0., h));
            let (width, height) = (right - left, bottom - top);
            if width <= 0. || height <= 0. || !confidence.is_finite() {
                continue;
            }
            let class = model.classes.iter().find(|class| class.id == class_id);
            let code = class.map(|class| class.code.as_str());
            if self.compute {
                let x = (left + right) * 0.5;
                match code {
                    Some("safety_cone") => cones.push([x, (top + bottom) * 0.5]),
                    Some("utility_pole") => poles.push([x, bottom, height * 0.35]),
                    _ => (),
                }
            }
            let show = self.view == "objects" || code != Some("safety_cone");
            let fallback = format!("class-{class_id}");
            objects.push(json!({
                "object_id": format!("det_{index}"),
                "show_box": show,
                "show_label": show,
                "feedback_show_box": true,
                "feedback_show_label": true,
                "display_name": class.map_or(fallback.as_str(), |class| class.display_name.as_str()),
                "code": code,
                "color": class.map_or(UNKNOWN_COLOR, |class| class.color.as_str()),
                "label": code.unwrap_or(&fallback),
                "confidence": confidence,
                "bbox": {"x": left / w, "y": top / h, "w": width / w, "h": height / h},
                "is_flagged": false,
                "flag_reason": null,
                "flag_note": null,
            }));
        }

        let mut regions = Vec::new();
        let (cone, pole) = if self.compute {
            let cones = cone_regions(&cones).map_err(|_| Fault::unavailable())?;
            let poles = pole_regions(&poles).map_err(|_| Fault::unavailable())?;
            (
                append_regions(
                    &mut regions,
                    coordinates(&cones),
                    "cone",
                    "安全錐管制區域",
                    CONE_COLOR,
                    [w, h],
                ),
                append_regions(
                    &mut regions,
                    coordinates(&poles.0),
                    "pole",
                    "電桿管制區域",
                    POLE_COLOR,
                    [w, h],
                ),
            )
        } else {
            ("not_computed", "not_computed")
        };
        Ok(json!({
            "model_id": model.id,
            "model_version": model.version,
            "view": self.view,
            "overlay_schema_version": 1,
            "image_size": {"width": detection.width, "height": detection.height},
            "overlay_coordinates": {
                "space": "normalized", "bbox_format": "xywh", "polygon_format": "xy",
                "origin": "top_left", "reference": "original_image",
            },
            "class_metadata": model.classes,
            "overlay_style": {
                "schema_version": 1, "class_colors": colors, "unknown_class_color": UNKNOWN_COLOR,
                "cone_polygon_color": CONE_COLOR, "pole_polygon_color": POLE_COLOR,
                "warning_color": "#F44336",
            },
            "overlay_objects": objects,
            "overlay_regions": regions,
            "region_status": {"cone": cone, "pole": pole},
        }))
    }
}

fn append_regions(
    regions: &mut Vec<Value>,
    rings: Polygons,
    kind: &'static str,
    label: &'static str,
    color: &'static str,
    [width, height]: [f64; 2],
) -> &'static str {
    if rings.is_empty() {
        return "empty";
    }
    if !rings.iter().all(|ring| {
        let area: f64 = ring
            .iter()
            .zip(ring.iter().cycle().skip(1))
            .map(|([ax, ay], [bx, by])| ax * by - bx * ay)
            .sum();
        area.is_finite() && area.abs() >= 1e-9
    }) {
        return "invalid";
    }
    for (index, ring) in rings.iter().enumerate() {
        let points: Vec<_> = ring.iter().map(|[x, y]| [x / width, y / height]).collect();
        regions.push(json!({
            "id": format!("{kind}_{index}"),
            "kind": kind,
            "display_name": label,
            "points": points,
            "color": color,
            "fill_opacity": 0.4,
            "closed": true,
        }));
    }
    "available"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_default_regions_using_catalog_class_codes() {
        let model: Model = serde_json::from_value(json!({
            "id": "fixture", "display_name": "Fixture", "version": "v1",
            "engine": "fixture.engine", "sha256": "0".repeat(64),
            "classes": [
                {"id": 2, "code": "utility_pole", "display_name": "Pole", "color": "#448AFF"},
                {"id": 3, "code": "safety_cone", "display_name": "Cone", "color": "#FF4081"}
            ]
        }))
        .unwrap();
        let detection = Detection {
            width: 100,
            height: 100,
            rows: vec![(45, 10, 55, 120, 0.9, 2), (10, 10, 20, 20, 0.9, 3)],
        };
        let default = Options::parse(None, None).ok().unwrap();
        let response = default.render(&detection, &model).ok().unwrap();
        assert_eq!(response["view"], "regions");
        assert_eq!(
            response["region_status"],
            json!({"cone": "empty", "pole": "available"})
        );
        assert_eq!(response["overlay_regions"][0]["id"], "pole_0");
        assert_eq!(
            response["overlay_regions"][0]["points"]
                .as_array()
                .unwrap()
                .len(),
            257
        );
        assert_eq!(
            response["overlay_objects"][0]["bbox"],
            json!({"x":0.45,"y":0.10,"w":0.10,"h":0.90})
        );
        assert_eq!(response["overlay_objects"][1]["show_box"], false);
        assert_eq!(response["overlay_objects"][1]["feedback_show_box"], true);

        let objects = Options::parse(Some("objects"), Some("false")).ok().unwrap();
        let response = objects.render(&detection, &model).ok().unwrap();
        assert_eq!(response["view"], "objects");
        assert_eq!(response["overlay_regions"], json!([]));
        assert_eq!(response["region_status"]["pole"], "not_computed");
        assert_eq!(response["overlay_objects"][1]["show_box"], true);
        assert!(Options::parse(None, Some("no")).is_err());
    }
}

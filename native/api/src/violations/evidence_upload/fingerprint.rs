//! Byte-for-byte compatibility with the Python evidence fingerprint.
use std::io::{self, Write};

use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{Evidence, WarningMap, invalid};
use crate::ApiError;

pub(super) fn calculate(
    evidence: &Evidence,
    detection_time: Option<DateTime<Utc>>,
    warnings: Option<&WarningMap>,
    image_sha: [u8; 32],
) -> Result<String, ApiError> {
    let stamp = detection_time.map(|time| {
        let precision = if time.timestamp_subsec_micros() == 0 {
            SecondsFormat::Secs
        } else {
            SecondsFormat::Micros
        };
        time.to_rfc3339_opts(precision, false)
    });
    let mut payload = json!({
        "version": 1,
        "site_id": evidence.site_id,
        "camera": evidence.stream_name,
        "model_id": evidence.model_id,
        "model_version": evidence.model_version,
        "detection_time": stamp,
        "image_sha256": image_sha.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
        "warnings_json": warnings.cloned().unwrap_or_default(),
        "detections_json": evidence.detections.as_deref().unwrap_or_default(),
        "cone_polygon_json": evidence.cone_polygon.as_ref().cloned().unwrap_or_else(|| json!([])),
        "pole_polygon_json": evidence.pole_polygon.as_ref().cloned().unwrap_or_else(|| json!([])),
    });
    payload.sort_all_objects();
    let mut bytes = Vec::new();
    payload
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            CanonicalJson,
        ))
        .map_err(|_| invalid("Invalid evidence metadata"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

// json.dumps(..., sort_keys=True, separators=(',', ':')) uses ASCII unicode
// escapes, while _numbers in the Python implementation normalizes integral
// floats to integers before serializing the payload.
struct CanonicalJson;

impl serde_json::ser::Formatter for CanonicalJson {
    fn write_string_fragment<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        value: &str,
    ) -> io::Result<()> {
        let mut start = 0;
        for (index, character) in value.char_indices() {
            if character < '\u{7f}' {
                continue;
            }
            writer.write_all(value[start..index].as_bytes())?;
            for unit in character.encode_utf16(&mut [0; 2]) {
                write!(writer, "\\u{unit:04x}")?;
            }
            start = index + character.len_utf8();
        }
        writer.write_all(value[start..].as_bytes())
    }

    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        if value == 0.0 {
            return writer.write_all(b"0");
        }
        if value.fract() == 0.0 {
            return write!(writer, "{value:.0}");
        }
        let repr = format!("{value:?}");
        if let Some((mantissa, exponent)) = repr.split_once('e') {
            let exponent: i32 = exponent.parse().expect("finite float exponent");
            write!(writer, "{mantissa}e{exponent:+03}")
        } else {
            writer.write_all(repr.as_bytes())
        }
    }
}

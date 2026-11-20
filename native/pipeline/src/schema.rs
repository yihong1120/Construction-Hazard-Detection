//! Published-model JSON normalization and Python-compatible revision hashing.
use anyhow::{Context, Result, ensure};
use construction_stream_contract::overlay_palette;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

pub fn color(code: &str) -> String {
    overlay_palette()
        .get(code)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            format!(
                "#{}",
                &hex::encode(Sha256::digest(code.as_bytes()))[..6].to_uppercase()
            )
        })
}
pub fn normalize(mut value: Value) -> Result<Value> {
    let object = value
        .as_object_mut()
        .context("Invalid published model definition")?;
    object.remove("id");
    let allowed = [
        "display_name",
        "version",
        "hf_repo",
        "hf_filename",
        "hf_revision",
        "artifact",
        "sha256",
        "enabled",
        "capabilities",
        "tenant_ids",
        "site_ids",
        "roles",
        "classes",
    ];
    ensure!(
        object.keys().all(|key| allowed.contains(&key.as_str())),
        "Unknown published model fields"
    );
    for key in ["hf_repo", "hf_filename", "hf_revision"] {
        object.entry(key).or_insert(Value::Null);
    }
    object.entry("enabled").or_insert(json!(true));
    object.entry("tenant_ids").or_insert(json!([]));
    object.entry("site_ids").or_insert(json!([]));
    object
        .entry("roles")
        .or_insert(json!(["admin", "user", "guest", "super_admin"]));
    for key in ["display_name", "version", "artifact", "sha256"] {
        ensure!(
            object
                .get(key)
                .and_then(Value::as_str)
                .is_some_and(|v| !v.is_empty()),
            "Missing published model identity"
        );
    }
    let classes = object
        .get_mut("classes")
        .and_then(Value::as_array_mut)
        .context("Missing published classes")?;
    let mut seen = std::collections::HashSet::new();
    for class in classes {
        let class = class.as_object_mut().context("Invalid class definition")?;
        ensure!(
            class
                .keys()
                .all(|key| ["id", "code", "display_name", "color"].contains(&key.as_str())),
            "Unknown class fields"
        );
        ensure!(
            class
                .get("id")
                .and_then(Value::as_u64)
                .is_some_and(|id| seen.insert(id)),
            "Invalid or duplicate class ID"
        );
        let code = class
            .get("code")
            .and_then(Value::as_str)
            .context("Missing class code")?;
        let raw = class.get("color").and_then(Value::as_str);
        let color = raw
            .filter(|v| {
                v.len() == 7 && v.starts_with('#') && v[1..].bytes().all(|c| c.is_ascii_hexdigit())
            })
            .map(str::to_uppercase)
            .unwrap_or_else(|| color(code));
        class.insert("color".into(), json!(color));
    }
    Ok(value)
}
struct PythonFormat;
impl serde_json::ser::Formatter for PythonFormat {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}
pub fn revision(value: &Value) -> Result<String> {
    let mut bytes = Vec::new();
    value.serialize(&mut serde_json::Serializer::with_formatter(
        &mut bytes,
        PythonFormat,
    ))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_revision_preserves_python_spaces_and_unicode() {
        let value = json!({"z":"繁中","a":[1,2]});
        let expected = "{\"a\": [1, 2], \"z\": \"繁中\"}";
        assert_eq!(
            revision(&value).unwrap(),
            hex::encode(Sha256::digest(expected.as_bytes()))
        );
    }
}

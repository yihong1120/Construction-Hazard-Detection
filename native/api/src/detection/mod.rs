use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

const MAX_UPLOAD_BYTES: usize = 32 * 1024 * 1024;
const MAX_IMAGE_PIXELS: u64 = 64 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub default_model_id: Option<String>,
    #[serde(default)]
    pub device: i32,
    #[serde(default = "default_upload_bytes")]
    pub max_upload_bytes: usize,
    #[serde(default = "default_image_pixels")]
    pub max_image_pixels: u64,
    #[serde(default = "default_capacity")]
    pub capacity: usize,
    pub catalog_tenant_id: Option<String>,
    pub models: Vec<Model>,
}

fn default_upload_bytes() -> usize {
    20 * 1024 * 1024
}

fn default_image_pixels() -> u64 {
    MAX_IMAGE_PIXELS
}

fn default_capacity() -> usize {
    8
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub id: String,
    pub display_name: String,
    pub version: String,
    pub engine: PathBuf,
    pub sha256: String,
    #[serde(default = "default_imgsz")]
    pub imgsz: usize,
    pub classes: Vec<Class>,
    #[serde(default)]
    pub allowed_subjects: HashSet<String>,
    pub source_artifact: Option<SourceArtifact>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceArtifact {
    pub path: PathBuf,
    pub sha256: String,
}

fn default_imgsz() -> usize {
    640
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Class {
    pub id: u32,
    pub code: String,
    pub display_name: String,
    pub color: String,
}

fn nonempty_bounded(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 256
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_UPLOAD_BYTES).contains(&self.max_upload_bytes)
                && (1..=MAX_IMAGE_PIXELS).contains(&self.max_image_pixels)
                && (1..=16).contains(&self.capacity)
                && self.device >= 0,
            "Invalid detection capacity, image limits, or CUDA device"
        );
        ensure!(
            (1..=16).contains(&self.models.len()),
            "Configure between one and sixteen pinned detection models"
        );
        ensure!(
            self.catalog_tenant_id
                .as_deref()
                .is_none_or(nonempty_bounded),
            "Invalid detection catalog tenant"
        );
        let mut ids = HashSet::new();
        for model in &self.models {
            ensure!(
                model
                    .engine
                    .extension()
                    .is_some_and(|item| item == "engine")
                    && valid_sha256(&model.sha256)
                    && (32..=4096).contains(&model.imgsz),
                "Detection model must name a TensorRT .engine, size, and SHA-256"
            );
            ensure!(
                nonempty_bounded(&model.id)
                    && nonempty_bounded(&model.display_name)
                    && nonempty_bounded(&model.version)
                    && ids.insert(model.id.as_str()),
                "Invalid or duplicate detection model identity"
            );
            ensure!(
                (self.catalog_tenant_id.is_some() || !model.allowed_subjects.is_empty())
                    && model
                        .allowed_subjects
                        .iter()
                        .all(|subject| nonempty_bounded(subject)),
                "Each detection model requires catalog authorization or an explicit subject allowlist"
            );
            if let Some(artifact) = &model.source_artifact {
                ensure!(
                    artifact.path.is_absolute()
                        && artifact
                            .path
                            .extension()
                            .is_some_and(|extension| extension == "pt")
                        && valid_sha256(&artifact.sha256),
                    "Detection source artifact must be an absolute .pt path and SHA-256"
                );
            }
            ensure!(model.classes.len() <= 4096, "Too many detection classes");
            let mut class_ids = HashSet::new();
            let mut class_codes = HashSet::new();
            for class in &model.classes {
                ensure!(
                    nonempty_bounded(&class.code)
                        && nonempty_bounded(&class.display_name)
                        && class_ids.insert(class.id)
                        && class_codes.insert(class.code.as_str())
                        && class.color.len() == 7
                        && class.color.starts_with('#')
                        && class.color.as_bytes()[1..]
                            .iter()
                            .all(u8::is_ascii_hexdigit),
                    "Invalid or duplicate detection class"
                );
            }
        }
        ensure!(
            self.default_model_id
                .as_deref()
                .is_none_or(|default| ids.contains(default)),
            "Unknown default detection model"
        );
        Ok(())
    }
}

pub use runtime::{Inference, routes, start};

mod management;
mod overlay;
mod preparation;
mod runtime;
mod websocket;

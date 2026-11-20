use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default = "default_true")]
    pub exact_cpu_resize: bool,
    #[serde(default)]
    pub metadata_async: bool,
    #[serde(default = "default_preprocess_workers")]
    pub preprocess_workers: usize,
    #[serde(default = "default_cpu_workers")]
    pub cpu_workers: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_workers_auto: Option<CpuWorkersAuto>,
    #[serde(default = "default_overlay_age")]
    pub overlay_max_age_ms: u64,
    pub state_dir: PathBuf,
    pub models: Vec<Model>,
    pub streams: Vec<Stream>,
    #[serde(default)]
    pub database: Option<Database>,
    #[serde(default)]
    pub redis: Option<Redis>,
    #[serde(default)]
    pub delivery: Delivery,
    #[serde(default)]
    pub outbox: OutboxSettings,
    #[serde(default)]
    pub gpu: Gpu,
    #[serde(default)]
    pub fonts: BTreeMap<String, Font>,
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default = "default_pending")]
    pub pending_requests: usize,
    #[serde(default)]
    pub lossless: bool,
    #[serde(default)]
    pub max_frames: Option<u64>,
    #[serde(default)]
    pub experimental_replay: bool,
    #[serde(default)]
    pub capture: CaptureSettings,
    #[serde(default)]
    pub record_detections: bool,
}
fn default_preprocess_workers() -> usize {
    1
}
fn default_cpu_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CpuWorkersAuto {
    pub cameras_per_worker: usize,
    pub min_workers: usize,
    pub reserve_cpus: usize,
}

impl Default for CpuWorkersAuto {
    fn default() -> Self {
        Self {
            cameras_per_worker: 2,
            min_workers: 1,
            reserve_cpus: 2,
        }
    }
}

impl CpuWorkersAuto {
    fn validate(self, maximum: usize) -> Result<()> {
        ensure!(
            (1..=64).contains(&maximum)
                && self.cameras_per_worker > 0
                && (1..=maximum).contains(&self.min_workers),
            "Automatic CPU workers require a positive camera ratio and a minimum within cpu_workers"
        );
        Ok(())
    }

    pub(crate) fn limit(self, maximum: usize, available_cpus: usize) -> Result<usize> {
        self.validate(maximum)?;
        Ok(maximum.min(available_cpus.saturating_sub(self.reserve_cpus).max(1)))
    }

    pub(crate) fn workers(self, cameras: usize, limit: usize) -> usize {
        cameras
            .div_ceil(self.cameras_per_worker)
            .clamp(self.min_workers.min(limit), limit)
    }
}

#[cfg(test)]
mod cpu_worker_tests {
    use super::CpuWorkersAuto;

    #[test]
    fn automatic_cpu_budget_rounds_up_within_configured_and_cpu_limits() {
        let policy = CpuWorkersAuto::default();
        let limit = policy.limit(16, 20).unwrap();
        assert_eq!(limit, 16);
        for (cameras, workers) in [(0, 1), (2, 1), (3, 2), (13, 7), (32, 16), (50, 16)] {
            assert_eq!(policy.workers(cameras, limit), workers);
        }
        assert_eq!(policy.limit(8, 28).unwrap(), 8);
        let minimum = CpuWorkersAuto {
            min_workers: 4,
            ..policy
        };
        assert_eq!(minimum.workers(1, limit), 4);
        assert_eq!(minimum.limit(16, 4).unwrap(), 2);
        assert_eq!(minimum.workers(13, 2), 2);
        assert_eq!(minimum.limit(16, 1).unwrap(), 1);
        assert!(minimum.limit(3, 20).is_err());
        assert!(
            CpuWorkersAuto {
                cameras_per_worker: 0,
                ..policy
            }
            .limit(8, 20)
            .is_err()
        );
    }
}
fn default_overlay_age() -> u64 {
    1000
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureSettings {
    pub rust_rtsp: bool,
    pub host_yuv: bool,
    pub nvdec_device: Option<i32>,
    pub max_convert_fps: u32,
    pub read_timeout_ms: u64,
    pub timestamp_stall_seconds: u64,
    pub freeze_seconds: u64,
    pub max_lag_ms: u64,
    pub buffer_frames: usize,
}
impl Default for CaptureSettings {
    fn default() -> Self {
        Self {
            rust_rtsp: false,
            host_yuv: false,
            nvdec_device: None,
            max_convert_fps: 0,
            read_timeout_ms: 10_000,
            timestamp_stall_seconds: 45,
            freeze_seconds: 15,
            max_lag_ms: 2000,
            buffer_frames: 6,
        }
    }
}
fn default_batch() -> usize {
    1
}
fn default_pending() -> usize {
    32
}
fn default_size() -> usize {
    640
}
fn default_fps() -> u32 {
    15
}
fn default_languages() -> Vec<String> {
    vec!["en".into()]
}
fn default_true() -> bool {
    true
}
fn default_end() -> u32 {
    18
}
fn default_start() -> u32 {
    7
}
fn default_timezone() -> String {
    "Asia/Taipei".into()
}
fn default_cooldown() -> u64 {
    300
}
fn default_limit() -> u64 {
    512 * 1024 * 1024
}
fn default_namespace() -> String {
    "gpu0".into()
}
fn default_nvenc() -> usize {
    6
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    #[serde(default)]
    pub catalog_revision: Option<String>,
    #[serde(default)]
    pub class_metadata: Vec<serde_json::Value>,
    pub key: String,
    pub artifact: PathBuf,
    pub sha256: String,
    pub version: String,
    #[serde(default)]
    pub source_artifact: Option<PathBuf>,
    #[serde(default)]
    pub source_sha256: Option<String>,
    #[serde(default = "default_size")]
    pub imgsz: usize,
    #[serde(default)]
    pub replay: Option<PathBuf>,
    #[serde(default)]
    pub labels: BTreeMap<u32, BTreeMap<String, String>>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Stream {
    pub id: String,
    pub site_id: i64,
    pub site: String,
    pub name: String,
    pub source: String,
    pub model: String,
    pub width: u32,
    pub height: u32,
    #[serde(default = "default_fps")]
    pub fps: u32,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub expires: Option<DateTime<Utc>>,
    #[serde(default = "default_start")]
    pub work_start_hour: u32,
    #[serde(default = "default_end")]
    pub work_end_hour: u32,
    #[serde(default = "default_timezone")]
    pub timezone: String,
    #[serde(default = "default_languages")]
    pub languages: Vec<String>,
    #[serde(default)]
    pub rules: Rules,
    #[serde(default)]
    pub tracker: TrackerKind,
    #[serde(default)]
    pub output: Output,
    #[serde(default = "default_cooldown")]
    pub evidence_cooldown_seconds: u64,
}

impl Stream {
    pub(crate) fn enabled_at(&self, now: DateTime<Utc>) -> bool {
        self.enabled && self.expires.is_none_or(|expires| expires > now)
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewOutput {
    pub clean: Option<String>,
    pub annotated: Option<String>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MediaDefaults {
    pub publish_base: String,
    pub languages: Vec<String>,
    #[serde(default)]
    pub nvenc: bool,
    #[serde(default)]
    pub on_demand: bool,
    #[serde(default)]
    pub clean_passthrough: bool,
    pub preview: Option<PreviewOutput>,
}
pub fn output_path(template: &str, stream: &str, language: &str) -> String {
    use base64::Engine;
    template
        .replace("{stream}", stream)
        .replace(
            "{language_b64}",
            &base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(language),
        )
        .replace("{language}", language)
}

pub(crate) fn output_language(language: &str, on_demand: bool) -> &str {
    if on_demand {
        crate::locale::canonical(language).unwrap_or(language)
    } else {
        language
    }
}

fn valid_languages(languages: &[String], on_demand: bool) -> bool {
    let mut variants = HashSet::new();
    !languages.is_empty()
        && languages.len() <= 8
        && languages.iter().all(|language| {
            let code = if on_demand {
                crate::locale::canonical(language)
            } else {
                Some(language.as_str())
            };
            code.is_some_and(|code| identifier(code) && variants.insert(code))
        })
}
#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackerKind {
    #[default]
    Centroid,
    Hungarian,
}
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    pub annotated: Option<String>,
    pub clean: Option<String>,
    #[serde(default)]
    pub nvenc: bool,
    #[serde(default)]
    pub clean_passthrough: bool,
    #[serde(default)]
    pub client_overlay: bool,
    #[serde(default)]
    pub on_demand: bool,
    #[serde(default)]
    pub preview: Option<PreviewOutput>,
}
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Rules {
    pub detect_no_safety_vest_or_helmet: bool,
    pub detect_near_machinery_or_vehicle: bool,
    pub detect_in_restricted_area: bool,
    pub detect_in_utility_pole_restricted_area: bool,
    pub detect_machinery_close_to_pole: bool,
}
impl Default for Rules {
    fn default() -> Self {
        Self {
            detect_no_safety_vest_or_helmet: true,
            detect_near_machinery_or_vehicle: true,
            detect_in_restricted_area: true,
            detect_in_utility_pole_restricted_area: false,
            detect_machinery_close_to_pole: false,
        }
    }
}
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutboxSettings {
    pub url_env: Option<String>,
    pub namespace: Option<String>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Database {
    pub url_env: String,
    #[serde(default)]
    pub media: Option<MediaDefaults>,
    pub engine_manifest: PathBuf,
    #[serde(default = "default_fps")]
    pub refresh_seconds: u32,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default = "default_fps")]
    pub fps: u32,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Redis {
    pub url_env: String,
    #[serde(default)]
    pub prefix: String,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Font {
    pub path: PathBuf,
    #[serde(default)]
    pub family: Option<String>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Gpu {
    pub device: i32,
    pub namespace: String,
    pub nvenc_sessions: usize,
    pub lease_dir: Option<PathBuf>,
    pub inference_slots: usize,
    pub memory_limit_mb: usize,
    pub model_reservation_mb: usize,
    pub asynchronous: bool,
    pub exact_resize: bool,
    pub direct_pinned: bool,
    pub pipeline_depth: usize,
}
impl Default for Gpu {
    fn default() -> Self {
        Self {
            device: 0,
            namespace: default_namespace(),
            nvenc_sessions: default_nvenc(),
            lease_dir: None,
            inference_slots: 0,
            memory_limit_mb: 0,
            model_reservation_mb: 0,
            asynchronous: false,
            exact_resize: false,
            direct_pinned: false,
            pipeline_depth: 1,
        }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Delivery {
    pub enabled: bool,
    pub upload_url: Option<String>,
    pub notification_url: Option<String>,
    pub token_url: Option<String>,
    pub client_id_env: Option<String>,
    pub client_secret_env: Option<String>,
    pub outbox_limit_bytes: u64,
    pub max_attempts: u32,
    pub evidence_workers: usize,
    pub notification_workers: usize,
}
impl Default for Delivery {
    fn default() -> Self {
        Self {
            enabled: false,
            upload_url: None,
            notification_url: None,
            token_url: None,
            client_id_env: None,
            client_secret_env: None,
            outbox_limit_bytes: default_limit(),
            max_attempts: 20,
            evidence_workers: 2,
            notification_workers: 2,
        }
    }
}
impl Delivery {
    pub(crate) fn validate_workers(&self) -> Result<()> {
        ensure!(
            (1..=32).contains(&self.evidence_workers)
                && (1..=32).contains(&self.notification_workers),
            "Delivery worker counts must be between 1 and 32"
        );
        Ok(())
    }
}

pub fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 100
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
}
pub fn validate_endpoint(value: &str) -> Result<()> {
    let url = reqwest::Url::parse(value).context("Invalid HTTP endpoint")?;
    ensure!(
        matches!(url.scheme(), "http" | "https")
            && url.username().is_empty()
            && url.password().is_none(),
        "HTTP endpoint requires http(s) without embedded credentials"
    );
    Ok(())
}
impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(path)?)
            .context("Invalid pipeline configuration")?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.gpu.direct_pinned || self.gpu.asynchronous,
            "Direct pinned preparation requires asynchronous execution"
        );
        ensure!(
            !self.capture.host_yuv || self.capture.nvdec_device.is_none(),
            "Host YUV and NVDEC are separate capture modes"
        );
        if let Some(device) = self.capture.nvdec_device {
            ensure!(
                cfg!(feature = "tensorrt") && !self.experimental_replay,
                "NVDEC capture requires a TensorRT build and real inference"
            );
            ensure!(
                device == self.gpu.device && self.gpu.asynchronous,
                "NVDEC capture requires the inference CUDA device and asynchronous execution"
            );
        }
        ensure!(
            !self.gpu.exact_resize || self.gpu.asynchronous,
            "Exact GPU resize requires asynchronous execution"
        );
        ensure!(
            (1..=16).contains(&self.preprocess_workers),
            "Preprocess worker budget must be 1..16"
        );
        ensure!(
            (1..=2).contains(&self.gpu.pipeline_depth),
            "GPU pipeline depth must be 1 or 2"
        );
        ensure!(
            self.gpu.pipeline_depth == 1 || self.gpu.asynchronous,
            "Two in-flight batches require asynchronous GPU execution"
        );
        ensure!(
            self.gpu.pipeline_depth == 1 || self.gpu.inference_slots == 0,
            "Global inference admission requires pipeline depth 1"
        );
        ensure!(
            self.outbox
                .namespace
                .as_ref()
                .is_none_or(|s| !s.is_empty() && s.len() <= 256),
            "Invalid outbox namespace"
        );
        ensure!(
            self.outbox.url_env.as_ref().is_none_or(|s| !s.is_empty()),
            "Invalid outbox database environment name"
        );
        ensure!(self.state_dir.is_absolute(), "state_dir must be absolute");
        ensure!(
            (1..=64).contains(&self.cpu_workers),
            "Invalid CPU worker budget"
        );
        if let Some(policy) = self.cpu_workers_auto {
            policy.validate(self.cpu_workers)?;
        }
        ensure!(
            (100..=60_000).contains(&self.overlay_max_age_ms),
            "Invalid overlay age limit"
        );
        ensure!(
            self.max_frames.is_none_or(|n| n > 0),
            "max_frames must be positive"
        );
        ensure!(
            (100..=120_000).contains(&self.capture.read_timeout_ms)
                && (1..=600).contains(&self.capture.timestamp_stall_seconds)
                && (1..=600).contains(&self.capture.freeze_seconds)
                && self.capture.max_lag_ms <= 60_000
                && self.capture.max_convert_fps <= 120
                && (3..=16).contains(&self.capture.buffer_frames),
            "Invalid capture timeout"
        );
        if self.experimental_replay {
            if let Some(redis) = &self.redis {
                ensure!(
                    !redis.prefix.is_empty(),
                    "Replay metadata requires an isolated Redis prefix"
                );
            }
        }
        ensure!(
            (1..=64).contains(&self.batch_size) && (1..=1024).contains(&self.pending_requests),
            "Invalid worker limits"
        );
        ensure!(
            self.gpu.device >= 0
                && identifier(&self.gpu.namespace)
                && (1..=64).contains(&self.gpu.nvenc_sessions),
            "Invalid GPU configuration"
        );
        ensure!(
            self.gpu.memory_limit_mb == 0
                || (self.gpu.memory_limit_mb >= 64 && self.gpu.model_reservation_mb > 0),
            "Memory admission requires explicit model reservation"
        );
        let mut models = HashSet::new();
        for model in &self.models {
            ensure!(
                identifier(&model.key) && models.insert(&model.key),
                "Duplicate or invalid model key"
            );
            ensure!(
                !model.version.is_empty() && (32..=4096).contains(&model.imgsz),
                "Invalid model version/shape"
            );
            if model.replay.is_some() {
                ensure!(
                    self.experimental_replay,
                    "Replay detector requires experimental_replay=true"
                );
            } else {
                ensure!(
                    model.sha256.len() == 64 && hex::decode(&model.sha256).is_ok(),
                    "Model requires SHA256"
                );
            }
            ensure!(
                model.source_artifact.is_some() == model.source_sha256.is_some(),
                "Source artifact and digest must be paired"
            );
        }
        let mut streams = HashSet::new();
        let mut outputs = HashSet::new();
        for stream in &self.streams {
            ensure!(
                stream.site.len() <= 256 && stream.name.len() <= 256,
                "Camera labels exceed text limit"
            );
            if stream.output.on_demand || stream.output.clean_passthrough {
                ensure!(
                    !self.lossless
                        && stream
                            .output
                            .clean
                            .as_ref()
                            .is_some_and(|t| t.starts_with("rtsp://") || t.starts_with("rtsps://")),
                    "Demand/passthrough requires live RTSP clean output"
                );
            }
            if stream.output.clean_passthrough {
                ensure!(
                    !self.capture.rust_rtsp
                        && (stream.source.starts_with("rtsp://")
                            || stream.source.starts_with("rtsps://")),
                    "Encoded passthrough currently requires libav RTSP ingress"
                );
            }
            ensure!(
                !stream.output.on_demand || self.redis.is_some(),
                "On-demand outputs require playback Redis leases"
            );
            ensure!(
                !stream.output.client_overlay || self.redis.is_some(),
                "Browser overlay requires Redis metadata"
            );
            if stream.output.annotated.is_some() || stream.output.clean.is_some() {
                ensure!(
                    stream.width % 2 == 0 && stream.height % 2 == 0,
                    "H264 outputs require even dimensions"
                );
            }
            if stream.languages.len() > 1 {
                if let Some(target) = &stream.output.annotated {
                    ensure!(
                        (target.contains("{language}") || target.contains("{language_b64}")),
                        "Multiple languages require a distinct output template"
                    );
                }
            }
            ensure!(
                identifier(&stream.id) && streams.insert(&stream.id),
                "Duplicate or invalid camera id"
            );
            ensure!(models.contains(&stream.model), "Unknown model for camera");
            ensure!(
                stream.site_id > 0 && !stream.site.is_empty() && !stream.name.is_empty(),
                "Camera requires site identity"
            );
            ensure!(
                stream.width > 0
                    && stream.height > 0
                    && stream.width <= 16384
                    && stream.height <= 16384
                    && u64::from(stream.width) * u64::from(stream.height) <= 64 * 1024 * 1024,
                "Invalid frame dimensions"
            );
            ensure!(
                (1..=120).contains(&stream.fps)
                    && stream.work_start_hour <= 24
                    && stream.work_end_hour <= 24,
                "Invalid camera schedule/FPS"
            );
            ensure!(
                stream.timezone.parse::<chrono_tz::Tz>().is_ok(),
                "Invalid camera timezone"
            );
            ensure!(
                valid_languages(&stream.languages, stream.output.on_demand),
                "Invalid language variants"
            );
            ensure!(
                stream.source.starts_with("rtsp://")
                    || stream.source.starts_with("rtsps://")
                    || stream.source.starts_with("http://")
                    || stream.source.starts_with("https://")
                    || Path::new(&stream.source).is_file(),
                "Source must be RTSP, HTTP(S), or an existing local video"
            );
            if let Some(preview) = &stream.output.preview {
                validate_preview(preview)?;
                if let Some(target) = &preview.clean {
                    ensure!(
                        outputs.insert(output_path(target, &stream.id, "clean")),
                        "Duplicate preview target"
                    );
                }
                if let Some(target) = &preview.annotated {
                    ensure!(
                        stream.languages.len() == 1
                            || target.contains("{language}")
                            || target.contains("{language_b64}"),
                        "Preview languages need distinct targets"
                    );
                    for language in &stream.languages {
                        let language = output_language(language, stream.output.on_demand);
                        ensure!(
                            outputs.insert(output_path(target, &stream.id, language)),
                            "Duplicate preview target"
                        );
                    }
                }
            }
            for target in [&stream.output.annotated, &stream.output.clean]
                .into_iter()
                .flatten()
            {
                ensure!(
                    target.starts_with("rtsp://") || Path::new(target).is_absolute(),
                    "Output must be RTSP or absolute local path"
                );
            }
            for language in &stream.languages {
                if let Some(target) = &stream.output.annotated {
                    let language = output_language(language, stream.output.on_demand);
                    ensure!(
                        outputs.insert(output_path(target, &stream.id, language)),
                        "Camera outputs collide"
                    );
                }
            }
            if let Some(target) = &stream.output.clean {
                ensure!(
                    outputs.insert(
                        target
                            .replace("{stream}", &stream.id)
                            .replace("{language}", "clean")
                    ),
                    "Camera outputs collide"
                );
            }
        }
        if self.delivery.enabled {
            self.delivery.validate_workers()?;
            let url = self
                .delivery
                .upload_url
                .as_ref()
                .context("Delivery requires upload_url")?;
            validate_endpoint(url)?;
            if self.experimental_replay || self.delivery.token_url.is_none() {
                for endpoint in [&self.delivery.upload_url, &self.delivery.notification_url]
                    .into_iter()
                    .flatten()
                {
                    let endpoint = reqwest::Url::parse(endpoint)?;
                    let host = endpoint.host_str().unwrap_or("").trim_matches(['[', ']']);
                    ensure!(
                        host == "localhost"
                            || host
                                .parse::<std::net::IpAddr>()
                                .is_ok_and(|ip| ip.is_loopback()),
                        "Replay or unauthenticated delivery is restricted to loopback endpoints"
                    );
                }
            }
            if let Some(url) = &self.delivery.notification_url {
                validate_endpoint(url)?;
            }
            if let Some(url) = &self.delivery.token_url {
                validate_endpoint(url)?;
                ensure!(
                    self.delivery.client_id_env.is_some()
                        && self.delivery.client_secret_env.is_some(),
                    "OAuth requires credential environment names"
                );
            }
        }
        ensure!(
            self.delivery.max_attempts > 0
                && (1024..=i64::MAX as u64).contains(&self.delivery.outbox_limit_bytes),
            "Invalid outbox bounds"
        );
        if let Some(media) = self.database.as_ref().and_then(|d| d.media.as_ref()) {
            ensure!(
                media.publish_base.starts_with("rtsp://")
                    && valid_languages(&media.languages, media.on_demand),
                "Invalid catalog media defaults"
            );
            if let Some(preview) = &media.preview {
                validate_preview(preview)?;
            }
            ensure!(
                !media.on_demand || self.redis.is_some(),
                "Catalog media demand needs Redis"
            );
        }
        if self.streams.is_empty() && self.database.is_none() {
            bail!("At least one camera or database source is required");
        }
        Ok(())
    }
}

fn validate_preview(preview: &PreviewOutput) -> Result<()> {
    ensure!(
        preview.width > 0
            && preview.height > 0
            && preview.width % 2 == 0
            && preview.height % 2 == 0
            && preview.width <= 16384
            && preview.height <= 16384
            && u64::from(preview.width) * u64::from(preview.height) <= 64 * 1024 * 1024
            && (1..=120).contains(&preview.fps),
        "Invalid preview dimensions/FPS"
    );
    for target in [&preview.clean, &preview.annotated].into_iter().flatten() {
        ensure!(
            target.starts_with("rtsp://")
                || target.starts_with("rtsps://")
                || Path::new(target).is_absolute(),
            "Invalid preview target"
        );
    }
    Ok(())
}

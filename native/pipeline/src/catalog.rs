use crate::config::{Config, Model, Output, Rules, Stream, TrackerKind};
use anyhow::{Context, Result, ensure};
use futures_util::{StreamExt, stream};
use serde_json::Value;
use sqlx::{
    PgPool, Row,
    postgres::{PgListener, PgPoolOptions},
};
use std::{
    collections::HashMap,
    process::Stdio,
    time::{Duration, Instant},
};

pub struct Catalog {
    pool: PgPool,
    listener: Option<PgListener>,
    dimensions: HashMap<String, (u32, u32)>,
    retry_probe_at: HashMap<String, Instant>,
}
impl Catalog {
    pub async fn new(config: &Config) -> Result<Self> {
        let spec = config
            .database
            .as_ref()
            .context("Database configuration missing")?;
        let url =
            std::env::var(&spec.url_env)?.replacen("postgresql+asyncpg://", "postgresql://", 1);
        let pool = PgPoolOptions::new()
            .max_connections(3)
            .acquire_timeout(Duration::from_secs(5))
            .after_connect(|connection, _| {
                Box::pin(async move {
                    sqlx::query("SET TIME ZONE 'UTC'")
                        .execute(connection)
                        .await?;
                    Ok(())
                })
            })
            .connect(&url)
            .await?;
        let mut result = Self {
            pool,
            listener: None,
            dimensions: HashMap::new(),
            retry_probe_at: HashMap::new(),
        };
        result.reconnect_listener().await;
        Ok(result)
    }
    async fn reconnect_listener(&mut self) {
        if let Ok(mut listener) = PgListener::connect_with(&self.pool).await {
            if listener
                .listen_all(["stream_config_changed", "model_catalog_changed"])
                .await
                .is_ok()
            {
                self.listener = Some(listener)
            }
        }
    }
    pub async fn changed(&mut self, seconds: u32) {
        if self.listener.is_none() {
            self.reconnect_listener().await;
        }
        if let Some(listener) = self.listener.as_mut() {
            tokio::select! {_ = tokio::time::sleep(Duration::from_secs(seconds.max(1) as u64))=>(),
            result=listener.recv()=>{if result.is_err(){self.listener=None}}}
        } else {
            tokio::time::sleep(Duration::from_secs(seconds.max(1) as u64)).await;
        }
    }
    pub async fn load(&mut self, base: &Config) -> Result<Config> {
        let db = base
            .database
            .as_ref()
            .context("Database configuration missing")?;
        let engines: Value = serde_json::from_slice(&std::fs::read(&db.engine_manifest)?)?;
        ensure!(engines["format"] == 1, "Unsupported engine manifest format");
        let rows = sqlx::query(
            "SELECT sc.id::text AS id,sc.site_id::bigint AS site_id,s.name AS site,sc.stream_name,sc.video_url,
          sc.model_key,sc.recognition_enabled,sc.expire_date::timestamptz AS expires,
          sc.work_start_hour::int AS work_start_hour,sc.work_end_hour::int AS work_end_hour,
          sc.detect_no_safety_vest_or_helmet,sc.detect_near_machinery_or_vehicle,
          sc.detect_in_restricted_area,sc.detect_in_utility_pole_restricted_area,
          sc.detect_machinery_close_to_pole,c.definition::text AS model_definition
          FROM stream_configs sc JOIN sites s ON s.id=sc.site_id
          LEFT JOIN detection_model_catalog c ON c.model_key=sc.model_key ORDER BY sc.id",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut config = base.clone();
        config.models.clear();
        config.streams.clear();
        for row in rows {
            if !row.try_get::<bool, _>("recognition_enabled")? {
                continue;
            }
            let expires = row.try_get::<Option<chrono::DateTime<chrono::Utc>>, _>("expires")?;
            if expires.is_some_and(|stamp| stamp <= chrono::Utc::now()) {
                continue;
            }
            let Some(definition) = row.try_get::<Option<String>, _>("model_definition")? else {
                continue;
            };
            let definition: Value = crate::schema::normalize(serde_json::from_str(&definition)?)?;
            if !definition["enabled"].as_bool().unwrap_or(true)
                || !definition["capabilities"]
                    .as_array()
                    .is_some_and(|v| v.iter().any(|x| x == "stream"))
            {
                continue;
            }
            let key: String = row.try_get("model_key")?;
            let engine = &engines["models"][&key];
            ensure!(
                engine["source_sha256"] == definition["sha256"]
                    && engine["version"] == definition["version"],
                "Catalog/engine manifest mismatch"
            );
            if !config.models.iter().any(|m| m.key == key) {
                config.models.push(Model {
                    catalog_revision: Some(crate::schema::revision(&definition)?),
                    class_metadata: definition["classes"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default(),
                    key: key.clone(),
                    artifact: engine["artifact"]
                        .as_str()
                        .context("Missing engine artifact")?
                        .into(),
                    sha256: engine["sha256"]
                        .as_str()
                        .context("Missing engine digest")?
                        .into(),
                    version: definition["version"]
                        .as_str()
                        .context("Missing model version")?
                        .into(),
                    source_artifact: Some(
                        definition["artifact"]
                            .as_str()
                            .context("Missing published weights")?
                            .into(),
                    ),
                    source_sha256: Some(
                        definition["sha256"]
                            .as_str()
                            .context("Missing weights digest")?
                            .into(),
                    ),
                    imgsz: 640,
                    replay: None,
                    labels: Default::default(),
                });
            }
            let id: String = row.try_get("id")?;
            let source: String = row.try_get("video_url")?;
            let template = base.streams.iter().find(|stream| stream.id == id);
            let (width, height) = if let (Some(w), Some(h)) = (db.width, db.height) {
                (w, h)
            } else if let Some(template) = template.filter(|stream| stream.source == source) {
                (template.width, template.height)
            } else {
                self.dimensions.get(&source).copied().unwrap_or_default()
            };
            let rules = Rules {
                detect_no_safety_vest_or_helmet: row.try_get("detect_no_safety_vest_or_helmet")?,
                detect_near_machinery_or_vehicle: row
                    .try_get("detect_near_machinery_or_vehicle")?,
                detect_in_restricted_area: row.try_get("detect_in_restricted_area")?,
                detect_in_utility_pole_restricted_area: row
                    .try_get("detect_in_utility_pole_restricted_area")?,
                detect_machinery_close_to_pole: row.try_get("detect_machinery_close_to_pole")?,
            };
            let site: String = row.try_get("site")?;
            let name: String = row.try_get("stream_name")?;
            let media_output = db
                .media
                .as_ref()
                .map(|media| catalog_output(media, &site, &name));
            config.streams.push(Stream {
                id,
                site_id: row.try_get("site_id")?,
                site: row.try_get("site")?,
                name: row.try_get("stream_name")?,
                source,
                model: key,
                width,
                height,
                fps: db.fps,
                enabled: true,
                expires,
                work_start_hour: row
                    .try_get::<Option<i32>, _>("work_start_hour")?
                    .unwrap_or(7) as u32,
                work_end_hour: row
                    .try_get::<Option<i32>, _>("work_end_hour")?
                    .unwrap_or(18) as u32,
                timezone: "Asia/Taipei".into(),
                languages: template
                    .map(|s| s.languages.clone())
                    .or_else(|| db.media.as_ref().map(|media| media.languages.clone()))
                    .unwrap_or_else(|| vec!["en".into()]),
                rules,
                tracker: template.map(|s| s.tracker).unwrap_or(TrackerKind::Centroid),
                output: template
                    .map(|s| s.output.clone())
                    .filter(|output| {
                        output.clean.is_some()
                            || output.annotated.is_some()
                            || output.preview.is_some()
                    })
                    .or(media_output)
                    .unwrap_or_else(Output::default),
                evidence_cooldown_seconds: 300,
            });
        }
        let sources: std::collections::HashSet<_> = config
            .streams
            .iter()
            .map(|stream| stream.source.as_str())
            .collect();
        self.dimensions
            .retain(|source, _| sources.contains(source.as_str()));
        self.retry_probe_at
            .retain(|source, _| sources.contains(source.as_str()));
        let now = Instant::now();
        let mut pending = Vec::new();
        for camera in &config.streams {
            if pending.len() == 4 {
                break;
            }
            if camera.width == 0
                && self
                    .retry_probe_at
                    .get(&camera.source)
                    .is_none_or(|at| *at <= now)
                && !pending.iter().any(|(_, source)| source == &camera.source)
            {
                pending.push((camera.id.clone(), camera.source.clone()));
            }
        }
        let mut probes = stream::iter(pending)
            .map(|(id, source)| async move {
                let dimensions = probe(&source).await;
                (id, source, dimensions)
            })
            .buffer_unordered(4);
        while let Some((id, source, result)) = probes.next().await {
            match result {
                Ok(dimensions) => {
                    self.retry_probe_at.remove(&source);
                    self.dimensions.insert(source, dimensions);
                }
                Err(_) => {
                    self.retry_probe_at
                        .insert(source, Instant::now() + Duration::from_secs(30));
                    tracing::warn!(camera=%id, "Source dimensions unavailable; will retry");
                }
            }
        }
        config.streams.retain_mut(|camera| {
            if camera.width > 0 {
                return true;
            }
            if let Some(&(width, height)) = self.dimensions.get(&camera.source) {
                camera.width = width;
                camera.height = height;
                return true;
            }
            false
        });
        config.validate()?;
        Ok(config)
    }
}
pub async fn probe(source: &str) -> Result<(u32, u32)> {
    let mut command = tokio::process::Command::new("ffprobe");
    command
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stderr(Stdio::null());
    if source.starts_with("rtsp://") || source.starts_with("rtsps://") {
        command.args(["-rtsp_transport", "tcp", "-timeout", "5000000"]);
    }
    command.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        "-i",
        source,
    ]);
    let result = tokio::time::timeout(Duration::from_secs(8), command.output()).await??;
    ensure!(result.status.success(), "Source probe failed");
    let data: Value = serde_json::from_slice(&result.stdout)?;
    let width = data["streams"][0]["width"]
        .as_u64()
        .context("Missing video width")? as u32;
    let height = data["streams"][0]["height"]
        .as_u64()
        .context("Missing video height")? as u32;
    ensure!(width > 0 && height > 0, "Invalid source dimensions");
    Ok((width, height))
}

fn catalog_output(media: &crate::config::MediaDefaults, site: &str, name: &str) -> Output {
    use base64::Engine;
    let encode = |value: &str| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value);
    let base = format!(
        "{}/hazard_{}_{}",
        media.publish_base.trim_end_matches('/'),
        encode(site),
        encode(name)
    );
    Output {
        clean: Some(base.clone()),
        annotated: Some(format!("{base}_annotated_{{language_b64}}")),
        nvenc: media.nvenc,
        on_demand: media.on_demand,
        clean_passthrough: media.clean_passthrough,
        client_overlay: false,
        preview: media
            .preview
            .as_ref()
            .map(|shape| crate::config::PreviewOutput {
                clean: Some(format!("{base}_preview")),
                annotated: Some(format!("{base}_preview_annotated_{{language_b64}}")),
                width: shape.width,
                height: shape.height,
                fps: shape.fps,
            }),
    }
}

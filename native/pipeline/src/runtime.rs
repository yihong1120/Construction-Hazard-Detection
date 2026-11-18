use crate::{
    capture,
    config::{Config, Stream},
    delivery,
    frame::SharedFrame,
    hazard::{HazardEngine, HazardResult, Track},
    inference::{Client, Workers},
    metadata,
    outbox::{Event, Outbox},
    publish,
    render::{self, Renderer, Text},
    tracking::Tracker,
};
use anyhow::{Context, Result, ensure};
use chrono::{Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

#[derive(Default, Serialize, Deserialize)]
pub struct CameraReport {
    pub id: String,
    pub captured: u64,
    pub capture_read_ms: f64,
    pub capture_convert_ms: f64,
    pub analysis_queue_ms: f64,
    pub analysis_cpu_ms: f64,
    pub encoded_unique: u64,
    pub encoded_repeated: u64,
    pub encoded_offline: u64,
    #[serde(default)]
    pub encoder_conversion_reused: u64,
    pub encode_ms: f64,
    pub converted: u64,
    pub capture_dropped: u64,
    #[serde(default)]
    pub capture_rate_dropped: u64,
    #[serde(default)]
    pub gpu_admitted: u64,
    #[serde(default)]
    pub host_retained: u64,
    #[serde(default)]
    pub host_materialized: u64,
    #[serde(default)]
    pub host_sampled: u64,
    #[serde(default)]
    pub gpu_materialized: u64,
    #[serde(default)]
    pub gpu_download_payload_bytes: u64,
    #[serde(default)]
    pub rust_rtsp_frames: u64,
    #[serde(default)]
    pub rust_rtsp_bytes: u64,
    #[serde(default)]
    pub rust_rtsp_lost_packets: u64,
    #[serde(default)]
    pub rust_rtsp_keyframe_dropped: u64,
    #[serde(default)]
    pub rust_rtsp_handoff_ms: f64,
    #[serde(default)]
    pub rust_rtsp_decoder_resets: u64,
    pub capture_reconnects: u64,
    pub capture_pool_peak: u64,
    pub rendered: u64,
    pub render_skipped: u64,
    pub render_pending: u64,
    #[serde(default)]
    pub render_passthrough: u64,
    pub processed: u64,
    pub skipped: u64,
    pub stale: u64,
    pub stored_events: u64,
    pub encoded: u64,
    #[serde(default)]
    pub relayed_packets: u64,
    #[serde(default)]
    pub relay_dropped: u64,
    #[serde(default)]
    pub relay_errors: u64,
    #[serde(default)]
    pub relay_fallback: u64,
    pub inference_ms: f64,
    pub total_ms: f64,
    pub failures: u64,
    #[serde(default)]
    pub metadata_offered: u64,
    #[serde(default)]
    pub metadata_sent: u64,
    #[serde(default)]
    pub metadata_superseded: u64,
    #[serde(default)]
    pub metadata_errors: u64,
    #[serde(default)]
    pub fonts_initialized: bool,
}
#[derive(Serialize)]
pub struct CpuWorkerReport {
    pub automatic: bool,
    pub initial: usize,
    pub current: usize,
    pub limit: usize,
    pub adjustments: u64,
}

#[derive(Serialize)]
pub struct Report {
    pub cameras: Vec<CameraReport>,
    pub cpu_workers: CpuWorkerReport,
    pub outbox: serde_json::Value,
    pub elapsed_seconds: f64,
    pub python_runtime: bool,
    pub replay: bool,
}

pub fn working(spec: &Stream, timestamp: i64) -> bool {
    let Some(stamp) = chrono::DateTime::from_timestamp_millis(timestamp) else {
        return false;
    };
    let Ok(zone) = spec.timezone.parse::<chrono_tz::Tz>() else {
        return false;
    };
    let hour = stamp.with_timezone(&zone).hour();
    // The PostgreSQL fields describe a same-day window, as in the Python
    // worker. Equal or inverted bounds have no working hours.
    spec.work_start_hour <= hour && hour < spec.work_end_hour
}

fn event_warning_allowed(key: &str, is_working: bool) -> bool {
    is_working
        || matches!(
            key,
            "warning_people_in_controlled_area" | "warning_people_in_utility_pole_controlled_area"
        )
}

#[cfg(test)]
#[test]
fn off_hours_warning_allowlist_contains_only_people_entering_zones() {
    for (key, allowed_after_hours) in [
        ("warning_people_in_controlled_area", true),
        ("warning_people_in_utility_pole_controlled_area", true),
        ("warning_no_safety_vest", false),
        ("warning_close_to_vehicle", false),
        ("detect_machinery_close_to_pole", false),
        ("future_warning", false),
    ] {
        assert!(event_warning_allowed(key, true), "{key}");
        assert_eq!(
            event_warning_allowed(key, false),
            allowed_after_hours,
            "{key}"
        );
    }
}
fn output(template: &str, spec: &Stream, language: &str) -> String {
    crate::config::output_path(template, &spec.id, language)
}
async fn join_publishers(publishers: Vec<publish::Publisher>) -> Result<u64> {
    let mut count = 0;
    let mut failure = None;
    for publisher in publishers {
        drop(publisher.sender);
        match publisher.task.await {
            Ok(Ok(())) => (),
            Ok(Err(error)) => {
                failure.get_or_insert(error);
            }
            Err(error) => {
                failure.get_or_insert(error.into());
            }
        }
        count += publisher.encoded.load(Ordering::Relaxed);
    }
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(count)
}
async fn camera(
    config: Arc<Config>,
    spec: Stream,
    client: Client,
    text: Arc<Text>,
    store: Outbox,
    cpu: crate::executor::Cpu,
    stop: CancellationToken,
) -> Result<CameraReport> {
    let _cancel_children = stop.clone().drop_guard();
    let started = Instant::now();
    let instance = uuid::Uuid::new_v4();
    let mut report = CameraReport {
        id: spec.id.clone(),
        ..Default::default()
    };
    let path = config
        .state_dir
        .join(format!("detections-{}.jsonl", spec.id));
    let mut log = if config.record_detections {
        Some(tokio::io::BufWriter::new(
            tokio::fs::File::create(&path).await?,
        ))
    } else {
        None
    };
    let mut publishers = vec![];
    let mut targets = vec![];
    let mut leases = vec![];
    let mut ready = vec![];
    let clean_gate = crate::demand::Gate::new(!spec.output.on_demand);
    let relay_stop = stop.child_token();
    let demand_stop = stop.child_token();
    let base_path = spec
        .output
        .clean
        .as_ref()
        .and_then(|t| reqwest::Url::parse(&output(t, &spec, "clean")).ok())
        .map(|u| u.path().trim_start_matches('/').to_owned())
        .unwrap_or_default();
    if spec.output.on_demand {
        leases.push((
            format!("media_clean_demand:{base_path}"),
            clean_gate.clone(),
        ));
    }
    if !spec.output.client_overlay
        || spec
            .output
            .annotated
            .as_ref()
            .is_some_and(|t| !t.starts_with("rtsp://") && !t.starts_with("rtsps://"))
    {
        for language in &spec.languages {
            let language = crate::config::output_language(language, spec.output.on_demand);
            if let Some(target) = &spec.output.annotated {
                let live = target.starts_with("rtsp://") || target.starts_with("rtsps://");
                let gate = crate::demand::Gate::new(!spec.output.on_demand || !live);
                if spec.output.on_demand && live {
                    use base64::Engine;
                    leases.push((
                        format!(
                            "media_overlay_demand:{base_path}:{}",
                            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(language)
                        ),
                        gate.clone(),
                    ));
                }
                let publisher = publish::spawn_gated_localized(
                    config.clone(),
                    output(target, &spec, language),
                    spec.width,
                    spec.height,
                    spec.fps,
                    spec.output.nvenc,
                    text.clone(),
                    crate::locale::canonical(language).unwrap_or("en"),
                    gate,
                    stop.clone(),
                );
                if spec.output.on_demand && live {
                    let path = reqwest::Url::parse(&output(target, &spec, language))?
                        .path()
                        .trim_start_matches('/')
                        .to_owned();
                    ready.push((
                        format!("media_overlay_ready:{path}"),
                        publisher.stats.ready.clone(),
                    ));
                }
                targets.push(crate::overlay::Target {
                    language: language.to_owned(),
                    sender: publisher.sender.clone(),
                    fps: spec.fps,
                    size: None,
                });
                publishers.push(publisher);
            }
        }
    }
    let relay = if spec.output.clean_passthrough {
        Some(crate::relay::spawn(
            output(spec.output.clean.as_ref().unwrap(), &spec, "clean"),
            spec.width,
            spec.height,
            spec.fps,
            clean_gate.clone(),
            relay_stop.clone(),
        ))
    } else {
        None
    };
    let relay_stats = relay.as_ref().map(|r| r.sender.stats.clone());
    let clean = if let Some(target) = &spec.output.clean {
        let gate = if let Some(relay) = &relay {
            let mode = relay.sender.mode.clone();
            crate::demand::Gate::filtered(clean_gate.clone(), move || {
                mode.load(Ordering::Acquire) == 2
            })
        } else {
            clean_gate
        };
        let publisher = publish::spawn_gated(
            config.clone(),
            output(target, &spec, "clean"),
            spec.width,
            spec.height,
            spec.fps,
            spec.output.nvenc,
            text.clone(),
            gate,
            stop.clone(),
        );
        let sender = publisher.sender.clone();
        publishers.push(publisher);
        Some(sender)
    } else {
        None
    };
    let mut clean_senders = clean.into_iter().collect::<Vec<_>>();
    if let Some(preview) = &spec.output.preview {
        let preview_base = preview
            .clean
            .as_ref()
            .and_then(|target| reqwest::Url::parse(&output(target, &spec, "clean")).ok())
            .map(|url| url.path().trim_start_matches('/').to_owned())
            .unwrap_or_else(|| format!("{base_path}_preview"));
        if let Some(target) = &preview.clean {
            let live = target.starts_with("rtsp://") || target.starts_with("rtsps://");
            let gate = crate::demand::Gate::new(!spec.output.on_demand || !live);
            if spec.output.on_demand && live {
                leases.push((format!("media_clean_demand:{preview_base}"), gate.clone()));
            }
            let publisher = publish::spawn_gated(
                config.clone(),
                output(target, &spec, "clean"),
                preview.width,
                preview.height,
                preview.fps,
                spec.output.nvenc,
                text.clone(),
                gate,
                stop.clone(),
            );
            if spec.output.on_demand && live {
                ready.push((
                    format!("media_clean_ready:{preview_base}"),
                    publisher.stats.ready.clone(),
                ));
            }
            clean_senders.push(publisher.sender.clone());
            publishers.push(publisher);
        }
        if let Some(target) = &preview.annotated {
            let live = target.starts_with("rtsp://") || target.starts_with("rtsps://");
            if !spec.output.client_overlay || !live {
                for language in &spec.languages {
                    use base64::Engine;
                    let language = crate::config::output_language(language, spec.output.on_demand);
                    let gate = crate::demand::Gate::new(!spec.output.on_demand || !live);
                    if spec.output.on_demand && live {
                        leases.push((
                            format!(
                                "media_overlay_demand:{preview_base}:{}",
                                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(language)
                            ),
                            gate.clone(),
                        ));
                    }
                    let target = output(target, &spec, language);
                    let publisher = publish::spawn_gated_localized(
                        config.clone(),
                        target.clone(),
                        preview.width,
                        preview.height,
                        preview.fps,
                        spec.output.nvenc,
                        text.clone(),
                        crate::locale::canonical(language).unwrap_or("en"),
                        gate,
                        stop.clone(),
                    );
                    if spec.output.on_demand && live {
                        let url = reqwest::Url::parse(&target)?;
                        ready.push((
                            format!("media_overlay_ready:{}", url.path().trim_start_matches('/')),
                            publisher.stats.ready.clone(),
                        ));
                    }
                    targets.push(crate::overlay::Target {
                        language: language.to_owned(),
                        sender: publisher.sender.clone(),
                        fps: preview.fps,
                        size: Some((preview.width, preview.height)),
                    });
                    publishers.push(publisher);
                }
            }
        }
    }
    let clean = match clean_senders.len() {
        0 => None,
        1 => clean_senders.pop(),
        _ => Some(publish::Sender::Group(clean_senders)),
    };
    let demand_task = if leases.is_empty() {
        None
    } else {
        Some(crate::demand::spawn(
            config.redis.as_ref().unwrap().clone(),
            leases,
            ready,
            demand_stop.clone(),
        ))
    };
    let publisher_stats = publishers
        .iter()
        .map(|p| p.stats.clone())
        .collect::<Vec<_>>();
    let capture = capture::spawn(
        spec.clone(),
        config.lossless,
        config.max_frames,
        config.capture.clone(),
        clean,
        relay.as_ref().map(|r| r.sender.clone()),
        stop.clone(),
    );
    let (mut frames, capture_task, produced, capture_stats) = (
        capture.frames,
        capture.task,
        capture.produced,
        capture.stats,
    );
    let actor = if !config.lossless && !targets.is_empty() {
        Some(crate::overlay::spawn(
            client.model.clone(),
            frames.latest().expect("live mailbox"),
            targets.clone(),
            text.clone(),
            cpu.clone(),
            config.overlay_max_age_ms,
            stop.clone(),
        ))
    } else {
        None
    };
    let mut lossless_renderers = if config.lossless {
        targets
            .iter()
            .map(|target| {
                (
                    target.language.clone(),
                    Renderer::new(spec.width, spec.height, text.clone()),
                )
            })
            .collect::<Vec<_>>()
    } else {
        vec![]
    };
    let metadata_actor = if config.metadata_async {
        config
            .redis
            .as_ref()
            .map(|redis| metadata::Actor::new(redis.clone(), spec.clone(), client.model.clone()))
    } else {
        None
    };
    let mut sink = if let Some(redis) = config.redis.as_ref().filter(|_| !config.metadata_async) {
        metadata::Sink::new(redis, &spec).await.ok()
    } else {
        None
    };
    let mut analysis = Some((Tracker::new(spec.tracker), HazardEngine::default()));
    let (mut seen, mut generation, mut last_event) = (0, 0, 0);
    let (mut logged_bytes, mut log_flushed) = (0usize, Instant::now());
    let mut result: Result<()> = async {
        loop {
            let initial = tokio::select! {
                frame = frames.next(seen, &stop) => frame,
                _ = async {
                    if let Some(expires) = spec.expires {
                        tokio::time::sleep((expires - Utc::now()).to_std().unwrap_or(Duration::ZERO)).await;
                    } else {
                        std::future::pending::<()>().await
                    }
                } => {
                    stop.cancel();
                    None
                }
            };
            let Some(initial) = initial else { break };
            let source = if let Some(feed) = frames.latest() {
                let minimum = initial.sequence;
                drop(initial);
                crate::inference::Input::Latest { feed, minimum }
            } else {
                crate::inference::Input::Fixed(initial)
            };
            let infer = Instant::now();
            let prediction = tokio::select! {
                _ = stop.cancelled() => break,
                result = client.predict(source) => result
            };
            report.inference_ms += infer.elapsed().as_secs_f64() * 1000.;
            let prediction = match prediction {
                Ok(value) => value,
                Err(_) => {
                    report.failures += 1;
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }
            };
            let frame = prediction.frame;
            if frames.generation().is_some_and(|g| g != frame.generation) {
                report.stale += 1;
                continue;
            }
            report.skipped += frame.sequence.saturating_sub(seen + 1);
            seen = frame.sequence;
            if frame.generation != generation {
                generation = frame.generation;
                analysis = Some((Tracker::new(spec.tracker), HazardEngine::default()));
            }

            let mut state = analysis.take().expect("analysis owner");
            let rules = spec.rules.clone();
            let detections = prediction.detections;
            let analysis_queued = Instant::now();
            let (state_out, tracks, detected, analysis_queue_ms, analysis_cpu_ms) = cpu
                .run(move || -> Result<_> {
                    let started = Instant::now();
                    let queued = started.duration_since(analysis_queued).as_secs_f64() * 1000.;
                    let tracks = state.0.update(&detections);
                    let detected = state.1.detect(&tracks, &rules)?;
                    Ok((state, tracks, detected, queued, started.elapsed().as_secs_f64() * 1000.))
                })
                .await??;
            report.analysis_queue_ms += analysis_queue_ms;
            report.analysis_cpu_ms += analysis_cpu_ms;
            analysis = Some(state_out);
            if frames.generation().is_some_and(|g| g != generation) {
                report.stale += 1;
                continue;
            }

            let tracks = Arc::new(tracks);
            let detected = Arc::new(detected);
            let is_working = working(&spec, frame.timestamp_ms);
            if let Some(actor) = &actor {
                let snapshot = crate::overlay::Snapshot {
                    sequence: frame.sequence,
                    generation,
                    captured: frame.captured,
                    tracks: tracks.clone(),
                    hazard: detected.clone(),
                };
                ensure!(
                    actor.updates.send(Some(Arc::new(snapshot))).is_ok() || stop.is_cancelled(),
                    "Annotation actor unavailable"
                );
            }

            if let Some(writer) = &mut log {
                let payload = serde_json::json!({
                    "sequence": frame.sequence,
                    "generation": generation,
                    "timestamp_ms": frame.timestamp_ms,
                    "completed_timestamp_ms": Utc::now().timestamp_millis(),
                    "frame_age_ms": frame.captured.elapsed().as_secs_f64() * 1000.,
                    "source_pts": frame.source_pts,
                    "source_lag_ms": frame.source_lag_ms,
                    "batch_size": prediction.batch_size,
                    "inference_timing": prediction.timing,
                    "analysis_queue_ms": analysis_queue_ms,
                    "analysis_cpu_ms": analysis_cpu_ms,
                    "model_id": client.model.key,
                    "model_version": client.model.version,
                    "model_sha256": client.model.sha256,
                    "replay": client.model.replay.is_some(),
                    "tracks": tracks.as_ref(),
                    "warnings": detected.warnings,
                    "cone_polys": detected.cone_polys,
                    "pole_polys": detected.pole_polys
                });
                let bytes = serde_json::to_vec(&payload)?;
                if logged_bytes + bytes.len() > 10 * 1024 * 1024 {
                    writer.flush().await?;
                    tokio::fs::rename(&path, path.with_extension("jsonl.1")).await?;
                    *writer = tokio::io::BufWriter::new(tokio::fs::File::create(&path).await?);
                    logged_bytes = 0;
                }
                writer.write_all(&bytes).await?;
                writer.write_all(b"\n").await?;
                logged_bytes += bytes.len() + 1;
                if log_flushed.elapsed() >= Duration::from_secs(1) {
                    writer.flush().await?;
                    log_flushed = Instant::now();
                }
            }

            if let Some(actor) = &metadata_actor {
                actor.offer(metadata::Snapshot {
                    sequence: frame.sequence,
                    generation,
                    timestamp: frame.timestamp_ms,
                    tracks: tracks.clone(),
                    hazard: detected.clone(),
                    working: is_working,
                    source_pts: frame.source_pts,
                });
            } else {
                if sink.is_none() {
                    if let Some(redis) = &config.redis {
                        sink = metadata::Sink::new(redis, &spec).await.ok();
                    }
                }
                if let Some(metadata) = &mut sink {
                    if metadata
                        .publish_source(
                            &spec,
                            frame.sequence,
                            generation,
                            frame.timestamp_ms,
                            &tracks,
                            &detected,
                            is_working,
                            &client.model,
                            frame.source_pts,
                        )
                        .await
                        .is_err()
                    {
                        report.failures += 1;
                    }
                }
            }

            if frame.timestamp_ms / 1000 - last_event >= spec.evidence_cooldown_seconds as i64
                && let Some(event) = event(&spec, &client, &frame, &tracks, &detected, instance, is_working)
            {
                let pixels = frame.clone();
                let persisted = match cpu.run(move || render::jpeg(&pixels)).await? {
                    Ok(jpeg) => store.put(&event, &jpeg).await,
                    Err(error) => Err(error),
                };
                match persisted {
                    Ok(true) => {
                        report.stored_events += 1;
                        last_event = frame.timestamp_ms / 1000;
                    }
                    Ok(false) => (),
                    Err(error) => {
                        report.failures += 1;
                        tracing::error!(%error, camera = %spec.id, "Evidence persistence failed");
                    }
                }
            }

            if config.lossless && !targets.is_empty() {
                let mut renderers = std::mem::take(&mut lossless_renderers);
                let pixels = frame.clone();
                let model = client.model.clone();
                let (renderers_out, images) = cpu
                    .run(move || {
                        let images = renderers
                            .iter_mut()
                            .map(|(lang, renderer)| renderer.render(&pixels, &tracks, &detected, &model, lang))
                            .collect::<Result<Vec<_>>>();
                        (renderers, images)
                    })
                    .await?;
                lossless_renderers = renderers_out;
                let images = images?;
                for (target, image) in targets.iter().zip(images) {
                    ensure!(
                        target.sender.send(image, &stop).await || stop.is_cancelled(),
                        "Encoder unavailable"
                    );
                }
            }
            report.processed += 1;
        }
        Ok(())
    }
    .await;
    if result.is_err() {
        stop.cancel();
    }
    drop(frames);
    let captured = capture_task
        .await
        .map_err(anyhow::Error::from)
        .and_then(|value| value);
    if let Err(error) = captured {
        stop.cancel();
        if result.is_ok() {
            result = Err(error);
        }
    }
    relay_stop.cancel();
    demand_stop.cancel();
    if let Some(task) = demand_task {
        let _ = task.await;
    }
    if let Some(relay) = relay {
        let _ = relay.task.await;
    }
    report.captured = produced.load(Ordering::Relaxed);
    report.capture_read_ms = capture_stats.read_ns.load(Ordering::Relaxed) as f64 / 1e6;
    report.capture_convert_ms = capture_stats.convert_ns.load(Ordering::Relaxed) as f64 / 1e6;
    report.converted = capture_stats.converted.load(Ordering::Relaxed);
    report.capture_dropped = capture_stats.admission_dropped.load(Ordering::Relaxed);
    report.capture_rate_dropped = capture_stats.rate_dropped.load(Ordering::Relaxed);
    report.capture_reconnects = capture_stats.reconnects.load(Ordering::Relaxed);
    report.capture_pool_peak = capture_stats.pool_peak.load(Ordering::Relaxed);
    if let Some(actor) = actor {
        drop(actor.updates);
        if let Err(error) = actor
            .task
            .await
            .map_err(anyhow::Error::from)
            .and_then(|value| value)
        {
            stop.cancel();
            if result.is_ok() {
                result = Err(error);
            }
        }
        report.rendered = actor.stats.rendered.load(Ordering::Relaxed);
        report.render_skipped = actor.stats.skipped.load(Ordering::Relaxed);
        report.render_pending = actor.stats.pending.load(Ordering::Relaxed);
        report.render_passthrough = actor.stats.passthrough.load(Ordering::Relaxed);
    }
    drop(targets);
    drop(lossless_renderers);
    match join_publishers(publishers).await {
        Ok(count) => report.encoded = count,
        Err(error) => {
            if result.is_ok() {
                result = Err(error);
            }
        }
    }
    for stats in publisher_stats {
        report.encoded_unique += stats.unique.load(Ordering::Relaxed);
        report.encoded_repeated += stats.repeats.load(Ordering::Relaxed);
        report.encoded_offline += stats.offline.load(Ordering::Relaxed);
        report.encoder_conversion_reused += stats.conversion_reused.load(Ordering::Relaxed);
        report.encode_ms += stats.encode_ns.load(Ordering::Relaxed) as f64 / 1e6;
    }
    if let Some(stats) = relay_stats {
        report.relayed_packets = stats.packets.load(Ordering::Relaxed);
        report.relay_dropped = stats.dropped.load(Ordering::Relaxed);
        report.relay_errors = stats.errors.load(Ordering::Relaxed);
        report.relay_fallback = stats.fallback.load(Ordering::Relaxed);
    }
    report.fonts_initialized = text.initialized();
    if let Some(actor) = metadata_actor {
        let stats = actor.stats.clone();
        actor.finish().await;
        report.metadata_offered = stats.offered.load(Ordering::Relaxed);
        report.metadata_sent = stats.sent.load(Ordering::Relaxed);
        report.metadata_superseded = stats.superseded.load(Ordering::Relaxed);
        report.metadata_errors = stats.errors.load(Ordering::Relaxed);
    }
    if let Some(sink) = sink {
        let _ = sink.close().await;
    }
    if let Some(log) = &mut log {
        log.flush().await?;
    }
    report.host_retained = capture_stats.gpu.host_retained.load(Ordering::Relaxed);
    report.host_materialized = capture_stats.gpu.host_materialized.load(Ordering::Relaxed);
    report.host_sampled = capture_stats.gpu.host_sampled.load(Ordering::Relaxed);
    report.gpu_admitted = capture_stats.gpu_admitted.load(Ordering::Relaxed);
    report.gpu_materialized = capture_stats.gpu.materialized.load(Ordering::Relaxed);
    report.gpu_download_payload_bytes = capture_stats
        .gpu
        .download_payload_bytes
        .load(Ordering::Relaxed);
    report.rust_rtsp_frames = capture_stats.ingress.frames.load(Ordering::Relaxed);
    report.rust_rtsp_bytes = capture_stats.ingress.bytes.load(Ordering::Relaxed);
    report.rust_rtsp_lost_packets = capture_stats.ingress.lost_packets.load(Ordering::Relaxed);
    report.rust_rtsp_keyframe_dropped = capture_stats
        .ingress
        .keyframe_dropped
        .load(Ordering::Relaxed);
    report.rust_rtsp_handoff_ms =
        capture_stats.ingress.handoff_ns.load(Ordering::Relaxed) as f64 / 1e6;
    report.rust_rtsp_decoder_resets = capture_stats.ingress.decoder_resets.load(Ordering::Relaxed);
    report.total_ms = started.elapsed().as_secs_f64() * 1000.;
    result?;
    Ok(report)
}
fn event(
    spec: &Stream,
    client: &Client,
    frame: &SharedFrame,
    tracks: &[Track],
    hazard: &HazardResult,
    instance: uuid::Uuid,
    is_working: bool,
) -> Option<Event> {
    let warnings = hazard
        .warnings
        .iter()
        .filter(|(key, _)| event_warning_allowed(key, is_working))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<crate::hazard::Warnings>();
    if warnings.is_empty() {
        return None;
    }
    let key = format!(
        "{}:{}:{}:{}:{}:{}",
        spec.site_id, spec.id, instance, frame.generation, frame.sequence, client.model.sha256
    );
    let detection_time = chrono::DateTime::from_timestamp_millis(frame.timestamp_ms)?.to_rfc3339();
    Some(Event {
        event_id: uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, key.as_bytes()).to_string(),
        site_id: spec.site_id,
        site: spec.site.clone(),
        stream_name: spec.name.clone(),
        detection_time,
        model_id: client.model.key.clone(),
        model_version: client.model.version.clone(),
        warnings,
        detections: tracks.to_vec(),
        cone_polygon: hazard.cone_polys.clone(),
        pole_polygon: hazard.pole_polys.clone(),
    })
}
pub async fn run(mut config: Config, stop: CancellationToken) -> Result<Report> {
    let started = Instant::now();
    std::fs::create_dir_all(&config.state_dir)?;
    let owner = crate::lease::acquire(&config.state_dir, "pipeline", 1, 1)?
        .context("Another pipeline owns this state directory")?;
    let base = config.clone();
    let mut catalog = if config.database.is_some() {
        Some(crate::catalog::Catalog::new(&config).await?)
    } else {
        None
    };
    if let Some(catalog) = &mut catalog {
        config = catalog.load(&base).await?;
    }
    let now = Utc::now();
    let active_cameras = config
        .streams
        .iter()
        .filter(|stream| stream.enabled_at(now))
        .count();
    let auto = config.cpu_workers_auto;
    let (cpu, initial, limit) = if let Some(policy) = auto {
        let available = std::thread::available_parallelism().map_or(1, |count| count.get());
        let limit = policy.limit(config.cpu_workers, available)?;
        let initial = policy.workers(active_cameras, limit);
        (
            crate::executor::Cpu::elastic(initial, limit)?,
            initial,
            limit,
        )
    } else {
        let fixed = config.cpu_workers.clamp(1, 64);
        (crate::executor::Cpu::new(fixed)?, fixed, fixed)
    };
    let mut cpu_workers = CpuWorkerReport {
        automatic: auto.is_some(),
        initial,
        current: initial,
        limit,
        adjustments: 0,
    };
    tracing::info!(
        automatic = cpu_workers.automatic,
        workers = initial,
        worker_limit = limit,
        active_cameras,
        "CPU worker budget configured"
    );
    let store = Outbox::open(&config).await?;
    let imported = store
        .migrate_sqlite(&config.state_dir.join("outbox.sqlite3"))
        .await?;
    if imported > 0 {
        tracing::info!(imported, "Imported legacy SQLite events into PostgreSQL");
    }
    let dispatcher_stop = CancellationToken::new();
    let dispatcher = if config.delivery.enabled {
        Some(tokio::spawn(delivery::dispatch(
            store.clone(),
            config.delivery.clone(),
            dispatcher_stop.clone(),
        )))
    } else {
        None
    };
    let text = Text::new(&config.fonts)?;
    let mut workers = Some(Workers::new(&config)?);
    let mut active = HashMap::<String, (Vec<u8>, CancellationToken)>::new();
    let mut completed = HashSet::new();
    let mut tasks = tokio::task::JoinSet::new();
    let mut reports = vec![];
    let mut fatal = None;
    loop {
        let now = Utc::now();
        let desired: HashMap<_, _> = config
            .streams
            .iter()
            .filter(|s| !completed.contains(&s.id))
            .filter(|s| s.enabled_at(now))
            .map(|s| (s.id.clone(), s.clone()))
            .collect();
        if let Some(policy) = auto {
            let target = policy.workers(desired.len(), limit);
            if target != cpu_workers.current {
                cpu.set_workers(target)?;
                tracing::info!(
                    previous_workers = cpu_workers.current,
                    workers = target,
                    active_cameras = desired.len(),
                    "CPU worker budget adjusted"
                );
                cpu_workers.current = target;
                cpu_workers.adjustments += 1;
            }
        }
        for (id, (fingerprint, token)) in &active {
            if !desired.contains_key(id) || serde_json::to_vec(&desired[id])? != *fingerprint {
                token.cancel();
            }
        }
        for (id, spec) in &desired {
            if !active.contains_key(id) {
                let fingerprint = serde_json::to_vec(spec)?;
                let token = stop.child_token();
                let spec = spec.clone();
                let client = workers.as_ref().expect("workers").get(&spec.model)?;
                tasks.spawn(camera(
                    Arc::new(config.clone()),
                    spec,
                    client,
                    text.clone(),
                    store.clone(),
                    cpu.clone(),
                    token.clone(),
                ));
                active.insert(id.clone(), (fingerprint, token));
            }
        }
        if tasks.is_empty() && catalog.is_none() {
            break;
        }
        tokio::select! {
            _ = stop.cancelled() => break,
            result = tasks.join_next(), if !tasks.is_empty() => {
                match result {
                    Some(Ok(Ok(report))) => {
                        active.remove(&report.id);
                        if catalog.is_none() {
                            completed.insert(report.id.clone());
                        }
                        reports.push(report);
                    }
                    Some(Ok(Err(error))) => {
                        fatal = Some(error);
                        break;
                    }
                    Some(Err(_)) => {
                        fatal = Some(anyhow::anyhow!("Camera task panicked"));
                        break;
                    }
                    None => (),
                }
                if catalog.is_none() && tasks.is_empty() {
                    break;
                }
            },
            _ = async {
                if let Some(catalog) = &mut catalog {
                    catalog
                        .changed(base.database.as_ref().expect("database").refresh_seconds)
                        .await;
                } else {
                    std::future::pending::<()>().await
                }
            } => {
                if let Some(catalog) = &mut catalog {
                    match catalog.load(&base).await {
                        Ok(next) => {
                            // Rebuild only after old camera owners and model requests are drained.
                            if serde_json::to_vec(&next.models)? != serde_json::to_vec(&config.models)? {
                                for (_, token) in active.values() {
                                    token.cancel();
                                }
                                while let Some(result) = tasks.join_next().await {
                                    if let Ok(Ok(report)) = result {
                                        reports.push(report);
                                    }
                                }
                                active.clear();
                                if let Some(old) = workers.take() {
                                    old.close();
                                }
                                workers = Some(Workers::new(&next)?);
                            }
                            config = next;
                        }
                        Err(_) => tracing::warn!(
                            "Configuration refresh failed; retaining last verified generation"
                        ),
                    }
                }
            }
        }
    }
    for (_, token) in active.values() {
        token.cancel();
    }
    while let Some(result) = tasks.join_next().await {
        if let Ok(Ok(report)) = result {
            reports.push(report)
        }
    }
    active.clear();
    if let Some(workers) = workers.take() {
        workers.close();
    }
    if config.delivery.enabled {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            let counts = store.counts().await?;
            if counts["pending"].as_u64().unwrap_or(0) + counts["inflight"].as_u64().unwrap_or(0)
                == 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    dispatcher_stop.cancel();
    if let Some(dispatcher) = dispatcher {
        dispatcher.await??;
    }
    let report = Report {
        cameras: reports,
        cpu_workers,
        outbox: store.counts().await?,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        python_runtime: false,
        replay: config.experimental_replay,
    };
    std::fs::write(
        config.state_dir.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    drop(owner);
    if let Some(error) = fatal {
        return Err(error);
    }
    Ok(report)
}
pub async fn inspect(config: &Config) -> Result<serde_json::Value> {
    Outbox::open(config).await?.counts().await
}

//! Live annotations follow the latest source frame and inference snapshot.
use crate::{
    capture::{ConversionCadence, Latest},
    config::Model,
    executor::Cpu,
    hazard::{HazardResult, Track},
    locale::Status,
    publish,
    render::{Preview, Renderer, Text, has_visible_tracks},
};
use anyhow::Result;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{Notify, watch};
use tokio_util::sync::CancellationToken;
pub struct Snapshot {
    pub sequence: u64,
    pub generation: u64,
    pub captured: Instant,
    pub tracks: Arc<Vec<Track>>,
    pub hazard: Arc<HazardResult>,
}
#[derive(Default)]
pub struct Stats {
    pub rendered: AtomicU64,
    pub skipped: AtomicU64,
    pub pending: AtomicU64,
    pub passthrough: AtomicU64,
}
pub struct Actor {
    pub updates: watch::Sender<Option<Arc<Snapshot>>>,
    pub task: tokio::task::JoinHandle<Result<()>>,
    pub stats: Arc<Stats>,
    changed: Arc<Notify>,
}
impl Actor {
    pub async fn finish(self) -> Result<()> {
        drop(self.updates);
        self.changed.notify_waiters();
        self.task.await?
    }
}
#[derive(Clone)]
pub struct Target {
    pub language: String,
    pub sender: publish::Sender,
    pub fps: u32,
    pub size: Option<(u32, u32)>,
}

pub fn spawn(
    model: Model,
    mut source: watch::Receiver<Latest>,
    targets: Vec<Target>,
    text: Arc<Text>,
    cpu: Cpu,
    changed: Arc<Notify>,
    max_age_ms: u64,
    stop: CancellationToken,
) -> Actor {
    let (updates, mut receiver) = watch::channel::<Option<Arc<Snapshot>>>(None);
    let stats = Arc::new(Stats::default());
    let counters = stats.clone();
    let demand = changed.clone();
    let task = tokio::spawn(async move {
        let mut renderers = HashMap::<(String, (u32, u32)), Renderer>::new();
        let mut previews = HashMap::<(u32, u32), Preview>::new();
        let fps = targets
            .iter()
            .map(|target| target.fps)
            .max()
            .unwrap_or(1)
            .max(30);
        let mut interval = tokio::time::interval(Duration::from_secs_f64(1. / f64::from(fps)));
        let mut cadence = targets
            .iter()
            .map(|target| ConversionCadence::new(target.fps))
            .collect::<Vec<_>>();
        let renditions = Arc::new(
            targets
                .iter()
                .map(|target| (target.language.clone(), target.size))
                .collect::<Vec<_>>(),
        );
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut previous = vec![None; targets.len()];
        let model = Arc::new(model);
        let mut refresh = false;
        loop {
            // Arm the waiter before reading the gates so activation cannot be lost.
            let demand_changed = demand.notified();
            if targets.iter().all(|target| !target.sender.active()) {
                renderers.clear();
                previews.clear();
                previous.fill(None);
                if receiver.has_changed().is_err() {
                    break;
                }
                tokio::select! {
                    _ = stop.cancelled() => break,
                    _ = demand_changed => {
                        // The last image can already be marked as seen while
                        // the camera is stalled. Reoffer it to the new viewer.
                        refresh = true;
                        interval.reset_immediately();
                    }
                }
                continue;
            }
            let retained = source
                .borrow()
                .frame
                .as_ref()
                .is_some_and(|frame| frame.host.is_some());
            if !std::mem::take(&mut refresh) && receiver.has_changed().is_ok() {
                if retained {
                    // The YUV publisher owns the output clock and rasterises
                    // only the frame it encodes. A second 20 FPS clock here
                    // would compound sampling losses when the phases differ.
                    let source_open = source.has_changed().is_ok();
                    tokio::select! {
                        _ = stop.cancelled() => break,
                        _ = demand_changed => (),
                        _ = source.changed(), if source_open => (),
                        _ = receiver.changed() => (),
                    }
                } else {
                    tokio::select! {
                        _ = stop.cancelled() => break,
                        _ = demand_changed => continue,
                        _ = interval.tick() => (),
                    }
                }
            }
            let finishing = receiver.has_changed().is_err();
            let latest = source.borrow_and_update().clone();
            let Some(frame) = latest.frame else {
                if finishing {
                    break;
                }
                continue;
            };
            let snapshot = receiver.borrow_and_update().clone();
            let key = (
                frame.generation,
                frame.sequence,
                snapshot.as_ref().map(|s| s.sequence),
            );
            let fresh = snapshot
                .as_ref()
                .filter(|snapshot| {
                    snapshot.generation == frame.generation
                        && snapshot.captured.elapsed() <= Duration::from_millis(max_age_ms)
                })
                .cloned();
            let clean = fresh.as_ref().is_some_and(|snapshot| {
                snapshot.hazard.warnings.is_empty()
                    && snapshot.hazard.cone_polys.is_empty()
                    && snapshot.hazard.pole_polys.is_empty()
                    && !has_visible_tracks(&model, &snapshot.tracks)
            });
            let now = Instant::now();
            let active: Vec<_> = targets
                .iter()
                .map(|target| target.sender.active())
                .collect();
            let mut next_cadence = cadence.clone();
            let due: Vec<_> = active
                .iter()
                .zip(&mut next_cadence)
                .zip(&mut previous)
                .map(|((&active, cadence), previous)| {
                    if !active {
                        *previous = None;
                        return false;
                    }
                    *previous != Some(key)
                        && (frame.host.is_some() || clean || finishing || cadence.admit(now))
                })
                .collect();
            if !due.iter().any(|&due| due) {
                if finishing {
                    break;
                }
                continue;
            }
            let images = if clean {
                // The publisher already samples its latest frame at the output FPS.
                // Two equal-rate samplers can otherwise miss each other's updates.
                counters.passthrough.fetch_add(1, Ordering::Relaxed);
                due.iter().map(|&due| due.then(|| frame.clone())).collect()
            } else {
                let permit = if finishing {
                    cpu.permit().await?
                } else {
                    let Some(permit) = cpu.try_permit() else {
                        counters.skipped.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    permit
                };
                let model = model.clone();
                let renditions = renditions.clone();
                let render_due = due.clone();
                let text = text.clone();
                let mut current = std::mem::take(&mut renderers);
                let mut scalers = std::mem::take(&mut previews);
                let pending = fresh.is_none();
                let (next_renderers, next_scalers, images) = cpu
                    .run_permitted(permit, move || {
                        let empty = HazardResult::default();
                        let (tracks, hazard) =
                            fresh.as_ref().map_or((&[][..], &empty), |snapshot| {
                                (snapshot.tracks.as_slice(), snapshot.hazard.as_ref())
                            });
                        let original = (frame.width, frame.height);
                        let dimensions = |size: Option<(u32, u32)>| {
                            size.filter(|&(w, h)| {
                                frame.host.is_some() || (w <= original.0 && h <= original.1)
                            })
                            .unwrap_or(original)
                        };
                        current.retain(|(language, shape), _| {
                            renditions.iter().zip(&active).any(|((lang, size), on)| {
                                *on && lang == language && dimensions(*size) == *shape
                            })
                        });
                        scalers.retain(|shape, _| {
                            renditions
                                .iter()
                                .zip(&active)
                                .any(|((_, size), on)| *on && dimensions(*size) == *shape)
                        });

                        let mut resized = HashMap::new();
                        let mut rendered = HashMap::new();
                        let images = renditions
                            .iter()
                            .zip(render_due)
                            .map(|((language, size), due)| {
                                if !due {
                                    return Ok(None);
                                }
                                let shape = dimensions(*size);
                                let key = (language.clone(), shape);
                                if let Some(image) = rendered.get(&key) {
                                    return Ok(Some(Arc::clone(image)));
                                }
                                let prepared = if shape == original || frame.host.is_some() {
                                    &frame
                                } else {
                                    if let std::collections::hash_map::Entry::Vacant(entry) =
                                        resized.entry(shape)
                                    {
                                        let scaler = scalers
                                            .entry(shape)
                                            .or_insert_with(|| Preview::new(shape));
                                        entry.insert(scaler.scale(&frame)?);
                                    }
                                    &resized[&shape]
                                };
                                let renderer = current.entry(key.clone()).or_insert_with(|| {
                                    Renderer::new(shape.0, shape.1, text.clone())
                                });
                                let Some(image) = renderer.render_scaled(
                                    prepared,
                                    original,
                                    tracks,
                                    hazard,
                                    &model,
                                    language,
                                    pending.then(|| Status::Pending.text(language)),
                                )?
                                else {
                                    return Ok(None);
                                };
                                rendered.insert(key, image.clone());
                                Ok(Some(image))
                            })
                            .collect::<Result<Vec<_>>>();
                        (current, scalers, images)
                    })
                    .await?;
                renderers = next_renderers;
                previews = next_scalers;
                counters.rendered.fetch_add(1, Ordering::Relaxed);
                if pending {
                    counters.pending.fetch_add(1, Ordering::Relaxed);
                }
                images?
            };
            // A full CPU queue must not consume this rendition's next render slot.
            cadence = next_cadence;
            for ((target, image), previous) in targets.iter().zip(images).zip(&mut previous) {
                let Some(image) = image else { continue };
                if !target.sender.send(image, &stop).await && !stop.is_cancelled() {
                    anyhow::bail!("Annotation encoder unavailable")
                }
                *previous = Some(key);
            }
            if finishing {
                break;
            }
        }
        Ok(())
    });
    Actor {
        updates,
        task,
        stats,
        changed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Frame, Pool};
    #[tokio::test]
    async fn playback_advances_without_inference_and_discards_old_generation_metadata() {
        let model:Model=serde_json::from_value(serde_json::json!({"key":"fixture","artifact":"fixture","sha256":"0".repeat(64),"version":"fixture"})).unwrap();
        let text = Text::new(&Default::default()).unwrap();
        let pool = Pool::new(640 * 352 * 3, 2);
        let (tx, feed) = watch::channel(Latest::default());
        let (target, mut output) = watch::channel(None);
        let stop = CancellationToken::new();
        let changed = Arc::new(Notify::new());
        let gate = crate::demand::Gate::observed(false, changed.clone());
        let actor = spawn(
            model,
            feed,
            vec![Target {
                language: "en".into(),
                sender: publish::Sender::Gated(
                    gate.clone(),
                    Box::new(publish::Sender::Latest(target)),
                ),
                fps: 15,
                size: None,
            }],
            text,
            Cpu::new(1).unwrap(),
            changed,
            1000,
            stop.clone(),
        );
        for (sequence, generation, value) in [(1, 1, 20), (2, 2, 170)] {
            let mut pixels = pool.take();
            pixels.fill(value);
            tx.send(Latest {
                frame: Some(Arc::new(Frame {
                    pixels,
                    device: None,
                    host: None,
                    annotation: None,
                    width: 640,
                    height: 352,
                    sequence,
                    generation,
                    timestamp_ms: 0,
                    captured: Instant::now(),
                    source_pts: None,
                    source_lag_ms: 0.,
                })),
                generation,
                terminal: false,
            })
            .unwrap();
            if sequence == 1 {
                assert!(
                    tokio::time::timeout(Duration::from_millis(75), output.changed())
                        .await
                        .is_err()
                );
                gate.set(true);
            }
            if generation == 2 {
                actor
                    .updates
                    .send(Some(Arc::new(Snapshot {
                        sequence: 1,
                        generation: 1,
                        captured: Instant::now(),
                        tracks: Arc::new(vec![]),
                        hazard: Arc::new(HazardResult::default()),
                    })))
                    .unwrap();
            }
            let actual = tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    output.changed().await.unwrap();
                    let image = output.borrow_and_update().clone().unwrap();
                    if image.sequence == sequence {
                        break image;
                    }
                }
            })
            .await
            .unwrap();
            assert_eq!(actual.pixels[(300 * 640 + 300) * 3], value);
        }
        gate.set(false);
        let stats = actor.stats.clone();
        tokio::time::timeout(Duration::from_secs(3), actor.finish())
            .await
            .unwrap()
            .unwrap();
        assert!(stats.pending.load(Ordering::Relaxed) >= 2);
    }
    #[tokio::test]
    async fn clean_preview_skips_composition_and_scaled_boxes_follow_source_coordinates() {
        let model: Model = serde_json::from_value(serde_json::json!({
            "key":"fixture","artifact":"fixture","sha256":"0".repeat(64),"version":"fixture",
            "class_metadata":[{"id":42,"code":"safety_cone"}]
        }))
        .unwrap();
        let pool = Pool::new(640 * 352 * 3, 3);
        let (source, feed) = watch::channel(Latest::default());
        let (sender, mut output) = watch::channel(None);
        let stop = CancellationToken::new();
        let actor = spawn(
            model,
            feed,
            vec![Target {
                language: "en".into(),
                sender: publish::Sender::Latest(sender),
                fps: 15,
                size: Some((320, 176)),
            }],
            Text::new(&Default::default()).unwrap(),
            Cpu::new(1).unwrap(),
            Arc::default(),
            1000,
            stop.clone(),
        );

        for (sequence, tracks) in [
            (1, vec![]),
            (
                2,
                vec![
                    [30., 40., 70., 80., 0.95, 42., 1., 0.],
                    [200., 100., 400., 240., 0.24, 2., 2., 0.],
                ],
            ),
            (3, vec![[200., 100., 400., 240., 0.95, 2., 3., 0.]]),
        ] {
            let mut pixels = pool.take();
            pixels.fill(40);
            let frame = Arc::new(Frame {
                pixels,
                device: None,
                host: None,
                annotation: None,
                width: 640,
                height: 352,
                sequence,
                generation: 1,
                timestamp_ms: 0,
                captured: Instant::now(),
                source_pts: None,
                source_lag_ms: 0.,
            });
            actor
                .updates
                .send(Some(Arc::new(Snapshot {
                    sequence,
                    generation: 1,
                    captured: frame.captured,
                    tracks: Arc::new(tracks),
                    hazard: Arc::new(HazardResult::default()),
                })))
                .unwrap();
            source
                .send(Latest {
                    frame: Some(frame.clone()),
                    generation: 1,
                    terminal: false,
                })
                .unwrap();
            tokio::time::timeout(Duration::from_secs(3), output.changed())
                .await
                .unwrap()
                .unwrap();
            let image = output.borrow_and_update().clone().unwrap();
            assert_eq!(image.sequence, sequence);
            if sequence < 3 {
                assert!(
                    Arc::ptr_eq(&image, &frame),
                    "An overlay without visible marks needs no frame copy"
                );
            } else {
                assert_eq!((image.width, image.height), (320, 176));
                let pixel = |x: usize, y: usize| {
                    let start = (y * 320 + x) * 3;
                    &image.pixels[start..start + 3]
                };
                assert_eq!(pixel(170, 119), &[54, 67, 244]);
                assert_eq!(pixel(200, 119), pixel(170, 119));
                assert_eq!(pixel(202, 119), &[40, 40, 40]);
                assert_eq!(pixel(215, 80), &[40, 40, 40]);
            }
        }
        stop.cancel();
        actor.task.await.unwrap().unwrap();
        assert_eq!(actor.stats.passthrough.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn retained_yuv_delivers_new_frames_without_a_second_encoder_clock() {
        let model = serde_json::from_value(serde_json::json!({
            "key": "fixture", "artifact": "fixture", "sha256": "0".repeat(64), "version": "fixture",
        }))
        .unwrap();
        let stop = CancellationToken::new();
        let mut decoder = construction_media::Decoder::open(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/videos/test.mp4"),
            stop.clone(),
            Duration::from_secs(5),
        )
        .unwrap();
        let (source, feed) = watch::channel(Latest::default());
        let (sender, mut output) = watch::channel(None);
        let changed = Arc::new(Notify::new());
        let gate = crate::demand::Gate::observed(true, changed.clone());
        let actor = spawn(
            model,
            feed,
            vec![Target {
                language: "en".into(),
                sender: publish::Sender::Gated(
                    gate.clone(),
                    Box::new(publish::Sender::Latest(sender)),
                ),
                fps: 1,
                size: Some((320, 176)),
            }],
            Text::new(&Default::default()).unwrap(),
            Cpu::new(1).unwrap(),
            changed,
            5000,
            stop.clone(),
        );
        actor
            .updates
            .send(Some(Arc::new(Snapshot {
                sequence: 1,
                generation: 1,
                captured: Instant::now(),
                tracks: Arc::new(vec![[30., 40., 120., 170., 0.95, 2., 1., 0.]]),
                hazard: Arc::new(HazardResult::default()),
            })))
            .unwrap();
        let pool = Pool::new(0, 2);
        for (sequence, timeout) in [(1, Duration::from_secs(3)), (2, Duration::from_millis(700))] {
            assert!(decoder.next_frame().unwrap().is_some());
            source
                .send(Latest {
                    generation: 1,
                    terminal: false,
                    frame: Some(Arc::new(Frame {
                        pixels: pool.take(),
                        host: Some(decoder.retain_host(640, 352).unwrap()),
                        device: None,
                        annotation: None,
                        width: 640,
                        height: 352,
                        sequence,
                        generation: 1,
                        captured: Instant::now(),
                        timestamp_ms: 0,
                        source_pts: None,
                        source_lag_ms: 0.0,
                    })),
                })
                .unwrap();
            tokio::time::timeout(timeout, output.changed())
                .await
                .unwrap()
                .unwrap();
            let output = output.borrow_and_update();
            let frame = output.as_ref().unwrap();
            assert_eq!(frame.sequence, sequence);
            assert!(frame.host.is_some() && frame.annotation.is_some());
            assert_eq!((frame.width, frame.height), (320, 176));
        }
        gate.set(false);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), output.changed())
                .await
                .is_err()
        );
        gate.set(true);
        tokio::time::timeout(Duration::from_secs(3), output.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(output.borrow_and_update().as_ref().unwrap().sequence, 2);
        stop.cancel();
        actor.finish().await.unwrap();
    }
}

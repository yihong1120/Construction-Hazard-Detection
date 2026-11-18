//! In-process decoding with bounded BGR admission. Overload drains decoded YUV
//! without converting it, so a full output queue cannot back up a live source.
use crate::{
    config::{CaptureSettings, Stream},
    frame::{Frame, Pool, SharedFrame},
};
use anyhow::{Result, ensure};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Default)]
pub struct Latest {
    pub frame: Option<SharedFrame>,
    pub terminal: bool,
    pub generation: u64,
}
pub enum Frames {
    Latest(watch::Receiver<Latest>),
    Lossless(mpsc::Receiver<SharedFrame>),
}
#[derive(Default)]
pub struct Stats {
    pub ingress: Arc<crate::rtsp::Stats>,
    pub gpu: Arc<construction_media::GpuMetrics>,
    pub gpu_admitted: AtomicU64,
    pub read_ns: AtomicU64,
    pub convert_ns: AtomicU64,
    pub converted: AtomicU64,
    pub admission_dropped: AtomicU64,
    pub rate_dropped: AtomicU64,
    pub reconnects: AtomicU64,
    pub pool_peak: AtomicU64,
}
pub struct Capture {
    pub frames: Frames,
    pub task: tokio::task::JoinHandle<Result<()>>,
    pub produced: Arc<AtomicU64>,
    pub stats: Arc<Stats>,
}
enum Target {
    Latest(watch::Sender<Latest>),
    Lossless(mpsc::Sender<SharedFrame>),
}
impl Target {
    fn send(&self, mut frame: SharedFrame, stop: &CancellationToken) -> bool {
        match self {
            Self::Latest(sender) => sender
                .send(Latest {
                    generation: frame.generation,
                    frame: Some(frame),
                    terminal: false,
                })
                .is_ok(),
            Self::Lossless(sender) => loop {
                match sender.try_send(frame) {
                    Ok(()) => return true,
                    Err(mpsc::error::TrySendError::Closed(_)) => return false,
                    Err(mpsc::error::TrySendError::Full(pending)) => frame = pending,
                }
                if stop.is_cancelled() {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(2));
            },
        }
    }
    fn invalidate(&self, generation: u64) {
        if let Self::Latest(sender) = self {
            let _ = sender.send(Latest {
                frame: None,
                generation,
                terminal: false,
            });
        }
    }
    fn finish(&self) {
        if let Self::Latest(sender) = self {
            sender.send_modify(|value| value.terminal = true);
        }
    }
}
impl Frames {
    pub async fn next(&mut self, seen: u64, stop: &CancellationToken) -> Option<SharedFrame> {
        match self {
            Self::Lossless(receiver) => {
                tokio::select! {_ = stop.cancelled()=>None,frame=receiver.recv()=>frame}
            }
            Self::Latest(receiver) => loop {
                {
                    let latest = receiver.borrow_and_update();
                    if let Some(frame) = &latest.frame {
                        if frame.sequence > seen {
                            return Some(frame.clone());
                        }
                    }
                    if latest.terminal {
                        return None;
                    }
                }
                tokio::select! {_ = stop.cancelled()=>return None,result=receiver.changed()=>if result.is_err(){return None}}
            },
        }
    }
    pub fn generation(&self) -> Option<u64> {
        match self {
            Self::Latest(receiver) => Some(receiver.borrow().generation),
            Self::Lossless(_) => None,
        }
    }
    pub fn latest(&self) -> Option<watch::Receiver<Latest>> {
        match self {
            Self::Latest(receiver) => Some(receiver.clone()),
            _ => None,
        }
    }
}
fn sleep_until(deadline: Instant, stop: &CancellationToken) -> bool {
    while Instant::now() < deadline {
        if stop.is_cancelled() {
            return false;
        }
        std::thread::sleep(
            deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10)),
        );
    }
    !stop.is_cancelled()
}
pub fn spawn(
    spec: Stream,
    lossless: bool,
    max_frames: Option<u64>,
    settings: CaptureSettings,
    clean: Option<crate::publish::Sender>,
    relay: Option<crate::relay::Sender>,
    stop: CancellationToken,
) -> Capture {
    let (target, frames) = if lossless {
        let (tx, rx) = mpsc::channel(2);
        (Target::Lossless(tx), Frames::Lossless(rx))
    } else {
        let (tx, rx) = watch::channel(Latest::default());
        (Target::Latest(tx), Frames::Latest(rx))
    };
    let produced = Arc::new(AtomicU64::new(0));
    let count = produced.clone();
    let stats = Arc::new(Stats::default());
    let counters = stats.clone();
    let runtime = tokio::runtime::Handle::current();
    let task = crate::executor::dedicated(format!("decode-{}", spec.id), move || {
        let pool = Pool::new(
            if settings.nvdec_device.is_some() || settings.host_yuv {
                0
            } else {
                spec.width as usize * spec.height as usize * 3
            },
            if lossless {
                settings.buffer_frames.max(8)
            } else {
                settings.buffer_frames
            },
        );
        let rtsp = spec.source.starts_with("rtsp://") || spec.source.starts_with("rtsps://");
        // Live HTTP(S) can time out just like RTSP, but finite HTTP recordings
        // still need file pacing and lossless runs still end at EOF.
        let remote = rtsp
            || (!lossless
                && (spec.source.starts_with("http://") || spec.source.starts_with("https://")));
        let (mut sequence, mut generation) = (0, 0);
        let mut backoff = Duration::from_millis(250);
        loop {
            if stop.is_cancelled() {
                break;
            }
            generation += 1;
            target.invalidate(generation);
            let decoder = crate::rtsp::Source::open(
                &runtime,
                settings.rust_rtsp,
                &spec.source,
                settings.nvdec_device.unwrap_or(-1),
                stop.clone(),
                Duration::from_millis(settings.read_timeout_ms),
                counters.gpu.clone(),
                counters.ingress.clone(),
            );
            let mut decoder = match decoder {
                Ok(value) => value,
                Err(error) => {
                    if !remote && !stop.is_cancelled() {
                        return Err(error);
                    }
                    if !stop.is_cancelled() {
                        tracing::warn!(camera=%spec.id, reason="open", "Capture interrupted; reconnecting");
                    }
                    if !sleep_until(Instant::now() + backoff, &stop) {
                        break;
                    }
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                    counters.reconnects.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };
            if let Some(relay) = &relay {
                decoder.tap_packets(relay.clone());
            }
            let mut liveness = Liveness::new();
            let mut clock = SourceClock::default();
            let mut began = Instant::now();
            let mut first_pts = None;
            let mut decoded_in_generation = 0u64;
            let mut cadence = ConversionCadence::new(if lossless {
                0
            } else {
                settings.max_convert_fps
            });
            let mut reconnect_reason = "cancelled";
            loop {
                if stop.is_cancelled() {
                    break;
                }
                let read_began = Instant::now();
                let decoded = decoder.next_frame();
                counters
                    .read_ns
                    .fetch_add(read_began.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let decoded = match decoded {
                    Ok(Some(value)) => value,
                    Ok(None) => {
                        reconnect_reason = "end_of_stream";
                        break;
                    }
                    Err(error) => {
                        if !remote && !stop.is_cancelled() {
                            return Err(error);
                        }
                        tracing::debug!(camera=%spec.id, %error, "Source decoder read failed");
                        reconnect_reason = "read_or_decode";
                        break;
                    }
                };
                if decoder.take_discontinuity() || (remote && liveness.timestamp_reset(decoded.pts))
                {
                    generation += 1;
                    target.invalidate(generation);
                    liveness = Liveness::new();
                    clock = SourceClock::default();
                    began = Instant::now();
                    first_pts = None;
                    decoded_in_generation = 0;
                }
                let now = Instant::now();
                let lag = clock.lag(decoded.pts, now);
                if remote && liveness.stalled_pts(decoded.pts, now, &settings) {
                    reconnect_reason = "timestamp_stall";
                    break;
                }
                if remote && settings.max_lag_ms > 0 && lag > settings.max_lag_ms as f64 {
                    reconnect_reason = "relative_timestamp_lag";
                    break;
                }
                decoded_in_generation += 1;
                if !rtsp && !lossless {
                    let elapsed = decoded
                        .pts
                        .map(|pts| pts - *first_pts.get_or_insert(pts))
                        .filter(|v| *v >= 0.)
                        .unwrap_or(
                            (decoded_in_generation - 1) as f64
                                / if decoded.fps > 0. {
                                    decoded.fps
                                } else {
                                    spec.fps as f64
                                },
                        );
                    if !sleep_until(began + Duration::from_secs_f64(elapsed), &stop) {
                        break;
                    }
                }
                sequence += 1;
                count.store(sequence, Ordering::Relaxed);
                backoff = Duration::from_millis(250);
                if !cadence.admit(Instant::now()) {
                    counters.rate_dropped.fetch_add(1, Ordering::Relaxed);
                    if max_frames.is_some_and(|n| sequence >= n) {
                        break;
                    }
                    continue;
                }
                let mut pixels = loop {
                    if let Some(buffer) = pool.try_take() {
                        break Some(buffer);
                    }
                    if !lossless || stop.is_cancelled() {
                        break None;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                };
                let mut device = None;
                let mut host = None;
                if let Some(ref mut pixels) = pixels {
                    let converted = Instant::now();
                    if settings.nvdec_device.is_some() {
                        let frame = decoder.retain_nv12()?;
                        let info = frame.info();
                        ensure!(
                            info.width == spec.width as i32 && info.height == spec.height as i32,
                            "NVDEC capture requires native source dimensions"
                        );
                        // Preserve no-PTS freeze detection without downloading
                        // every GPU frame. Valid-PTS streams need no sampling.
                        if remote
                            && decoded.pts.is_none()
                            && liveness.sample_due(Instant::now())
                            && liveness.frozen(frame.bgr_padded()?, Instant::now(), &settings)
                        {
                            reconnect_reason = "frozen_frame";
                            break;
                        }
                        device = Some(frame);
                        counters.gpu_admitted.fetch_add(1, Ordering::Relaxed);
                    } else if settings.host_yuv {
                        let frame = decoder.retain_host(spec.width, spec.height)?;
                        if remote
                            && decoded.pts.is_none()
                            && liveness.sample_due(Instant::now())
                            && liveness.frozen(
                                &frame.bgr_padded()?
                                    [..spec.width as usize * spec.height as usize * 3],
                                Instant::now(),
                                &settings,
                            )
                        {
                            reconnect_reason = "frozen_frame";
                            break;
                        }
                        host = Some(frame);
                    } else {
                        decoder.bgr(spec.width, spec.height, pixels.padded_mut())?;
                        counters.converted.fetch_add(1, Ordering::Relaxed);
                    }
                    counters
                        .convert_ns
                        .fetch_add(converted.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    if settings.nvdec_device.is_none()
                        && !settings.host_yuv
                        && remote
                        && decoded.pts.is_none()
                        && liveness.frozen(pixels, Instant::now(), &settings)
                    {
                        reconnect_reason = "frozen_frame";
                        break;
                    }
                } else {
                    counters.admission_dropped.fetch_add(1, Ordering::Relaxed);
                    if stop.is_cancelled() {
                        break;
                    }
                    continue;
                }
                let frame = Arc::new(Frame {
                    pixels: pixels.expect("admitted buffer"),
                    device,
                    host,
                    width: spec.width,
                    height: spec.height,
                    sequence,
                    generation,
                    timestamp_ms: chrono::Utc::now().timestamp_millis(),
                    captured: Instant::now(),
                    source_pts: decoded.pts,
                    source_lag_ms: lag,
                });
                counters
                    .pool_peak
                    .fetch_max(pool.allocated() as u64, Ordering::Relaxed);
                if let Some(clean) = &clean {
                    if !clean.send_blocking(frame.clone(), &stop) {
                        break;
                    }
                }
                if !target.send(frame, &stop) {
                    break;
                }
                if max_frames.is_some_and(|n| sequence >= n) {
                    break;
                }
            }
            drop(decoder);
            if !remote || stop.is_cancelled() || max_frames.is_some_and(|n| sequence >= n) {
                break;
            }
            counters.reconnects.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(camera=%spec.id, reason=reconnect_reason,"Capture interrupted; reconnecting");
            if !sleep_until(Instant::now() + backoff, &stop) {
                break;
            }
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
        target.finish();
        ensure!(sequence > 0 || stop.is_cancelled(), "No frames decoded");
        Ok(())
    });
    Capture {
        frames,
        task,
        produced,
        stats,
    }
}
/// Phase-aware live admission. Continue decoding for PTS/liveness, but avoid
/// spending BGR conversion work on frames above the explicitly chosen budget.
#[derive(Clone, Copy)]
pub(crate) struct ConversionCadence {
    interval: Option<Duration>,
    next: Option<Instant>,
}
impl ConversionCadence {
    pub(crate) fn new(fps: u32) -> Self {
        Self {
            interval: (fps > 0).then(|| Duration::from_secs_f64(1. / f64::from(fps))),
            next: None,
        }
    }
    pub(crate) fn admit(&mut self, now: Instant) -> bool {
        let Some(interval) = self.interval else {
            return true;
        };
        let tolerance = (interval / 10).min(Duration::from_millis(5));
        if let Some(next) = self.next {
            if now + tolerance < next {
                return false;
            }
            self.next = Some(if now >= next + interval {
                now + interval
            } else {
                next + interval
            });
        } else {
            self.next = Some(now + interval);
        }
        true
    }
}
#[derive(Default)]
struct SourceClock {
    anchor: Option<(Instant, f64)>,
}
impl SourceClock {
    fn lag(&mut self, pts: Option<f64>, now: Instant) -> f64 {
        let Some(pts) = pts else { return 0. };
        let (instant, first) = *self.anchor.get_or_insert((now, pts));
        if pts < first {
            self.anchor = Some((now, pts));
            return 0.;
        }
        ((now.duration_since(instant).as_secs_f64() - (pts - first)) * 1000.).max(0.)
    }
}
struct Liveness {
    pts: Option<f64>,
    advanced: Instant,
    sampled: Instant,
    changed: Instant,
    sample: Vec<u8>,
}
impl Liveness {
    fn sample_due(&self, now: Instant) -> bool {
        now.duration_since(self.sampled) >= Duration::from_secs(1)
    }
    fn new() -> Self {
        let now = Instant::now();
        Self {
            pts: None,
            advanced: now,
            sampled: now,
            changed: now,
            sample: vec![],
        }
    }
    fn timestamp_reset(&self, pts: Option<f64>) -> bool {
        self.pts
            .zip(pts)
            .is_some_and(|(previous, current)| current < previous - 1.)
    }
    fn stalled_pts(&mut self, pts: Option<f64>, now: Instant, settings: &CaptureSettings) -> bool {
        let Some(pts) = pts else { return false };
        if self.pts.is_none_or(|old| pts > old) {
            self.advanced = now;
            self.pts = Some(pts)
        }
        now.duration_since(self.advanced) >= Duration::from_secs(settings.timestamp_stall_seconds)
    }
    fn frozen(&mut self, pixels: &[u8], now: Instant, settings: &CaptureSettings) -> bool {
        if !self.sample_due(now) {
            return false;
        }
        self.sampled = now;
        let sample: Vec<_> = pixels
            .iter()
            .step_by((pixels.len() / 4096).max(1))
            .copied()
            .collect();
        let delta: u64 = sample
            .iter()
            .zip(&self.sample)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        if sample.len() != self.sample.len() || delta as f64 > sample.len() as f64 * 0.2 {
            self.changed = now
        }
        self.sample = sample;
        now.duration_since(self.changed) >= Duration::from_secs(settings.freeze_seconds)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_pts_health_sampling_waits_one_second_and_detects_a_frozen_source() {
        let mut liveness = Liveness::new();
        let start = liveness.sampled;
        let settings = CaptureSettings {
            freeze_seconds: 3,
            ..Default::default()
        };
        for second in 1..=4 {
            let tick = start + Duration::from_secs(second);
            assert!(!liveness.sample_due(tick - Duration::from_millis(1)));
            assert!(liveness.sample_due(tick));
            assert!(!liveness.stalled_pts(None, tick, &settings));
            assert_eq!(liveness.frozen(&[17; 64], tick, &settings), second == 4);
        }
    }
    #[test]
    fn conversion_budget_keeps_phase_through_jitter_without_outage_catchup() {
        let start = Instant::now();
        let mut cadence = ConversionCadence::new(15);
        let mut admitted = 0;
        for i in 0..300u64 {
            // A 30 FPS arrival clock with alternating 1 ms delivery jitter.
            let arrival = start + Duration::from_micros(i * 1_000_000 / 30 + (i % 2) * 1000);
            admitted += usize::from(cadence.admit(arrival));
        }
        assert_eq!(admitted, 150);
        let resumed = start + Duration::from_secs(100);
        assert!(cadence.admit(resumed));
        assert!(!cadence.admit(resumed + Duration::from_millis(1)));
        assert!(cadence.admit(resumed + Duration::from_micros(66_000)));
        let mut disabled = ConversionCadence::new(0);
        assert!((0..100).all(|_| disabled.admit(start)));
    }
    #[test]
    fn advancing_pts_allows_static_scenes_and_clock_measures_backlog() {
        let mut state = Liveness::new();
        let start = Instant::now();
        let settings = CaptureSettings::default();
        for i in 0..60 {
            assert!(!state.stalled_pts(Some(i as f64), start + Duration::from_secs(i), &settings));
        }
        assert!(state.stalled_pts(Some(59.), start + Duration::from_secs(105), &settings));
        let mut clock = SourceClock::default();
        assert_eq!(clock.lag(Some(10.), start), 0.);
        assert_eq!(clock.lag(Some(11.), start + Duration::from_secs(3)), 2000.);
    }
}

//! Exclusive in-process encoder/mux owners. A latest-frame mailbox caps live
//! memory; lossless recording has an explicit two-frame backpressure queue.
use crate::{
    config::Config,
    frame::{Frame, SharedFrame},
    inference::lease_root,
    lease,
};
use anyhow::Result;
use std::{
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
#[derive(Clone)]
pub enum Sender {
    Group(Vec<Sender>),
    Gated(Arc<crate::demand::Gate>, Box<Sender>),
    Latest(watch::Sender<Option<SharedFrame>>),
    Lossless(mpsc::Sender<SharedFrame>),
}
enum Receiver {
    Latest(watch::Receiver<Option<SharedFrame>>),
    Lossless(mpsc::Receiver<SharedFrame>),
}
impl Sender {
    pub fn active(&self) -> bool {
        match self {
            Self::Group(senders) => senders.iter().any(Self::active),
            Self::Gated(g, _) => g.active(),
            _ => true,
        }
    }
    fn writable(&self) -> Option<&Self> {
        match self {
            Self::Gated(g, inner) => {
                if g.active() {
                    Some(inner)
                } else {
                    if let Self::Latest(tx) = inner.as_ref() {
                        tx.send_if_modified(|value| value.take().is_some());
                    }
                    None
                }
            }
            _ => Some(self),
        }
    }

    pub async fn send(&self, frame: SharedFrame, stop: &CancellationToken) -> bool {
        let Some(base) = self.writable() else {
            return !stop.is_cancelled();
        };
        match base {
            Self::Group(senders) => {
                let mut ok = true;
                for sender in senders {
                    ok &= Box::pin(sender.send(frame.clone(), stop)).await;
                }
                ok
            }
            Self::Gated(_, _) => unreachable!(),
            Self::Latest(tx) => tx.send(Some(frame)).is_ok(),
            Self::Lossless(tx) => {
                tokio::select! {_ = stop.cancelled()=>false,result=tx.send(frame)=>result.is_ok()}
            }
        }
    }
    pub fn send_blocking(&self, mut frame: SharedFrame, stop: &CancellationToken) -> bool {
        let Some(base) = self.writable() else {
            return !stop.is_cancelled();
        };
        match base {
            Self::Group(senders) => senders.iter().fold(true, |ok, sender| {
                sender.send_blocking(frame.clone(), stop) & ok
            }),
            Self::Gated(_, _) => unreachable!(),
            Self::Latest(tx) => tx.send(Some(frame)).is_ok(),
            Self::Lossless(tx) => loop {
                match tx.try_send(frame) {
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
}
struct Encoder {
    native: construction_media::Encoder,
    _lease: Option<lease::Lease>,
}
impl Encoder {
    fn new(
        config: &Config,
        target: &str,
        width: u32,
        height: u32,
        fps: u32,
        nvenc: bool,
    ) -> Result<Self> {
        if !target.starts_with("rtsp://") && !target.starts_with("rtsps://") {
            if let Some(parent) = std::path::Path::new(target).parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let lease = if nvenc {
            lease::acquire(
                &lease_root(config),
                &format!("{}.nvenc", config.gpu.namespace),
                config.gpu.nvenc_sessions,
                1,
            )?
        } else {
            None
        };
        if let Some(lease) = lease {
            match construction_media::Encoder::open(
                target,
                width,
                height,
                fps,
                Some(config.gpu.device),
            ) {
                Ok(native) => {
                    return Ok(Self {
                        native,
                        _lease: Some(lease),
                    });
                }
                Err(_) => {
                    drop(lease);
                    tracing::warn!("NVENC initialization failed; using CPU encoding");
                }
            }
        }
        Ok(Self {
            native: construction_media::Encoder::open(target, width, height, fps, None)?,
            _lease: None,
        })
    }
}
#[derive(Default)]
pub struct Stats {
    pub ready: Arc<AtomicBool>,
    pub unique: AtomicU64,
    pub repeats: AtomicU64,
    pub offline: AtomicU64,
    pub encode_ns: AtomicU64,
    pub conversion_reused: AtomicU64,
}
pub struct Publisher {
    pub stats: Arc<Stats>,
    pub sender: Sender,
    pub task: tokio::task::JoinHandle<Result<()>>,
    pub encoded: Arc<AtomicU64>,
}
pub fn spawn(
    config: Arc<Config>,
    target: String,
    width: u32,
    height: u32,
    fps: u32,
    nvenc: bool,
    text: Arc<crate::render::Text>,
    stop: CancellationToken,
) -> Publisher {
    spawn_gated(
        config,
        target,
        width,
        height,
        fps,
        nvenc,
        text,
        crate::demand::Gate::new(true),
        stop,
    )
}
pub fn spawn_gated(
    config: Arc<Config>,
    target: String,
    width: u32,
    height: u32,
    fps: u32,
    nvenc: bool,
    text: Arc<crate::render::Text>,
    gate: Arc<crate::demand::Gate>,
    stop: CancellationToken,
) -> Publisher {
    spawn_gated_localized(
        config, target, width, height, fps, nvenc, text, "en", gate, stop,
    )
}

pub(crate) fn spawn_gated_localized(
    config: Arc<Config>,
    target: String,
    width: u32,
    height: u32,
    fps: u32,
    nvenc: bool,
    text: Arc<crate::render::Text>,
    language: &'static str,
    gate: Arc<crate::demand::Gate>,
    stop: CancellationToken,
) -> Publisher {
    let (sender, mut receiver) = if config.lossless {
        let (tx, rx) = mpsc::channel(2);
        (Sender::Lossless(tx), Receiver::Lossless(rx))
    } else {
        let (tx, rx) = watch::channel(None);
        (Sender::Latest(tx), Receiver::Latest(rx))
    };
    let sender = Sender::Gated(gate.clone(), Box::new(sender));
    let encoded = Arc::new(AtomicU64::new(0));
    let count = encoded.clone();
    let stats = Arc::new(Stats::default());
    let counters = stats.clone();
    let task = tokio::spawn(async move {
        let mut prefetched = None;
        loop {
            // Register before checking the atomic gate to avoid lost edges.
            let changed = gate.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let closed = match &receiver {
                Receiver::Latest(rx) => rx.has_changed().is_err(),
                Receiver::Lossless(rx) => rx.is_closed() && rx.is_empty(),
            };
            let available = match &receiver {
                Receiver::Latest(rx) => rx.borrow().is_some(),
                Receiver::Lossless(_) => true,
            };
            if stop.is_cancelled() || (closed && !available) {
                return Ok(());
            }
            if !gate.active() || !available {
                if closed {
                    return Ok(());
                }
                match &mut receiver {
                    Receiver::Latest(rx) => tokio::select! {
                        _=stop.cancelled()=>return Ok(()),_=changed.as_mut()=>(),_=rx.changed()=>(),
                    },
                    Receiver::Lossless(rx) => tokio::select! {
                        _=stop.cancelled()=>return Ok(()),_=changed.as_mut()=>(),
                        frame=rx.recv()=>{
                            let Some(frame)=frame else{return Ok(())};
                            if gate.active(){prefetched=Some(frame);}
                        },
                    },
                }
                continue;
            }
            let (config, target, text, gate, stop, counters, count) = (
                config.clone(),
                target.clone(),
                text.clone(),
                gate.clone(),
                stop.clone(),
                counters.clone(),
                count.clone(),
            );
            let mut initial = prefetched.take();
            let (returned, paused) = crate::executor::dedicated("encode".into(), move || {
                let mut paused = false;
                struct Clear(Arc<AtomicBool>);
                impl Drop for Clear {
                    fn drop(&mut self) {
                        self.0.store(false, Ordering::Release);
                    }
                }
                let _clear = Clear(counters.ready.clone());
                let pool = crate::frame::Pool::new(width as usize * height as usize * 3, 1);
                let mut offline = None;
                let mut encoder: Option<Encoder> = None;
                let live = target.starts_with("rtsp://") || target.starts_with("rtsps://");
                let mut retry_at = Instant::now();
                let mut retry_delay = Duration::from_millis(250);
                let mut prefer_nvenc = nvenc;
                let interval = Duration::from_secs_f64(1. / fps as f64);
                let mut next = Instant::now();
                let mut previous = None;
                // Identity includes both source pixels and the rendered
                // annotation, which can change for the same source sequence.
                let mut converted: Weak<Frame> = Weak::new();
                loop {
                    if stop.is_cancelled() {
                        break;
                    }
                    if !gate.active() {
                        paused = true;
                        break;
                    }
                    if Instant::now() < retry_at {
                        std::thread::sleep(Duration::from_millis(20));
                        continue;
                    }
                    let frame = match &mut receiver {
                        Receiver::Lossless(rx) => {
                            match initial.take().map(Ok).unwrap_or_else(|| rx.try_recv()) {
                                Ok(frame) => Some(frame),
                                Err(mpsc::error::TryRecvError::Disconnected) => None,
                                Err(mpsc::error::TryRecvError::Empty) => {
                                    std::thread::sleep(Duration::from_millis(2));
                                    continue;
                                }
                            }
                        }
                        Receiver::Latest(rx) => {
                            let now = Instant::now();
                            if now < next {
                                std::thread::sleep(next - now);
                                continue;
                            }
                            next = (next + interval).max(now);
                            let frame = rx.borrow_and_update().clone();
                            if frame.is_none() && rx.has_changed().is_ok() {
                                continue;
                            }
                            frame
                        }
                    };
                    let Some(frame) = frame else { break };
                    let is_offline =
                        !config.lossless && frame.captured.elapsed() > Duration::from_secs(5);
                    let frame = if is_offline {
                        if offline.is_none() {
                            offline = Some(text.offline(&frame, &pool, width, height, language)?);
                        }
                        offline.as_ref().unwrap().clone()
                    } else {
                        offline = None;
                        frame
                    };
                    let encode_began = Instant::now();
                    let mut write_failed = false;
                    let reuse =
                        !config.lossless && std::ptr::eq(converted.as_ptr(), Arc::as_ptr(&frame));
                    let result = (|| -> Result<()> {
                        if encoder.is_none() {
                            encoder = Some(Encoder::new(
                                &config,
                                &target,
                                width,
                                height,
                                fps,
                                prefer_nvenc,
                            )?);
                        }
                        let native = &mut encoder.as_mut().expect("encoder").native;
                        let result = if reuse {
                            native.repeat_frame()
                        } else {
                            native.write_scaled(frame.bgr_padded()?, frame.width, frame.height)
                        };
                        write_failed = result.is_err();
                        result
                    })();
                    if let Err(error) = result {
                        if !live {
                            return Err(error.context("File encoder failed"));
                        }
                        counters.ready.store(false, Ordering::Release);
                        encoder = None;
                        converted = Weak::new();
                        if write_failed {
                            prefer_nvenc = false;
                        }
                        tracing::warn!("Live encoder unavailable; retrying");
                        retry_at = Instant::now() + retry_delay;
                        retry_delay = (retry_delay * 2).min(Duration::from_secs(5));
                        continue;
                    }
                    if reuse {
                        counters.conversion_reused.fetch_add(1, Ordering::Relaxed);
                    } else if !config.lossless {
                        // A weak reference prevents address reuse without pinning the pixel buffer.
                        converted = Arc::downgrade(&frame);
                    }
                    retry_delay = Duration::from_millis(250);
                    counters.ready.store(
                        encoder.as_ref().expect("encoder").native.packets_written() > 0,
                        Ordering::Release,
                    );
                    counters
                        .encode_ns
                        .fetch_add(encode_began.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let key = (frame.generation, frame.sequence);
                    if previous == Some(key) {
                        counters.repeats.fetch_add(1, Ordering::Relaxed);
                    } else {
                        counters.unique.fetch_add(1, Ordering::Relaxed);
                    }
                    previous = Some(key);
                    if is_offline {
                        counters.offline.fetch_add(1, Ordering::Relaxed);
                    }
                    count.fetch_add(1, Ordering::Relaxed);
                    if let Receiver::Latest(rx) = &receiver {
                        if rx.has_changed().is_err() {
                            break;
                        }
                    }
                }
                if let Some(mut encoder) = encoder {
                    let finished = encoder.native.finish();
                    if !live {
                        finished?;
                    }
                }
                Ok((receiver, paused))
            })
            .await??;
            receiver = returned;
            if !paused {
                return Ok(());
            }
        }
    });
    Publisher {
        stats,
        sender,
        task,
        encoded,
    }
}

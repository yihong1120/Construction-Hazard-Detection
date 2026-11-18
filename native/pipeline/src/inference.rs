use crate::{
    config::{Config, Model},
    frame::SharedFrame,
    lease,
};
use anyhow::{Context, Result, bail, ensure};
use construction_inference::{
    core::Image,
    engine::{NativeEngine, Nv12Image, Nv12Planes},
};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, watch};

pub type Detection = [f64; 6];
pub struct Request {
    pub enqueued: Instant,
    pub source: Input,
    pub reply: oneshot::Sender<Result<Prediction, String>>,
}
pub enum Input {
    Fixed(SharedFrame),
    Latest {
        feed: watch::Receiver<crate::capture::Latest>,
        minimum: u64,
    },
}
impl Input {
    fn select(self) -> Option<SharedFrame> {
        match self {
            Self::Fixed(frame) => Some(frame),
            Self::Latest { feed, minimum } => feed
                .borrow()
                .frame
                .as_ref()
                .filter(|f| f.sequence >= minimum)
                .cloned(),
        }
    }
}
#[derive(Clone, Default, serde::Serialize)]
pub struct Timing {
    pub batch_id: u64,
    pub gpu_preprocess: bool,
    pub nv12_preprocess: bool,
    pub cpu_exact_images: usize,
    pub queue_ms: f64,
    pub admission_ms: f64,
    pub prepare_ms: f64,
    pub submit_ms: f64,
    pub gpu_wait_decode_ms: f64,
    pub batch_span_ms: f64,
    pub in_flight: usize,
}
pub struct Prediction {
    pub timing: Timing,
    pub frame: SharedFrame,
    pub detections: Vec<Detection>,
    pub batch_size: usize,
}
#[derive(Clone)]
pub struct Client {
    pub sender: mpsc::Sender<Request>,
    pub model: Model,
}
pub struct Workers {
    clients: HashMap<String, Client>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

fn verified_cache(model: &Model, state: &Path) -> Result<PathBuf> {
    if let (Some(path), Some(digest)) = (&model.source_artifact, &model.source_sha256) {
        ensure!(
            hex::encode(Sha256::digest(std::fs::read(path)?)) == digest.to_lowercase(),
            "Published source digest mismatch"
        );
    }
    let bytes = std::fs::read(&model.artifact).context("Cannot read engine artifact")?;
    ensure!(
        hex::encode(Sha256::digest(&bytes)) == model.sha256.to_lowercase(),
        "Engine digest mismatch"
    );
    let dir = state.join("models");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.engine", model.sha256.to_lowercase()));
    if path.exists() {
        ensure!(
            hex::encode(Sha256::digest(std::fs::read(&path)?)) == model.sha256.to_lowercase(),
            "Cached engine digest mismatch"
        );
    } else {
        let temporary = dir.join(format!("{}.tmp", uuid::Uuid::new_v4()));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, &path)?;
        std::fs::File::open(&dir)?.sync_all()?;
    }
    Ok(path)
}
pub fn lease_root(config: &Config) -> PathBuf {
    config.gpu.lease_dir.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("chd-gpu-{}", unsafe { libc::getuid() }))
    })
}
enum Detector {
    Native(NativeEngine),
    Replay(Vec<Vec<Detection>>),
}
impl Detector {
    fn capacity(&self) -> usize {
        match self {
            Self::Native(engine) => engine.max_batch,
            Self::Replay(_) => 64,
        }
    }
}
type Reply = oneshot::Sender<Result<Prediction, String>>;
type Selected = (SharedFrame, Reply, Timing);
enum Work {
    Ticket(u64),
    Ready(Vec<Vec<Detection>>),
}
struct Pending {
    inputs: Vec<Selected>,
    work: Work,
    began: Instant,
    _permit: Option<lease::Lease>,
}
fn millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}
fn drain(detector: &mut Detector, pending: Pending) {
    let started = Instant::now();
    let rows = match pending.work {
        Work::Ticket(ticket) => match detector {
            Detector::Native(engine) => engine
                .collect(ticket)
                .map(|rows| {
                    rows.into_iter()
                        .map(|r| r.into_iter().map(|r| r.map(f64::from)).collect())
                        .collect()
                })
                .map_err(anyhow::Error::from),
            _ => unreachable!(),
        },
        Work::Ready(rows) => Ok(rows),
    };
    let waited = millis(started);
    let span = millis(pending.began);
    let batch_size = pending.inputs.len();
    match rows {
        Ok(rows) if rows.len() == batch_size => {
            for ((frame, reply, mut timing), detections) in pending.inputs.into_iter().zip(rows) {
                timing.gpu_wait_decode_ms = waited;
                timing.batch_span_ms = span;
                let _ = reply.send(Ok(Prediction {
                    frame,
                    detections,
                    batch_size,
                    timing,
                }));
            }
        }
        _ => {
            for (_, reply, _) in pending.inputs {
                let _ = reply.send(Err("Native inference failed".into()));
            }
        }
    }
}
fn admission(settings: &Config) -> Result<Option<lease::Lease>> {
    if settings.gpu.inference_slots == 0 {
        return Ok(None);
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(permit) = lease::acquire(
            &lease_root(settings),
            &format!("{}.inference", settings.gpu.namespace),
            settings.gpu.inference_slots,
            1,
        )? {
            return Ok(Some(permit));
        }
        if Instant::now() >= deadline {
            bail!("Inference admission timeout")
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn run_batches(
    detector: &mut Detector,
    settings: &Config,
    capacity: usize,
    receiver: &mut mpsc::Receiver<Request>,
    preprocess: Option<&rayon::ThreadPool>,
) {
    let depth = settings.gpu.pipeline_depth;
    let mut pending = VecDeque::<Pending>::new();
    let mut batch_id = 0u64;
    loop {
        // Never wait for another camera while a result is already in flight.
        let first = if pending.is_empty() {
            receiver.blocking_recv()
        } else {
            receiver.try_recv().ok()
        };
        let Some(first) = first else {
            if let Some(batch) = pending.pop_front() {
                drain(detector, batch);
                continue;
            } else {
                break;
            }
        };
        let mut requests = vec![first];
        let deadline = Instant::now() + Duration::from_millis(2);
        while requests.len() < capacity && Instant::now() < deadline {
            match receiver.try_recv() {
                Ok(value) => requests.push(value),
                Err(mpsc::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_micros(100))
                }
                Err(_) => break,
            }
        }
        requests.retain(|r| !r.reply.is_closed());
        if requests.is_empty() {
            continue;
        }
        // The single TensorRT context may reuse two buffers at one shape, but
        // must be completely drained before changing its dynamic batch shape.
        if pending
            .back()
            .is_some_and(|b| b.inputs.len() != requests.len())
        {
            while let Some(batch) = pending.pop_front() {
                drain(detector, batch);
            }
        }
        let admitted = Instant::now();
        let permit = match admission(settings) {
            Ok(value) => value,
            Err(_) => {
                for r in requests {
                    let _ = r.reply.send(Err("GPU admission failed".into()));
                }
                continue;
            }
        };
        let admission_ms = millis(admitted);
        let mut inputs = vec![];
        for request in requests {
            if let Some(frame) = request.source.select() {
                let timing = Timing {
                    queue_ms: (millis(request.enqueued) - admission_ms).max(0.),
                    admission_ms,
                    ..Default::default()
                };
                inputs.push((frame, request.reply, timing));
            } else {
                let _ = request
                    .reply
                    .send(Err("Source generation unavailable".into()));
            }
        }
        if inputs.is_empty() {
            continue;
        }
        if pending
            .back()
            .is_some_and(|b| b.inputs.len() != inputs.len())
        {
            while let Some(batch) = pending.pop_front() {
                drain(detector, batch);
            }
        }
        let began = Instant::now();
        let work = (|| -> Result<Work> {
            match detector {
                Detector::Native(engine) => {
                    if inputs.iter().all(|(f, _, _)| f.device.is_some()) {
                        let images: Vec<_> = inputs
                            .iter()
                            .map(|(f, _, _)| {
                                let device = f.device.as_ref().expect("NV12 batch").clone();
                                let p = device.info();
                                // SAFETY: the AVFrame owner retains immutable planes and its
                                // ready event, on the decoder's primary CUDA context.
                                unsafe {
                                    Nv12Image::new(
                                        Nv12Planes {
                                            y: p.y,
                                            uv: p.uv,
                                            ready: p.ready,
                                            lut: p.lut,
                                            width: p.width,
                                            height: p.height,
                                            yp: p.yp,
                                            uvp: p.uvp,
                                            colorspace: p.colorspace,
                                            range: p.range,
                                            device: p.device,
                                        },
                                        device,
                                    )
                                }
                            })
                            .collect();
                        let submitted = Instant::now();
                        if let Some(ticket) = engine.submit_nv12(&images)? {
                            let submit_ms = millis(submitted);
                            for (_, _, timing) in &mut inputs {
                                timing.gpu_preprocess = true;
                                timing.nv12_preprocess = true;
                                timing.submit_ms = submit_ms;
                            }
                            return Ok(Work::Ticket(ticket));
                        }
                    }
                    if inputs.iter().any(|(f, _, _)| f.host.is_some()) {
                        let shapes: Vec<_> = inputs
                            .iter()
                            .map(|(f, _, _)| (f.height as usize, f.width as usize))
                            .collect();
                        let fill = |i: usize,
                                    l: &construction_inference::core::Layout,
                                    target: &mut [u8]| {
                            let frame = &inputs[i].0;
                            if frame.host.as_ref().is_some_and(|host| {
                                host.letterbox(target, l.h, l.w, l.rh, l.rw, l.top, l.left)
                            }) {
                                return Ok(1);
                            }
                            let data = frame.bgr().map_err(|e| {
                                construction_inference::core::Error::Runtime(e.to_string())
                            })?;
                            NativeEngine::fill_letterbox(
                                &Image {
                                    h: frame.height as usize,
                                    w: frame.width as usize,
                                    data,
                                },
                                l,
                                target,
                            )?;
                            Ok(0)
                        };
                        if settings.gpu.direct_pinned {
                            let (ticket, count) =
                                engine.submit_surfaces_pinned(&shapes, preprocess, fill)?;
                            for (_, _, timing) in &mut inputs {
                                timing.prepare_ms = millis(began);
                                timing.cpu_exact_images = count;
                            }
                            return Ok(Work::Ticket(ticket));
                        }
                        let (prepared, count) =
                            engine.prepare_surfaces_each(&shapes, preprocess, fill)?;
                        let prepare_ms = millis(began);
                        let submitted = Instant::now();
                        let work = if settings.gpu.asynchronous {
                            Work::Ticket(engine.submit_prepared(prepared)?)
                        } else {
                            Work::Ready(
                                engine
                                    .run_prepared(prepared)?
                                    .into_iter()
                                    .map(|r| r.into_iter().map(|r| r.map(f64::from)).collect())
                                    .collect(),
                            )
                        };
                        let submit_ms = millis(submitted);
                        for (_, _, timing) in &mut inputs {
                            timing.prepare_ms = prepare_ms;
                            timing.submit_ms = submit_ms;
                            timing.cpu_exact_images = count;
                        }
                        return Ok(work);
                    }
                    let images: Vec<_> = inputs
                        .iter()
                        .map(|(f, _, _)| {
                            Ok(Image {
                                h: f.height as usize,
                                w: f.width as usize,
                                data: f.bgr()?,
                            })
                        })
                        .collect::<Result<_>>()?;
                    if settings.gpu.exact_resize {
                        let submitted = Instant::now();
                        if let Some(ticket) = engine.submit_images_exact(&images)? {
                            let submit_ms = millis(submitted);
                            for (_, _, timing) in &mut inputs {
                                timing.gpu_preprocess = true;
                                timing.submit_ms = submit_ms;
                            }
                            return Ok(Work::Ticket(ticket));
                        }
                    }
                    let (prepared, cpu_exact_images) = if settings.exact_cpu_resize {
                        engine.prepare_images_fast(&images, preprocess)?
                    } else {
                        (
                            match preprocess {
                                Some(pool) => engine.prepare_images_parallel(&images, pool)?,
                                None => engine.prepare_images(&images)?,
                            },
                            0,
                        )
                    };
                    let prepare_ms = millis(began);
                    let submitted = Instant::now();
                    let work = if settings.gpu.asynchronous {
                        Work::Ticket(engine.submit_prepared(prepared)?)
                    } else {
                        Work::Ready(
                            engine
                                .run_prepared(prepared)?
                                .into_iter()
                                .map(|r| r.into_iter().map(|r| r.map(f64::from)).collect())
                                .collect(),
                        )
                    };
                    let submit_ms = millis(submitted);
                    for (_, _, timing) in &mut inputs {
                        timing.cpu_exact_images = cpu_exact_images;
                        timing.prepare_ms = prepare_ms;
                        timing.submit_ms = submit_ms;
                    }
                    Ok(work)
                }
                Detector::Replay(rows) => Ok(Work::Ready(
                    inputs
                        .iter()
                        .map(|(frame, _, _)| {
                            rows[(frame.sequence.saturating_sub(1) as usize) % rows.len()].clone()
                        })
                        .collect(),
                )),
            }
        })();
        match work {
            Ok(work) => {
                batch_id = batch_id.wrapping_add(1);
                for (_, _, timing) in &mut inputs {
                    timing.batch_id = batch_id;
                    timing.in_flight = pending.len() + 1;
                }
                pending.push_back(Pending {
                    inputs,
                    work,
                    began,
                    _permit: permit,
                });
                if pending.len() >= depth {
                    drain(detector, pending.pop_front().expect("in-flight batch"));
                }
            }
            Err(_) => {
                for (_, reply, _) in inputs {
                    let _ = reply.send(Err("Native inference failed".into()));
                }
            }
        }
    }
}
impl Workers {
    pub fn new(config: &Config) -> Result<Self> {
        let mut workers = Self {
            clients: HashMap::new(),
            threads: Vec::new(),
        };
        let preprocess = if config.preprocess_workers > 1 {
            Some(std::sync::Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.preprocess_workers)
                    .thread_name(|i| format!("prepare-{i}"))
                    .build()?,
            ))
        } else {
            None
        };
        for model in &config.models {
            let (sender, mut receiver) = mpsc::channel::<Request>(config.pending_requests);
            let (ready_send, ready_receive) = std::sync::mpsc::sync_channel(1);
            let model = model.clone();
            let spec = model.clone();
            let settings = config.clone();
            let preprocess = preprocess.clone();
            let thread = std::thread::Builder::new()
                .name(format!("infer-{}", model.key))
                .spawn(move || {
                    let loaded = (|| -> Result<(Detector, Option<lease::Lease>)> {
                        let memory = if settings.gpu.memory_limit_mb > 0 {
                            lease::acquire(
                                &lease_root(&settings),
                                &format!("{}.memory", settings.gpu.namespace),
                                settings.gpu.memory_limit_mb / 64,
                                settings.gpu.model_reservation_mb.div_ceil(64),
                            )?
                            .context("GPU model memory budget exhausted")?
                            .into()
                        } else {
                            None
                        };
                        let detector = if let Some(path) = &model.replay {
                            let rows: Vec<Vec<Detection>> =
                                serde_json::from_slice(&std::fs::read(path)?)?;
                            ensure!(
                                !rows.is_empty()
                                    && rows.iter().all(|r| r.len() <= 3000
                                        && r.iter().flatten().all(|v| v.is_finite())),
                                "Invalid replay detections"
                            );
                            Detector::Replay(rows)
                        } else {
                            let path = verified_cache(&model, &settings.state_dir)?;
                            Detector::Native(NativeEngine::new(
                                path.to_str().context("Engine path is not UTF8")?,
                                settings.batch_size,
                                model.imgsz,
                                settings.gpu.device,
                                false,
                                0.25,
                                settings.gpu.asynchronous,
                            )?)
                        };
                        Ok((detector, memory))
                    })();
                    let (mut detector, memory) = match loaded {
                        Ok(value) => {
                            let _ = ready_send.send(Ok::<_, String>(()));
                            value
                        }
                        Err(_) => {
                            let _ = ready_send
                                .send(Err("Model validation or initialization failed".into()));
                            return;
                        }
                    };
                    let capacity = settings.batch_size.min(detector.capacity());
                    run_batches(
                        &mut detector,
                        &settings,
                        capacity,
                        &mut receiver,
                        preprocess.as_deref(),
                    );
                    drop(detector);
                    drop(memory);
                })?;
            ready_receive
                .recv_timeout(Duration::from_secs(90))
                .context("Model worker startup timed out")?
                .map_err(anyhow::Error::msg)?;
            workers.clients.insert(
                spec.key.clone(),
                Client {
                    sender,
                    model: spec,
                },
            );
            workers.threads.push(thread);
        }
        Ok(workers)
    }
    pub fn get(&self, key: &str) -> Result<Client> {
        self.clients
            .get(key)
            .cloned()
            .context("Unknown model worker")
    }
    pub fn close(mut self) {
        self.clients.clear();
        for thread in self.threads {
            let _ = thread.join();
        }
    }
}
impl Client {
    pub async fn predict(&self, source: Input) -> Result<Prediction> {
        let (reply, result) = oneshot::channel();
        self.sender
            .send(Request {
                source,
                reply,
                enqueued: Instant::now(),
            })
            .await
            .context("Inference worker unavailable")?;
        tokio::time::timeout(Duration::from_secs(30), result)
            .await
            .context("Inference result timeout")??
            .map_err(anyhow::Error::msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        capture::Latest,
        frame::{Frame, Pool},
    };
    use std::sync::Arc;
    fn frame(pool: &Pool, sequence: u64, generation: u64) -> SharedFrame {
        Arc::new(Frame {
            pixels: pool.take(),
            device: None,
            host: None,
            width: 2,
            height: 2,
            sequence,
            generation,
            timestamp_ms: 0,
            captured: Instant::now(),
            source_pts: None,
            source_lag_ms: 0.,
        })
    }
    #[test]
    fn two_slot_batches_keep_order_ownership_and_drain_partial_tails() {
        let settings:Config=serde_json::from_value(serde_json::json!({"state_dir":"unused","models":[],"streams":[],"gpu":{"asynchronous":true,"pipeline_depth":2}})).unwrap();
        let pool = Pool::new(12, 10);
        let (tx, mut rx) = mpsc::channel(16);
        let mut replies = vec![];
        for sequence in 1..=10 {
            let (reply, receive) = oneshot::channel();
            tx.try_send(Request {
                enqueued: Instant::now(),
                source: Input::Fixed(frame(&pool, sequence, 1)),
                reply,
            })
            .unwrap();
            replies.push(receive);
        }
        drop(tx);
        let rows = (1..=10)
            .map(|n| vec![[n as f64, 0., 1., 1., 0.9, 0.]])
            .collect();
        run_batches(&mut Detector::Replay(rows), &settings, 4, &mut rx, None);
        let mut overlapped = false;
        for (index, reply) in replies.into_iter().enumerate() {
            let prediction = reply.blocking_recv().unwrap().unwrap();
            assert_eq!(prediction.frame.sequence, index as u64 + 1);
            assert_eq!(prediction.detections[0][0], index as f64 + 1.);
            assert!(prediction.timing.in_flight <= 2);
            overlapped |= prediction.timing.in_flight == 2;
        }
        assert!(overlapped);
        // Every reply is gone, so all ten source buffers must be reusable.
        let mut buffers = vec![];
        while let Some(buffer) = pool.try_take() {
            buffers.push(buffer);
        }
        assert_eq!(buffers.len(), 10);
    }
    #[test]
    fn queued_live_request_selects_at_dispatch_and_does_not_pin_old_pixels() {
        let pool = Pool::new(12, 1);
        let first = frame(&pool, 1, 1);
        let (tx, feed) = watch::channel(Latest {
            frame: Some(first),
            generation: 1,
            terminal: false,
        });
        let pending = Input::Latest { feed, minimum: 1 };
        tx.send_modify(|v| v.frame = None);
        // Only one allocation exists: a queued request must not retain frame 1.
        let newest = frame(&pool, 7, 2);
        tx.send_modify(|v| {
            v.frame = Some(newest.clone());
            v.generation = 2;
        });
        let selected = pending.select().unwrap();
        assert!(Arc::ptr_eq(&selected, &newest));
        assert_eq!((selected.sequence, selected.generation), (7, 2));
        let unavailable = Input::Latest {
            feed: tx.subscribe(),
            minimum: 7,
        };
        tx.send_modify(|v| {
            v.frame = None;
            v.generation = 3;
        });
        assert!(unavailable.select().is_none());
    }
}

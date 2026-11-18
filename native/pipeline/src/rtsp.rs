//! Native Rust TCP RTSP/RTP ingress; the codec backend is independent.
use anyhow::{Context, Result, bail, ensure};
use construction_media::{Decoded, Decoder, GpuMetrics, Nv12Frame, PacketDecoder, PacketOutput};
use futures_util::StreamExt;
use retina::{
    client::{Credentials, Session, SessionOptions, SetupOptions, TeardownPolicy, Transport},
    codec::{CodecItem, ParametersRef},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{runtime::Handle, sync::mpsc, task::AbortHandle};
use tokio_util::sync::CancellationToken;

#[derive(Default)]
pub struct Stats {
    pub frames: AtomicU64,
    pub bytes: AtomicU64,
    pub lost_packets: AtomicU64,
    pub keyframe_dropped: AtomicU64,
    pub handoff_ns: AtomicU64,
    pub decoder_resets: AtomicU64,
}
#[derive(Default)]
struct LossGate {
    started: bool,
    damaged: bool,
}
impl LossGate {
    fn admit(&mut self, key: bool, loss: u16) -> Option<bool> {
        self.damaged |= loss > 0;
        if (!self.started || self.damaged) && !key {
            return None;
        }
        self.started = true;
        Some(std::mem::take(&mut self.damaged))
    }
}
#[derive(Clone, PartialEq)]
struct Parameters {
    codec: &'static str,
    extra: Vec<u8>,
    clock: u32,
    fps: f64,
}
struct Packet {
    parameters: Arc<Parameters>,
    data: Vec<u8>,
    pts: i64,
    key: bool,
    resync: bool,
}

struct Login {
    username: String,
    password: String,
}
impl std::fmt::Debug for Login {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Login")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}
fn authenticated_url(source: &str) -> Result<(url::Url, Option<Login>)> {
    let mut url = url::Url::parse(source).context("Invalid RTSP URL")?;
    ensure!(
        url.scheme() == "rtsp",
        "Rust ingress supports unencrypted RTSP over TCP"
    );
    let creds = if !url.username().is_empty() || url.password().is_some() {
        Some(Login {
            username: percent_encoding::percent_decode_str(url.username())
                .decode_utf8()?
                .into_owned(),
            password: percent_encoding::percent_decode_str(url.password().unwrap_or(""))
                .decode_utf8()?
                .into_owned(),
        })
    } else {
        None
    };
    url.set_username("")
        .map_err(|_| anyhow::anyhow!("Invalid RTSP authority"))?;
    url.set_password(None)
        .map_err(|_| anyhow::anyhow!("Invalid RTSP authority"))?;
    Ok((url, creds))
}

async fn receive(
    source: String,
    timeout: Duration,
    stop: CancellationToken,
    tx: &mpsc::Sender<Result<Packet, String>>,
    stats: Arc<Stats>,
) -> Result<()> {
    let (url, creds) = authenticated_url(&source)?;
    let connect = async {
        // TCP sockets close with the session; do not leave unbounded teardown
        // retries after cancellation. Vendor-specific/UDP/TLS paths stay libav.
        let options = SessionOptions::default()
            .creds(creds.map(|login| Credentials {
                username: login.username,
                password: login.password,
            }))
            .teardown(TeardownPolicy::Never);
        let mut session = Session::describe(url, options).await?;
        let (id, codec, clock) = session
            .streams()
            .iter()
            .enumerate()
            .find_map(|(i, s)| {
                let codec = if s.encoding_name().eq_ignore_ascii_case("h264") {
                    "h264"
                } else if s.encoding_name().eq_ignore_ascii_case("h265") {
                    "hevc"
                } else {
                    return None;
                };
                Some((i, codec, s.clock_rate_hz()))
            })
            .ok_or_else(|| anyhow::anyhow!("RTSP has no supported H.264/H.265 video track"))?;
        session
            .setup(
                id,
                SetupOptions::default().transport(Transport::Tcp(Default::default())),
            )
            .await?;
        let demux = session.play(Default::default()).await?.demuxed()?;
        Ok::<_, anyhow::Error>((id, codec, clock, demux))
    };
    let (id, codec, clock, mut demux) = tokio::select! {
        _=stop.cancelled()=>return Ok(()),
        result=tokio::time::timeout(timeout,connect)=>result.context("RTSP handshake timed out")??,
    };
    let mut parameters: Option<Arc<Parameters>> = None;
    let mut gate = LossGate::default();
    let mut deadline = tokio::time::Instant::now() + timeout;
    loop {
        let item = tokio::select! {
            _=stop.cancelled()=>return Ok(()),
            result=tokio::time::timeout_at(deadline,demux.next())=>result.context("RTSP video timed out")?,
        };
        let Some(item) = item else { return Ok(()) };
        let CodecItem::VideoFrame(frame) = item? else {
            continue;
        };
        if frame.stream_id() != id {
            continue;
        }
        deadline = tokio::time::Instant::now() + timeout;
        stats.frames.fetch_add(1, Ordering::Relaxed);
        stats
            .bytes
            .fetch_add(frame.data().len() as u64, Ordering::Relaxed);
        stats
            .lost_packets
            .fetch_add(u64::from(frame.loss()), Ordering::Relaxed);
        let key = frame.is_random_access_point();
        let Some(resync) = gate.admit(key, frame.loss()) else {
            stats.keyframe_dropped.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        if parameters.is_none() || frame.has_new_parameters() {
            let Some(ParametersRef::Video(p)) = demux.streams()[id].parameters() else {
                bail!("Missing RTSP codec parameters");
            };
            let fps = p
                .frame_rate()
                .filter(|(n, d)| *n > 0 && *d > 0)
                .map(|(n, d)| f64::from(d) / f64::from(n))
                .unwrap_or(0.);
            let new = Parameters {
                codec,
                extra: p.extra_data().to_vec(),
                clock,
                fps,
            };
            if parameters.as_ref().is_some_and(|old| old.as_ref() != &new) {
                bail!("RTSP codec parameters changed; reconnect required");
            }
            parameters = Some(Arc::new(new));
        }
        let pts = frame.timestamp().elapsed();
        let packet = Packet {
            parameters: parameters.as_ref().expect("codec parameters").clone(),
            pts,
            key,
            resync,
            data: frame.into_data(),
        };
        let began = Instant::now();
        // Bounded access-unit queue. Never drop arbitrary reference pictures
        // when the decoder is busy; capture's PTS backlog policy can reconnect.
        tokio::select! {
            _=stop.cancelled()=>return Ok(()),
            result=tx.send(Ok(packet))=>if result.is_err(){return Ok(())},
        }
        stats
            .handoff_ns
            .fetch_add(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

pub struct RustSource {
    receiver: mpsc::Receiver<Result<Packet, String>>,
    decoder: Option<PacketDecoder>,
    pending: Option<Packet>,
    parameters: Option<Arc<Parameters>>,
    stop: CancellationToken,
    abort: AbortHandle,
    metrics: Arc<GpuMetrics>,
    stats: Arc<Stats>,
    gpu: i32,
    eof: bool,
    flushed: bool,
    discontinuity: bool,
}
impl RustSource {
    fn open(
        handle: &Handle,
        source: &str,
        gpu: i32,
        stop: CancellationToken,
        timeout: Duration,
        metrics: Arc<GpuMetrics>,
        stats: Arc<Stats>,
    ) -> Result<Self> {
        authenticated_url(source)?;
        let stop = stop.child_token();
        let worker_stop = stop.clone();
        let (tx, receiver) = mpsc::channel(2);
        let address = source.to_owned();
        let counters = stats.clone();
        let task = handle.spawn(async move {
            if let Err(error) = receive(address, timeout, worker_stop.clone(), &tx, counters).await
            {
                tokio::select! {_=worker_stop.cancelled()=>(),_=tx.send(Err(error.to_string()))=>()}
            }
        });
        Ok(Self {
            receiver,
            decoder: None,
            pending: None,
            parameters: None,
            stop,
            abort: task.abort_handle(),
            metrics,
            stats,
            gpu,
            eof: false,
            flushed: false,
            discontinuity: false,
        })
    }
    fn next_frame(&mut self) -> Result<Option<Decoded>> {
        loop {
            if let Some(decoder) = &mut self.decoder {
                match decoder.receive()? {
                    PacketOutput::Frame(frame) => return Ok(Some(frame)),
                    PacketOutput::End => return Ok(None),
                    PacketOutput::NeedInput => (),
                }
            }
            if let Some(packet) = &self.pending {
                if self.decoder.as_mut().expect("packet decoder").send(
                    &packet.data,
                    packet.pts,
                    packet.key,
                )? {
                    self.pending = None;
                }
                continue;
            }
            if self.eof {
                if self.flushed || self.decoder.is_none() {
                    return Ok(None);
                }
                self.flushed = self.decoder.as_mut().unwrap().flush()?;
                continue;
            }
            let Some(packet) = self.receiver.blocking_recv() else {
                self.eof = true;
                continue;
            };
            let packet = packet.map_err(anyhow::Error::msg)?;
            if self.decoder.is_none() || packet.resync {
                self.decoder = None;
                let p = &packet.parameters;
                self.decoder = Some(PacketDecoder::open(
                    p.codec,
                    &p.extra,
                    p.clock,
                    p.fps,
                    self.gpu,
                    self.metrics.clone(),
                )?);
                self.parameters = Some(packet.parameters.clone());
                if packet.resync {
                    self.discontinuity = true;
                    self.stats.decoder_resets.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.pending = Some(packet);
        }
    }
}
impl Drop for RustSource {
    fn drop(&mut self) {
        self.stop.cancel();
        self.abort.abort();
    }
}

pub enum Source {
    Libav(Decoder),
    Rust(RustSource),
}
impl Source {
    pub fn open(
        handle: &Handle,
        rust_rtsp: bool,
        source: &str,
        gpu: i32,
        stop: CancellationToken,
        timeout: Duration,
        metrics: Arc<GpuMetrics>,
        stats: Arc<Stats>,
    ) -> Result<Self> {
        if rust_rtsp && source.starts_with("rtsp://") {
            Ok(Self::Rust(RustSource::open(
                handle, source, gpu, stop, timeout, metrics, stats,
            )?))
        } else {
            Ok(Self::Libav(Decoder::open_device(
                source, gpu, stop, timeout, metrics,
            )?))
        }
    }
    pub fn tap_packets(&mut self, relay: crate::relay::Sender) {
        if let Self::Libav(d) = self {
            relay.invalidate();
            let active = relay.gate.clone();
            d.tap_packets(
                move || active.active(),
                move |p| match p {
                    Some(p) => relay.offer(p),
                    None => relay.gap(),
                },
            );
        }
    }
    pub fn next_frame(&mut self) -> Result<Option<Decoded>> {
        match self {
            Self::Libav(d) => d.next_frame(),
            Self::Rust(d) => d.next_frame(),
        }
    }
    pub fn take_discontinuity(&mut self) -> bool {
        match self {
            Self::Rust(d) => std::mem::take(&mut d.discontinuity),
            _ => false,
        }
    }
    pub fn bgr(&mut self, w: u32, h: u32, p: &mut [u8]) -> Result<()> {
        match self {
            Self::Libav(d) => d.bgr(w, h, p),
            Self::Rust(d) => d.decoder.as_mut().context("No decoded frame")?.bgr(w, h, p),
        }
    }
    pub fn retain_nv12(&mut self) -> Result<Arc<Nv12Frame>> {
        match self {
            Self::Libav(d) => d.retain_nv12(),
            Self::Rust(d) => d
                .decoder
                .as_mut()
                .context("No decoded frame")?
                .retain_nv12(),
        }
    }
    pub fn retain_host(&mut self, w: u32, h: u32) -> Result<Arc<construction_media::HostFrame>> {
        match self {
            Self::Libav(d) => d.retain_host(w, h),
            Self::Rust(d) => d
                .decoder
                .as_mut()
                .context("No decoded frame")?
                .retain_host(w, h),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packet_loss_waits_for_random_access_and_resets_once() {
        let mut gate = LossGate::default();
        assert_eq!(gate.admit(false, 0), None);
        assert_eq!(gate.admit(true, 0), Some(false));
        assert_eq!(gate.admit(false, 0), Some(false));
        assert_eq!(gate.admit(false, 2), None);
        assert_eq!(gate.admit(false, 0), None);
        assert_eq!(gate.admit(true, 0), Some(true));
        assert_eq!(gate.admit(false, 0), Some(false));
    }
    #[test]
    fn credentials_are_decoded_and_removed_from_the_wire_url() {
        let (url, creds) = authenticated_url("rtsp://u%40x:p%3Ass@127.0.0.1/source").unwrap();
        assert_eq!(url.as_str(), "rtsp://127.0.0.1/source");
        let creds = creds.unwrap();
        assert_eq!(creds.username, "u@x");
        assert_eq!(creds.password, "p:ss");
        assert!(!format!("{creds:?}").contains("p:ss"));
    }
}

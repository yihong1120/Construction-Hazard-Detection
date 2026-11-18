//! Exclusive RAII handles to native codecs. No borrowed frame crosses this ABI.
use anyhow::{Result, ensure};
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_int, c_void},
    ptr::NonNull,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;

unsafe extern "C" {
    fn cm_init() -> c_int;
    fn cm_error() -> *const c_char;
    fn cm_decoder_open(
        source: *const c_char,
        gpu: c_int,
        interrupt: extern "C" fn(*mut c_void) -> c_int,
        opaque: *mut c_void,
    ) -> *mut c_void;
    fn cm_decoder_next(pointer: *mut c_void, pts: *mut f64, fps: *mut f64) -> c_int;
    fn cm_packets_open(
        name: *const c_char,
        extra: *const u8,
        length: usize,
        clock_rate: c_int,
        fps: f64,
        gpu: c_int,
    ) -> *mut c_void;
    fn cm_packets_send(
        pointer: *mut c_void,
        bytes: *const u8,
        length: usize,
        pts: i64,
        key: c_int,
    ) -> c_int;
    fn cm_packets_receive(pointer: *mut c_void, pts: *mut f64, fps: *mut f64) -> c_int;
    fn cm_packets_flush(pointer: *mut c_void) -> c_int;
    fn cm_decoder_bgr(
        pointer: *mut c_void,
        target: *mut u8,
        length: usize,
        width: c_int,
        height: c_int,
    ) -> c_int;
    fn cm_decoder_free(pointer: *mut c_void);
    fn cm_decoder_retain_cuda(pointer: *mut c_void, info: *mut Nv12Info) -> *mut c_void;
    fn cm_frame_free(pointer: *mut c_void);
    fn cm_frame_bgr(pointer: *mut c_void, target: *mut u8, length: usize) -> c_int;
    fn cm_lut_create(gpu: i32, space: i32, range: i32, address: *mut usize) -> *mut c_void;
    fn cm_lut_free(pointer: *mut c_void);
    fn cm_encoder_open(
        target: *const c_char,
        width: c_int,
        height: c_int,
        fps: c_int,
        gpu: c_int,
        interrupt: extern "C" fn(*mut c_void) -> c_int,
        opaque: *mut c_void,
    ) -> *mut c_void;
    fn cm_encoder_packets_written(pointer: *mut c_void) -> u64;
    fn cm_encoder_write(pointer: *mut c_void, data: *const u8, length: usize) -> c_int;
    fn cm_encoder_repeat_frame(pointer: *mut c_void) -> c_int;
    fn cm_encoder_write_scaled(
        pointer: *mut c_void,
        data: *const u8,
        length: usize,
        width: c_int,
        height: c_int,
    ) -> c_int;
    fn cm_encoder_finish(pointer: *mut c_void) -> c_int;
    fn cm_encoder_free(pointer: *mut c_void);
}
mod host;
mod remux;
mod scale;
pub use host::HostFrame;
pub use remux::{EncodedPacket, Remuxer};
pub use scale::BgrScaler;

static DECODERS: AtomicUsize = AtomicUsize::new(0);
static ENCODERS: AtomicUsize = AtomicUsize::new(0);
pub fn live_handles() -> (usize, usize) {
    (
        DECODERS.load(Ordering::Relaxed),
        ENCODERS.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod packet_tests {
    use super::*;
    #[test]
    fn empty_packet_decoder_flushes_and_releases_its_handle() {
        let before = live_handles();
        {
            let mut decoder =
                PacketDecoder::open("h264", &[], 90000, 0., -1, Arc::default()).unwrap();
            assert!(matches!(
                decoder.receive().unwrap(),
                PacketOutput::NeedInput
            ));
            assert!(decoder.send(&[], 0, true).is_err());
            assert!(decoder.flush().unwrap());
            assert!(matches!(decoder.receive().unwrap(), PacketOutput::End));
        }
        assert_eq!(live_handles(), before);
    }
}
fn error() -> anyhow::Error {
    // SAFETY: the ABI returns a thread-local, terminated error until next call.
    anyhow::anyhow!(
        unsafe { CStr::from_ptr(cm_error()) }
            .to_string_lossy()
            .into_owned()
    )
}
fn initialize() -> Result<()> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        if unsafe { cm_init() } < 0 {
            Err(error().to_string())
        } else {
            Ok(())
        }
    })
    .as_ref()
    .map_err(|message| anyhow::anyhow!(message.clone()))?;
    Ok(())
}
struct Interrupt {
    born: Instant,
    deadline: AtomicU64,
    stop: CancellationToken,
    timeout: Duration,
}
impl Interrupt {
    fn new(stop: CancellationToken, timeout: Duration) -> Box<Self> {
        let value = Box::new(Self {
            born: Instant::now(),
            deadline: AtomicU64::new(0),
            stop,
            timeout,
        });
        value.reset();
        value
    }
    fn reset(&self) {
        self.deadline.store(
            (self.born.elapsed() + self.timeout)
                .as_millis()
                .min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }
    fn pointer(&mut self) -> *mut c_void {
        (self as *mut Self).cast()
    }
}
extern "C" fn interrupted(opaque: *mut c_void) -> c_int {
    // SAFETY: a boxed Interrupt outlives the native handle and is never moved.
    let state = unsafe { &*opaque.cast::<Interrupt>() };
    i32::from(
        state.stop.is_cancelled()
            || state.born.elapsed().as_millis() >= state.deadline.load(Ordering::Relaxed) as u128,
    )
}
pub struct Decoder {
    pointer: NonNull<c_void>,
    interrupt: Box<Interrupt>,
    metrics: Arc<GpuMetrics>,
    profile: Option<Arc<ColorTable>>,
    host_profile: Option<Arc<host::HostTable>>,
    tap: Option<Box<remux::PacketTap>>,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Nv12Info {
    pub y: usize,
    pub uv: usize,
    pub ready: usize,
    pub lut: usize,
    pub width: i32,
    pub height: i32,
    pub yp: i32,
    pub uvp: i32,
    pub colorspace: i32,
    pub range: i32,
    pub device: i32,
}
#[derive(Default)]
pub struct GpuMetrics {
    pub host_retained: AtomicU64,
    pub host_materialized: AtomicU64,
    pub host_sampled: AtomicU64,
    pub materialized: AtomicU64,
    pub download_payload_bytes: AtomicU64,
}
/// Immutable AVFrame reference: decoder advancement/drop cannot recycle its
/// storage. The ready event belongs to the same retained CUDA context.
pub struct Nv12Frame {
    pointer: NonNull<c_void>,
    info: Nv12Info,
    cpu: OnceLock<Result<Vec<u8>, String>>,
    metrics: Arc<GpuMetrics>,
    profile: Option<Arc<ColorTable>>,
}
struct ColorTable {
    pointer: NonNull<c_void>,
    address: usize,
    key: (i32, i32, i32),
}
unsafe impl Send for ColorTable {}
unsafe impl Sync for ColorTable {}
impl Drop for ColorTable {
    fn drop(&mut self) {
        unsafe { cm_lut_free(self.pointer.as_ptr()) };
    }
}
fn color_table(info: Nv12Info) -> Result<Arc<ColorTable>> {
    ensure!(
        matches!(info.colorspace, 1 | 2 | 5 | 6) && matches!(info.range, 0..=2),
        "NV12 calibration supports BT.601/709, limited/full range only"
    );
    let key = (
        info.device,
        if info.colorspace == 1 { 1 } else { 5 },
        if info.range == 2 { 2 } else { 1 },
    );
    type Tables = Mutex<HashMap<(i32, i32, i32), Weak<ColorTable>>>;
    static TABLES: OnceLock<Tables> = OnceLock::new();
    let mut cache = TABLES
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(table) = cache.get(&key).and_then(Weak::upgrade) {
        return Ok(table);
    }
    let mut address = 0;
    let pointer = NonNull::new(unsafe { cm_lut_create(key.0, key.1, key.2, &mut address) })
        .ok_or_else(error)?;
    let table = Arc::new(ColorTable {
        pointer,
        address,
        key,
    });
    cache.insert(key, Arc::downgrade(&table));
    Ok(table)
}
// Native frame data is immutable; the only mutable CPU operation runs once.
// FFmpeg's referenced device/frame contexts and CUDA event outlive this handle.
unsafe impl Send for Nv12Frame {}
unsafe impl Sync for Nv12Frame {}
impl Nv12Frame {
    pub fn info(&self) -> Nv12Info {
        self.info
    }
    pub fn bgr_padded(&self) -> Result<&[u8]> {
        self.cpu
            .get_or_init(|| {
                let bytes = self.info.width as usize * self.info.height as usize * 3;
                let mut pixels = vec![0; bytes + 64];
                if unsafe { cm_frame_bgr(self.pointer.as_ptr(), pixels.as_mut_ptr(), pixels.len()) }
                    < 0
                {
                    return Err(error().to_string());
                }
                self.metrics.materialized.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .download_payload_bytes
                    .fetch_add((bytes / 2) as u64, Ordering::Relaxed);
                Ok(pixels)
            })
            .as_ref()
            .map(Vec::as_slice)
            .map_err(|e| anyhow::anyhow!(e.clone()))
    }
}
impl Drop for Nv12Frame {
    fn drop(&mut self) {
        unsafe { cm_frame_free(self.pointer.as_ptr()) };
    }
}
// A handle can be moved but every operation requires exclusive &mut access.
unsafe impl Send for Decoder {}
#[derive(Clone, Copy)]
pub struct Decoded {
    pub pts: Option<f64>,
    pub fps: f64,
}
pub enum PacketOutput {
    Frame(Decoded),
    NeedInput,
    End,
}
/// Codec-only owner. Network protocols and packet lifetimes belong to its
/// caller; the bridge copies each submitted packet into padded AVPacket data.
pub struct PacketDecoder {
    decoder: Decoder,
}
impl PacketDecoder {
    pub fn open(
        codec: &str,
        extra: &[u8],
        clock_rate: u32,
        fps: f64,
        gpu: i32,
        metrics: Arc<GpuMetrics>,
    ) -> Result<Self> {
        initialize()?;
        ensure!(
            matches!(codec, "h264" | "hevc") && clock_rate > 0 && clock_rate <= i32::MAX as u32,
            "Unsupported packet decoder configuration"
        );
        let codec = CString::new(codec)?;
        let pointer = NonNull::new(unsafe {
            cm_packets_open(
                codec.as_ptr(),
                extra.as_ptr(),
                extra.len(),
                clock_rate as i32,
                fps,
                gpu,
            )
        })
        .ok_or_else(error)?;
        DECODERS.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            decoder: Decoder {
                pointer,
                interrupt: Interrupt::new(CancellationToken::new(), Duration::from_secs(10)),
                metrics,
                profile: None,
                host_profile: None,
                tap: None,
            },
        })
    }
    /// False means drain output and retry this same packet, not discard it.
    pub fn send(&mut self, bytes: &[u8], pts: i64, key: bool) -> Result<bool> {
        let status = unsafe {
            cm_packets_send(
                self.decoder.pointer.as_ptr(),
                bytes.as_ptr(),
                bytes.len(),
                pts,
                i32::from(key),
            )
        };
        if status < 0 {
            return Err(error());
        }
        Ok(status == 0)
    }
    pub fn receive(&mut self) -> Result<PacketOutput> {
        let (mut pts, mut fps) = (f64::NAN, 0.);
        match unsafe { cm_packets_receive(self.decoder.pointer.as_ptr(), &mut pts, &mut fps) } {
            0 => Ok(PacketOutput::NeedInput),
            1 => Ok(PacketOutput::Frame(Decoded {
                pts: pts.is_finite().then_some(pts),
                fps,
            })),
            2 => Ok(PacketOutput::End),
            _ => Err(error()),
        }
    }
    pub fn flush(&mut self) -> Result<bool> {
        let status = unsafe { cm_packets_flush(self.decoder.pointer.as_ptr()) };
        if status < 0 {
            return Err(error());
        }
        Ok(status == 0)
    }
    pub fn bgr(&mut self, width: u32, height: u32, padded: &mut [u8]) -> Result<()> {
        self.decoder.bgr(width, height, padded)
    }
    pub fn retain_host(&mut self, w: u32, h: u32) -> Result<Arc<HostFrame>> {
        self.decoder.retain_host(w, h)
    }
    pub fn retain_nv12(&mut self) -> Result<Arc<Nv12Frame>> {
        self.decoder.retain_nv12()
    }
}
impl Decoder {
    pub fn open(source: &str, stop: CancellationToken, timeout: Duration) -> Result<Self> {
        Self::open_device(source, -1, stop, timeout, Arc::default())
    }
    pub fn open_device(
        source: &str,
        gpu: i32,
        stop: CancellationToken,
        timeout: Duration,
        metrics: Arc<GpuMetrics>,
    ) -> Result<Self> {
        initialize()?;
        let source = CString::new(source)?;
        let mut interrupt = Interrupt::new(stop, timeout);
        let pointer = NonNull::new(unsafe {
            cm_decoder_open(source.as_ptr(), gpu, interrupted, interrupt.pointer())
        })
        .ok_or_else(error)?;
        DECODERS.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            pointer,
            interrupt,
            metrics,
            profile: None,
            host_profile: None,
            tap: None,
        })
    }
    pub fn retain_nv12(&mut self) -> Result<Arc<Nv12Frame>> {
        let mut info = Nv12Info::default();
        let pointer =
            NonNull::new(unsafe { cm_decoder_retain_cuda(self.pointer.as_ptr(), &mut info) })
                .ok_or_else(error)?;
        let mut frame = Nv12Frame {
            pointer,
            info,
            cpu: OnceLock::new(),
            metrics: self.metrics.clone(),
            profile: None,
        };
        // Other layouts/profiles use the retained-frame CPU fallback. Do not
        // assume the calibrated SIMD conversion covers an untested row tail.
        if info.width % 32 != 0
            || !matches!(info.colorspace, 1 | 2 | 5 | 6)
            || !matches!(info.range, 0..=2)
        {
            return Ok(Arc::new(frame));
        }
        let key = (
            info.device,
            if info.colorspace == 1 { 1 } else { 5 },
            if info.range == 2 { 2 } else { 1 },
        );
        let profile = match &self.profile {
            Some(profile) if profile.key == key => profile.clone(),
            _ => color_table(info)?,
        };
        frame.info.lut = profile.address;
        frame.profile = Some(profile.clone());
        self.profile = Some(profile);
        Ok(Arc::new(frame))
    }
    pub fn next_frame(&mut self) -> Result<Option<Decoded>> {
        self.interrupt.reset();
        let (mut pts, mut fps) = (f64::NAN, 0.);
        let status = unsafe { cm_decoder_next(self.pointer.as_ptr(), &mut pts, &mut fps) };
        if status < 0 {
            return Err(error());
        }
        Ok((status == 1).then_some(Decoded {
            pts: pts.is_finite().then_some(pts),
            fps,
        }))
    }
    pub fn bgr(&mut self, width: u32, height: u32, padded: &mut [u8]) -> Result<()> {
        ensure!(
            width > 0 && height > 0 && width <= 16384 && height <= 16384,
            "Invalid BGR shape"
        );
        ensure!(
            padded.len() >= width as usize * height as usize * 3 + 64,
            "BGR buffer requires 64 byte tail padding"
        );
        if unsafe {
            cm_decoder_bgr(
                self.pointer.as_ptr(),
                padded.as_mut_ptr(),
                padded.len(),
                width as i32,
                height as i32,
            )
        } < 0
        {
            return Err(error());
        }
        Ok(())
    }
}
impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { cm_decoder_free(self.pointer.as_ptr()) };
        DECODERS.fetch_sub(1, Ordering::Relaxed);
    }
}
pub struct Encoder {
    pointer: NonNull<c_void>,
    interrupt: Box<Interrupt>,
    bytes: usize,
}
unsafe impl Send for Encoder {}
impl Encoder {
    pub fn open(target: &str, width: u32, height: u32, fps: u32, gpu: Option<i32>) -> Result<Self> {
        initialize()?;
        ensure!(
            width > 0
                && height > 0
                && width % 2 == 0
                && height % 2 == 0
                && width <= 16384
                && height <= 16384
                && (1..=120).contains(&fps),
            "Invalid encoder shape or FPS"
        );
        let target = CString::new(target)?;
        // Shutdown stops accepting frames, but gives native trailer I/O its own
        // bounded deadline; cancellation must not corrupt a local MP4 trailer.
        let mut interrupt = Interrupt::new(CancellationToken::new(), Duration::from_secs(5));
        let pointer = NonNull::new(unsafe {
            cm_encoder_open(
                target.as_ptr(),
                width as i32,
                height as i32,
                fps as i32,
                gpu.unwrap_or(-1),
                interrupted,
                interrupt.pointer(),
            )
        })
        .ok_or_else(error)?;
        ENCODERS.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            pointer,
            interrupt,
            bytes: width as usize * height as usize * 3,
        })
    }
    pub fn write(&mut self, padded: &[u8]) -> Result<()> {
        ensure!(
            padded.len() == self.bytes + 64,
            "Encoder BGR requires 64 byte tail padding"
        );
        self.interrupt.reset();
        if unsafe { cm_encoder_write(self.pointer.as_ptr(), padded.as_ptr(), padded.len()) } < 0 {
            return Err(error());
        }
        Ok(())
    }
    /// Packets successfully submitted to the muxer, excluding queued inputs.
    pub fn packets_written(&self) -> u64 {
        unsafe { cm_encoder_packets_written(self.pointer.as_ptr()) }
    }
    /// Encode the last submitted pixels at the next timestamp without rescaling.
    pub fn repeat_frame(&mut self) -> Result<()> {
        self.interrupt.reset();
        if unsafe { cm_encoder_repeat_frame(self.pointer.as_ptr()) } < 0 {
            return Err(error());
        }
        Ok(())
    }
    /// Resize and BGR-to-encoder colour conversion in the same cached scaler.
    pub fn write_scaled(&mut self, padded: &[u8], width: u32, height: u32) -> Result<()> {
        ensure!(
            width > 0
                && height > 0
                && u64::from(width) * u64::from(height) <= 64 * 1024 * 1024
                && padded.len() == width as usize * height as usize * 3 + 64,
            "Invalid source encoder frame"
        );
        self.interrupt.reset();
        if unsafe {
            cm_encoder_write_scaled(
                self.pointer.as_ptr(),
                padded.as_ptr(),
                padded.len(),
                width as i32,
                height as i32,
            )
        } < 0
        {
            return Err(error());
        }
        Ok(())
    }
    pub fn finish(&mut self) -> Result<()> {
        self.interrupt.reset();
        if unsafe { cm_encoder_finish(self.pointer.as_ptr()) } < 0 {
            return Err(error());
        }
        Ok(())
    }
}
impl Drop for Encoder {
    fn drop(&mut self) {
        // RTSP sockets belong to the muxer's trailer, not free_context.
        // Even an error/early return must close them before releasing a lease.
        self.interrupt.reset();
        unsafe { cm_encoder_finish(self.pointer.as_ptr()) };
        unsafe { cm_encoder_free(self.pointer.as_ptr()) };
        ENCODERS.fetch_sub(1, Ordering::Relaxed);
    }
}

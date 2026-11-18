//! Referenced CPU surfaces, with full BGR materialized only on demand.
use super::*;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Info {
    y: usize,
    u: usize,
    v: usize,
    width: i32,
    height: i32,
    yp: i32,
    up: i32,
    vp: i32,
    colorspace: i32,
    range: i32,
    planar420: i32,
}
unsafe extern "C" {
    fn cm_decoder_retain_host(pointer: *mut c_void, info: *mut Info) -> *mut c_void;
    fn cm_host_free(pointer: *mut c_void);
    fn cm_host_bgr(
        pointer: *mut c_void,
        target: *mut u8,
        length: usize,
        width: i32,
        height: i32,
    ) -> i32;
    fn cm_cpu_lut_fill(target: *mut u8, length: usize, space: i32, range: i32) -> i32;
}
pub(crate) struct HostTable {
    key: (i32, i32),
    // Green depends on all three components; blue only on U/Y, red on V/Y.
    // Validate this against the calibrated reference instead of approximating
    // libswscale coefficients or rounding.
    full: Option<Vec<u8>>,
    green: Vec<u8>,
    blue: Vec<u8>,
    red: Vec<u8>,
}
impl HostTable {
    fn compact(key: (i32, i32), bytes: Vec<u8>) -> Result<Self> {
        let mut green = vec![0; 256 * 256 * 256];
        let mut blue = vec![0; 256 * 256];
        let mut red = vec![0; 256 * 256];
        for u in 0..256 {
            for v in 0..256 {
                for y in 0..256 {
                    let i = (u * 256 + v) * 256 + y;
                    let p = &bytes[i * 3..i * 3 + 3];
                    if v == 0 {
                        blue[u * 256 + y] = p[0];
                    }
                    if u == 0 {
                        red[v * 256 + y] = p[2];
                    }
                    if blue[u * 256 + y] != p[0] || red[v * 256 + y] != p[2] {
                        // Other libswscale builds may use a nonseparable
                        // conversion. Keep their exact calibrated reference.
                        return Ok(Self {
                            key,
                            full: Some(bytes),
                            green: vec![],
                            blue: vec![],
                            red: vec![],
                        });
                    }
                    green[i] = p[1];
                }
            }
        }
        Ok(Self {
            key,
            full: None,
            green,
            blue,
            red,
        })
    }
    fn pixel(&self, y: usize, u: usize, v: usize) -> [u8; 3] {
        if let Some(bytes) = &self.full {
            let i = ((u * 256 + v) * 256 + y) * 3;
            return [bytes[i], bytes[i + 1], bytes[i + 2]];
        }
        [
            self.blue[u * 256 + y],
            self.green[(u * 256 + v) * 256 + y],
            self.red[v * 256 + y],
        ]
    }
}
pub struct HostFrame {
    pointer: NonNull<c_void>,
    info: Info,
    width: u32,
    height: u32,
    table: Option<Arc<HostTable>>,
    cpu: OnceLock<Result<Vec<u8>, String>>,
    metrics: Arc<GpuMetrics>,
}
// av_frame_clone retains immutable reference-counted plane buffers. Only the
// per-owner conversion cache mutates, once; decoder reuse cannot recycle them.
unsafe impl Send for HostFrame {}
unsafe impl Sync for HostFrame {}
impl Drop for HostFrame {
    fn drop(&mut self) {
        unsafe { cm_host_free(self.pointer.as_ptr()) };
    }
}
impl HostFrame {
    pub(super) fn as_ptr(&self) -> *const c_void {
        self.pointer.as_ptr()
    }
    pub fn lookup_bytes(&self) -> usize {
        self.table.as_ref().map_or(0, |t| {
            t.green.len() + t.blue.len() + t.red.len() + t.full.as_ref().map_or(0, Vec::len)
        })
    }
    pub fn bgr_padded(&self) -> Result<&[u8]> {
        self.cpu
            .get_or_init(|| {
                let mut bytes = vec![0; self.width as usize * self.height as usize * 3 + 64];
                if unsafe {
                    cm_host_bgr(
                        self.pointer.as_ptr(),
                        bytes.as_mut_ptr(),
                        bytes.len(),
                        self.width as i32,
                        self.height as i32,
                    )
                } < 0
                {
                    return Err(error().to_string());
                }
                self.metrics
                    .host_materialized
                    .fetch_add(1, Ordering::Relaxed);
                Ok(bytes)
            })
            .as_ref()
            .map(Vec::as_slice)
            .map_err(|e| anyhow::anyhow!(e.clone()))
    }
    /// Exact odd-integer INTER_LINEAR sampling of the calibrated full BGR
    /// conversion, without constructing full BGR. Unsupported layouts leave
    /// the output untouched and use the existing numerical reference path.
    pub fn letterbox(
        &self,
        target: &mut [u8],
        h: usize,
        w: usize,
        rh: usize,
        rw: usize,
        top: usize,
        left: usize,
    ) -> bool {
        let Some(table) = &self.table else {
            return false;
        };
        let (sh, sw) = (self.height as usize, self.width as usize);
        if rh == 0
            || rw == 0
            || sh % rh != 0
            || sw % rw != 0
            || sh / rh != sw / rw
            || (sw / rw) % 2 != 1
            || h.checked_mul(w).and_then(|n| n.checked_mul(3)) != Some(target.len())
            || top.checked_add(rh).is_none_or(|v| v > h)
            || left.checked_add(rw).is_none_or(|v| v > w)
        {
            return false;
        }
        let scale = sw / rw;
        let p = self.info;
        // SAFETY: the retained AVFrame owns the positive-stride plane ranges;
        // eligibility below checks native geometry and all plane widths.
        let (y, u, v) = unsafe {
            (
                std::slice::from_raw_parts(p.y as *const u8, p.yp as usize * sh),
                std::slice::from_raw_parts(p.u as *const u8, p.up as usize * (sh / 2)),
                std::slice::from_raw_parts(p.v as *const u8, p.vp as usize * (sh / 2)),
            )
        };
        // Content is completely overwritten below. Initialize only letterbox bands.
        target[..top * w * 3].fill(114);
        target[(top + rh) * w * 3..].fill(114);
        for row in top..top + rh {
            target[row * w * 3..(row * w + left) * 3].fill(114);
            target[(row * w + left + rw) * 3..(row + 1) * w * 3].fill(114);
        }
        for row in 0..rh {
            let sy = row * scale + scale / 2;
            let dst = &mut target[((top + row) * w + left) * 3..((top + row) * w + left + rw) * 3];
            for (x, pixel) in dst.chunks_exact_mut(3).enumerate() {
                let sx = x * scale + scale / 2;
                let yy = y[sy * p.yp as usize + sx] as usize;
                let uu = u[(sy / 2) * p.up as usize + sx / 2] as usize;
                let vv = v[(sy / 2) * p.vp as usize + sx / 2] as usize;
                pixel.copy_from_slice(&table.pixel(yy, uu, vv));
            }
        }
        self.metrics.host_sampled.fetch_add(1, Ordering::Relaxed);
        true
    }
}
impl Decoder {
    pub fn retain_host(&mut self, width: u32, height: u32) -> Result<Arc<HostFrame>> {
        ensure!(
            width > 0 && height > 0 && u64::from(width) * u64::from(height) <= 64 * 1024 * 1024,
            "Invalid host frame dimensions"
        );
        let mut info = Info::default();
        let pointer =
            NonNull::new(unsafe { cm_decoder_retain_host(self.pointer.as_ptr(), &mut info) })
                .ok_or_else(error)?;
        let mut frame = HostFrame {
            pointer,
            info,
            width,
            height,
            table: None,
            cpu: OnceLock::new(),
            metrics: self.metrics.clone(),
        };
        if info.planar420 != 0
            && info.width == width as i32
            && info.height == height as i32
            && width % 32 == 0
            && height % 2 == 0
            && info.yp >= width as i32
            && info.up >= width as i32 / 2
            && info.vp >= width as i32 / 2
            && info.y != 0
            && info.u != 0
            && info.v != 0
            && matches!(info.colorspace, 1 | 2 | 5 | 6)
            && matches!(info.range, 0..=2)
        {
            let key = (
                if info.colorspace == 1 { 1 } else { 5 },
                if info.range == 2 { 2 } else { 1 },
            );
            let table = if let Some(table) = self.host_profile.as_ref().filter(|t| t.key == key) {
                table.clone()
            } else {
                type Tables = Mutex<HashMap<(i32, i32), Weak<HostTable>>>;
                static TABLES: OnceLock<Tables> = OnceLock::new();
                let mut cache = TABLES
                    .get_or_init(Mutex::default)
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let table = if let Some(table) = cache.get(&key).and_then(Weak::upgrade) {
                    table
                } else {
                    let mut bytes = vec![0; 256 * 256 * 256 * 3];
                    if unsafe { cm_cpu_lut_fill(bytes.as_mut_ptr(), bytes.len(), key.0, key.1) } < 0
                    {
                        return Err(error());
                    }
                    let table = Arc::new(HostTable::compact(key, bytes)?);
                    cache.insert(key, Arc::downgrade(&table));
                    table
                };
                table
            };
            self.host_profile = Some(table.clone());
            frame.table = Some(table);
        }
        self.metrics.host_retained.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::new(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonseparable_calibration_preserves_reference_instead_of_rejecting_frames() {
        let mut bytes = vec![0; 256 * 256 * 256 * 3];
        // Blue changing with V violates the compact representation.
        bytes[256 * 3] = 37;
        let table = HostTable::compact((5, 1), bytes).unwrap();
        assert!(table.full.is_some());
        assert_eq!(table.pixel(0, 0, 1), [37, 0, 0]);
        assert_eq!(table.pixel(0, 0, 0), [0, 0, 0]);
    }
}

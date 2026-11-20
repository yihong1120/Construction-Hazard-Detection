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
    band_format: i32,
}
impl Info {
    const PLANAR_420: i32 = 1;

    fn supports_bands(&self, width: u32, height: u32) -> bool {
        self.band_format != 0
            && self.width == width as i32
            && self.height == height as i32
            && width % 2 == 0
            && height % 2 == 0
            && self.yp >= self.width
            && self.up >= self.width / 2
            && self.vp >= self.width / 2
            && self.y != 0
            && self.u != 0
            && self.v != 0
    }
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
    fn cm_host_band_free(scaler: *mut c_void);
    fn cm_host_bgr_band(
        pointer: *const c_void,
        scaler: *mut *mut c_void,
        row: i32,
        rows: i32,
        target: *mut u8,
        length: usize,
    ) -> i32;
    fn cm_cpu_lut_fill(target: *mut u8, length: usize, space: i32, range: i32) -> i32;
}
pub(crate) struct HostProfile {
    key: (i32, i32),
    table: OnceLock<Option<HostTable>>,
}

impl HostProfile {
    fn table(&self) -> Option<&HostTable> {
        self.table
            .get_or_init(|| {
                let mut bytes = vec![0; 256 * 256 * 256 * 3];
                let result = unsafe {
                    cm_cpu_lut_fill(bytes.as_mut_ptr(), bytes.len(), self.key.0, self.key.1)
                };
                (result == 0).then(|| HostTable::compact(bytes))
            })
            .as_ref()
    }
}

struct HostTable {
    // Green depends on all three components; blue only on U/Y, red on V/Y.
    // Validate this against the calibrated reference instead of approximating
    // libswscale coefficients or rounding.
    full: Option<Vec<u8>>,
    green: Vec<u8>,
    blue: Vec<u8>,
    red: Vec<u8>,
}
impl HostTable {
    fn compact(bytes: Vec<u8>) -> Self {
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
                        return Self {
                            full: Some(bytes),
                            green: vec![],
                            blue: vec![],
                            red: vec![],
                        };
                    }
                    green[i] = p[1];
                }
            }
        }
        Self {
            full: None,
            green,
            blue,
            red,
        }
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
    profile: Option<Arc<HostProfile>>,
    cpu: OnceLock<Result<Vec<u8>, String>>,
    metrics: Arc<GpuMetrics>,
}

// OpenCV's 8-bit INTER_LINEAR uses half-pixel coordinates rounded to float
// precision and 11-bit interpolation weights. Keep the vertical rounding
// separate from the horizontal pass to preserve the model's input bytes.
fn linear_sample(
    index: usize,
    scale: f64,
    source_extent: usize,
    horizontal: bool,
) -> ([usize; 2], [u32; 2]) {
    let position = ((index as f64 + 0.5) * scale - 0.5) as f32;
    let origin = position.floor() as isize;
    let last = source_extent as isize - 1;
    let indices = [
        origin.clamp(0, last) as usize,
        (origin + 1).clamp(0, last) as usize,
    ];
    // OpenCV sets the horizontal weights to (1, 0) at either border.
    // The vertical stage clamps both row indices but keeps its weights.
    let fraction = if horizontal && indices[0] == indices[1] {
        0.0
    } else {
        position - origin as f32
    };
    (
        indices,
        [
            ((1.0 - fraction) * 2048.0).round_ties_even() as u32,
            (fraction * 2048.0).round_ties_even() as u32,
        ],
    )
}

struct HostBands {
    scaler: *mut c_void,
    bgr: [Vec<u8>; 2],
    columns: Vec<([usize; 2], [u32; 2])>,
    column_geometry: (usize, usize),
    horizontal: [Vec<u16>; 2],
}

impl Default for HostBands {
    fn default() -> Self {
        Self {
            scaler: std::ptr::null_mut(),
            bgr: Default::default(),
            columns: Vec::new(),
            column_geometry: (0, 0),
            horizontal: Default::default(),
        }
    }
}

impl Drop for HostBands {
    fn drop(&mut self) {
        // SAFETY: this thread owns the scaler and no conversions are running.
        unsafe { cm_host_band_free(self.scaler) };
    }
}

impl HostBands {
    fn prepare_columns(&mut self, sw: usize, rw: usize) {
        if self.column_geometry != (sw, rw) {
            let scale = 1.0 / (rw as f64 / sw as f64);
            self.columns.clear();
            self.columns.extend((0..rw).map(|x| {
                let (columns, weights) = linear_sample(x, scale, sw, true);
                (columns.map(|column| column * 3), weights)
            }));
            self.column_geometry = (sw, rw);
        }
    }

    fn load(&mut self, host: &HostFrame, n: usize, row: usize, rows: usize) -> bool {
        let buffer = &mut self.bgr[n];
        buffer.resize(host.width as usize * rows * 3 + 64, 0);
        // SAFETY: the retained frame owns the native planes, and the scaler
        // plus writable scratch buffer are private to the current thread.
        unsafe {
            cm_host_bgr_band(
                host.as_ptr(),
                &mut self.scaler,
                row as i32,
                rows as i32,
                buffer.as_mut_ptr(),
                buffer.len(),
            ) == 0
        }
    }

    fn resize(
        &mut self,
        host: &HostFrame,
        target: &mut [u8],
        width: usize,
        rows: usize,
        columns: usize,
    ) -> bool {
        let sw = host.width as usize;
        let sh = host.height as usize;
        let half = sh == rows * 2 && sw == columns * 2;
        if !half {
            self.prepare_columns(sw, columns);
            for row in &mut self.horizontal {
                row.resize(columns * 3, 0);
            }
        }
        let stride = sw * 3;
        let scale = 1.0 / (rows as f64 / sh as f64);
        // Small reductions touch most source rows. Larger bands amortise the
        // converter call there; large reductions convert only the tapped rows.
        let band_rows = sh.min(if sh <= rows * 3 { 16 } else { 2 });
        let band_start = |row: usize| (row / band_rows * band_rows).min(sh - band_rows);
        let mut loaded = [None; 2];
        let mut horizontal_rows = [None; 2];
        for (row, output) in target.chunks_mut(width * 3).take(rows).enumerate() {
            let ([first, second], vertical) = if half {
                ([row * 2, row * 2 + 1], [0; 2])
            } else {
                linear_sample(row, scale, sh, false)
            };
            let start = band_start(first);
            let first_slot = usize::from(loaded[1] == Some(start));
            if loaded[first_slot] != Some(start) && !self.load(host, first_slot, start, band_rows) {
                return false;
            }
            loaded[first_slot] = Some(start);
            let (second_slot, second_start) = if second < start + band_rows {
                (first_slot, start)
            } else {
                let slot = 1 - first_slot;
                let start = band_start(second);
                if loaded[slot] != Some(start) && !self.load(host, slot, start, band_rows) {
                    return false;
                }
                loaded[slot] = Some(start);
                (slot, start)
            };
            let row_pixels = |slot: usize, row: usize, start: usize| {
                let offset = (row - start) * stride;
                &self.bgr[slot][offset..offset + stride]
            };
            let upper = row_pixels(first_slot, first, start);
            let lower = row_pixels(second_slot, second, second_start);
            if half {
                // INTER_LINEAR uses the integer area kernel for exactly 2×
                // reduction in both axes, rounding the sum of four pixels.
                for (out, (a, b)) in output
                    .chunks_exact_mut(3)
                    .zip(upper.chunks_exact(6).zip(lower.chunks_exact(6)))
                {
                    for c in 0..3 {
                        let sum = u16::from(a[c])
                            + u16::from(a[c + 3])
                            + u16::from(b[c])
                            + u16::from(b[c + 3]);
                        out[c] = ((sum + 2) >> 2) as u8;
                    }
                }
                continue;
            }
            let upper_idx = usize::from(horizontal_rows[1] == Some(first));
            if horizontal_rows[upper_idx] != Some(first) {
                resize_horizontal(upper, &self.columns, &mut self.horizontal[upper_idx]);
                horizontal_rows[upper_idx] = Some(first);
            }
            let lower_idx = if first == second {
                upper_idx
            } else {
                1 - upper_idx
            };
            if horizontal_rows[lower_idx] != Some(second) {
                resize_horizontal(lower, &self.columns, &mut self.horizontal[lower_idx]);
                horizontal_rows[lower_idx] = Some(second);
            }
            for (out, (&a, &b)) in output.iter_mut().zip(
                self.horizontal[upper_idx]
                    .iter()
                    .zip(&self.horizontal[lower_idx]),
            ) {
                // OpenCV truncates the terms independently, then rounds.
                let a = (vertical[0] * u32::from(a)) >> 16;
                let b = (vertical[1] * u32::from(b)) >> 16;
                *out = ((a + b + 2) >> 2) as u8;
            }
        }
        true
    }
}

fn resize_horizontal(source: &[u8], columns: &[([usize; 2], [u32; 2])], target: &mut [u16]) {
    for (out, &([x0, x1], [a, b])) in target.chunks_exact_mut(3).zip(columns) {
        let first = &source[x0..x0 + 3];
        let second = &source[x1..x1 + 3];
        for c in 0..3 {
            out[c] = ((u32::from(first[c]) * a + u32::from(second[c]) * b) >> 4) as u16;
        }
    }
}

thread_local! {
    static HOST_BANDS: std::cell::RefCell<HostBands> = std::cell::RefCell::default();
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
        self.profile
            .as_ref()
            .and_then(|profile| profile.table.get())
            .and_then(Option::as_ref)
            .map_or(0, |t| {
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
    /// Sample the calibrated full BGR conversion with INTER_LINEAR, without
    /// constructing a full BGR frame. On failure the caller fills the output
    /// with the ordinary reference path.
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
        if !self.info.supports_bands(self.width, self.height) {
            return false;
        }
        let (sh, sw) = (self.height as usize, self.width as usize);
        if rh == 0
            || rw == 0
            || h.checked_mul(w).and_then(|n| n.checked_mul(3)) != Some(target.len())
            || top.checked_add(rh).is_none_or(|v| v > h)
            || left.checked_add(rw).is_none_or(|v| v > w)
        {
            return false;
        }
        let odd = sw % rw == 0 && sh % rh == 0 && sw / rw == sh / rh && (sw / rw) % 2 == 1;
        let table = if odd {
            self.profile.as_ref().and_then(|profile| profile.table())
        } else {
            None
        };
        if table.is_none() && self.cpu.get().is_some_and(Result::is_ok) {
            return false;
        }
        // Content is completely overwritten below. Initialize only letterbox bands.
        target[..top * w * 3].fill(114);
        target[(top + rh) * w * 3..].fill(114);
        for row in top..top + rh {
            target[row * w * 3..(row * w + left) * 3].fill(114);
            target[(row * w + left + rw) * 3..(row + 1) * w * 3].fill(114);
        }
        if let Some(table) = table {
            let p = self.info;
            // SAFETY: retaining the AVFrame keeps the positive-stride plane
            // ranges alive. Profile eligibility checks all plane widths.
            let (y, u, v) = unsafe {
                (
                    std::slice::from_raw_parts(p.y as *const u8, p.yp as usize * sh),
                    std::slice::from_raw_parts(p.u as *const u8, p.up as usize * (sh / 2)),
                    std::slice::from_raw_parts(p.v as *const u8, p.vp as usize * (sh / 2)),
                )
            };
            let scale = sw / rw;
            for row in 0..rh {
                let sy = row * scale + scale / 2;
                let offset = ((top + row) * w + left) * 3;
                for (x, out) in target[offset..offset + rw * 3]
                    .chunks_exact_mut(3)
                    .enumerate()
                {
                    let sx = x * scale + scale / 2;
                    let yy = y[sy * p.yp as usize + sx] as usize;
                    let uu = u[(sy / 2) * p.up as usize + sx / 2] as usize;
                    let vv = v[(sy / 2) * p.vp as usize + sx / 2] as usize;
                    out.copy_from_slice(&table.pixel(yy, uu, vv));
                }
            }
        } else {
            let offset = (top * w + left) * 3;
            let processed = HOST_BANDS
                .with_borrow_mut(|bands| bands.resize(self, &mut target[offset..], w, rh, rw));
            if !processed {
                return false;
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
            profile: None,
            cpu: OnceLock::new(),
            metrics: self.metrics.clone(),
        };
        if info.supports_bands(width, height)
            && info.band_format == Info::PLANAR_420
            && width % 32 == 0
            && matches!(info.colorspace, 1 | 2 | 5 | 6)
            && matches!(info.range, 0..=2)
        {
            let key = (
                if info.colorspace == 1 { 1 } else { 5 },
                if info.range == 2 { 2 } else { 1 },
            );
            let profile = if let Some(profile) = self.host_profile.as_ref().filter(|p| p.key == key)
            {
                profile.clone()
            } else {
                type Profiles = Mutex<HashMap<(i32, i32), Weak<HostProfile>>>;
                static PROFILES: OnceLock<Profiles> = OnceLock::new();
                let mut cache = PROFILES
                    .get_or_init(Mutex::default)
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(profile) = cache.get(&key).and_then(Weak::upgrade) {
                    profile
                } else {
                    let profile = Arc::new(HostProfile {
                        key,
                        table: OnceLock::new(),
                    });
                    cache.insert(key, Arc::downgrade(&profile));
                    profile
                }
            };
            self.host_profile = Some(profile.clone());
            frame.profile = Some(profile);
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
        let table = HostTable::compact(bytes);
        assert!(table.full.is_some());
        assert_eq!(table.pixel(0, 0, 1), [37, 0, 0]);
        assert_eq!(table.pixel(0, 0, 0), [0, 0, 0]);
    }
}

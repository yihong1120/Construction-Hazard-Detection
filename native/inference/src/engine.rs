use crate::core::{Error, Image, decode, layout, validate, validate_dimensions};
use std::collections::HashMap;

#[cfg(feature = "tensorrt")]
pub(crate) mod ffi {
    use std::ffi::{c_char, c_void};
    unsafe extern "C" {
        pub fn ci_submit_nv12(
            engine: *mut c_void,
            planes: *const super::Nv12Planes,
            dims: *const i32,
            n: i32,
            h: i32,
            w: i32,
            ticket: *mut u64,
            rows: *mut i32,
        ) -> i32;
        pub fn ci_error() -> *const c_char;
        pub fn ci_error_code() -> i32;
        pub fn ci_header_version() -> i32;
        pub fn ci_create(
            data: *const u8,
            length: usize,
            batch: i32,
            size: i32,
            device: i32,
            asynchronous: i32,
        ) -> *mut c_void;
        pub fn ci_destroy(engine: *mut c_void);
        pub fn ci_max_batch(engine: *mut c_void) -> i32;
        pub fn ci_activation_bytes(engine: *mut c_void) -> usize;
        pub fn ci_rows(engine: *mut c_void, n: i32, h: i32, w: i32, rows: *mut i32) -> i32;
        pub fn ci_predict(
            engine: *mut c_void,
            source: *const u8,
            result: *mut f32,
            n: i32,
            h: i32,
            w: i32,
        ) -> i32;
        pub fn ci_encoder_create(w: i32, h: i32, fps: i32, device: i32) -> *mut c_void;
        pub fn ci_encoder_destroy(encoder: *mut c_void);
        pub fn ci_encoder_encode(
            encoder: *mut c_void,
            image: *mut c_void,
            output: *mut *const u8,
            length: *mut usize,
        ) -> i32;
        pub fn ci_engine_stats(engine: *mut c_void, values: *mut u64);
        pub fn ci_factory_stats(factory: *mut c_void, values: *mut u64);
        pub fn ci_gpu_sync(device: i32) -> i32;
        pub fn ci_factory_create(data: *const u8, length: usize, device: i32) -> *mut c_void;
        pub fn ci_factory_destroy(pointer: *mut c_void);
        pub fn ci_frame_import(
            factory: *mut c_void,
            y: *const u8,
            uv: *const u8,
            yp: i32,
            uvp: i32,
            w: i32,
            h: i32,
        ) -> *mut c_void;
        pub fn ci_frame_destroy(pointer: *mut c_void);
        pub fn ci_frame_copy(pointer: *mut c_void) -> *mut c_void;
        pub fn ci_frame_download(pointer: *mut c_void, target: *mut u8) -> i32;
        pub fn ci_frame_blend(
            pointer: *mut c_void,
            data: *const u8,
            w: i32,
            h: i32,
            x: i32,
            y: i32,
        ) -> i32;
        pub fn ci_submit_device(
            engine: *mut c_void,
            frames: *const *mut c_void,
            dims: *const i32,
            n: i32,
            h: i32,
            w: i32,
            ticket: *mut u64,
            rows: *mut i32,
        ) -> i32;
        pub fn ci_submit(
            engine: *mut c_void,
            source: *const u8,
            n: i32,
            h: i32,
            w: i32,
            ticket: *mut u64,
            rows: *mut i32,
        ) -> i32;
        pub fn ci_reserve_host(
            engine: *mut c_void,
            n: i32,
            h: i32,
            w: i32,
            ticket: *mut u64,
            target: *mut *mut u8,
            length: *mut usize,
        ) -> i32;
        pub fn ci_abort_host(engine: *mut c_void, ticket: u64) -> i32;
        pub fn ci_submit_host(engine: *mut c_void, ticket: u64, rows: *mut i32) -> i32;
        pub fn ci_submit_bgr(
            engine: *mut c_void,
            sources: *const *const u8,
            dims: *const i32,
            n: i32,
            h: i32,
            w: i32,
            ticket: *mut u64,
            rows: *mut i32,
        ) -> i32;
        pub fn ci_debug_input(
            engine: *mut c_void,
            ticket: u64,
            target: *mut f32,
            elements: usize,
        ) -> i32;
        pub fn ci_collect(engine: *mut c_void, ticket: u64, target: *mut f32) -> i32;
        pub fn ci_pinned_bytes(engine: *mut c_void) -> usize;
        pub fn ci_letterbox(
            source: *const u8,
            h: i32,
            w: i32,
            target: *mut u8,
            rh: i32,
            rw: i32,
            th: i32,
            tw: i32,
            top: i32,
            left: i32,
        ) -> i32;
    }
    pub fn status(value: i32) -> Result<(), super::Error> {
        if value == 0 {
            return Ok(());
        }
        let message = unsafe { std::ffi::CStr::from_ptr(ci_error()) }
            .to_string_lossy()
            .into_owned();
        Err(match value {
            -2 => super::Error::Unsupported(message),
            -3 => super::Error::Invalid(message),
            _ => super::Error::Runtime(message),
        })
    }
}

// Raw buffer views are deliberately !Send: they must stay with their owner.
pub struct BufferImage {
    pub h: usize,
    pub w: usize,
    pub bytes: usize,
    pub pointer: *const u8,
}

// Prepared batches own their pixels and dimensions for worker handoff.
pub struct PreparedBatch {
    pixels: Vec<u8>,
    shapes: Vec<(usize, usize)>,
    h: usize,
    w: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Nv12Planes {
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
pub struct Nv12Image {
    planes: Nv12Planes,
    owner: std::sync::Arc<dyn Send + Sync>,
}
impl Nv12Image {
    /// # Safety
    /// The owner must keep immutable CUDA NV12 planes and the recorded ready
    /// event and calibrated colour table alive. All must belong to the device's
    /// primary context. Pitches and dimensions must describe allocated storage.
    pub unsafe fn new(planes: Nv12Planes, owner: std::sync::Arc<dyn Send + Sync>) -> Self {
        Self { planes, owner }
    }
}

struct PendingBatch {
    shapes: Vec<(usize, usize)>,
    h: usize,
    w: usize,
    rows: usize,
}

pub struct NativeEngine {
    #[cfg(feature = "tensorrt")]
    pointer: std::ptr::NonNull<std::ffi::c_void>,
    pub max_batch: usize,
    size: usize,
    stride: usize,
    rect: bool,
    confidence: f32,
    pixels_cache: Vec<u8>,
    output_cache: Vec<f32>,
    pub workspace_growths: usize,
    pending: HashMap<u64, PendingBatch>,
    nv12_owners: HashMap<u64, Vec<std::sync::Arc<dyn Send + Sync>>>,
}
// A context is used exclusively through &mut self. The bridge selects the CUDA
// device on every operation and before destruction, allowing sequential moves.
#[cfg(feature = "tensorrt")]
unsafe impl Send for NativeEngine {}

impl NativeEngine {
    pub fn new(
        path: &str,
        batch: usize,
        size: usize,
        device: i32,
        rect: bool,
        confidence: f32,
        asynchronous: bool,
    ) -> Result<Self, Error> {
        if batch == 0
            || batch > 1024
            || size < 32
            || size > 4096
            || device < 0
            || !confidence.is_finite()
            || !(0.0..=1.0).contains(&confidence)
            || batch * 3 * size * size > i32::MAX as usize
        {
            return Err(Error::Invalid("Invalid native engine configuration".into()));
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            let _ = (path, rect, asynchronous);
            Err(Error::Unsupported(
                "Wheel was built without TensorRT support".into(),
            ))
        }
        #[cfg(feature = "tensorrt")]
        {
            let bytes = std::fs::read(path)
                .map_err(|e| Error::Runtime(format!("Cannot read engine: {e}")))?;
            let (plan, stride) = plan(&bytes)?;
            let size = size.div_ceil(stride) * stride;
            let pointer = unsafe {
                ffi::ci_create(
                    plan.as_ptr(),
                    plan.len(),
                    batch as i32,
                    size as i32,
                    device,
                    i32::from(asynchronous),
                )
            };
            let Some(pointer) = std::ptr::NonNull::new(pointer) else {
                ffi::status(unsafe { ffi::ci_error_code() })?;
                return Err(Error::Runtime("Native engine initialization failed".into()));
            };
            let max_batch = unsafe { ffi::ci_max_batch(pointer.as_ptr()) } as usize;
            Ok(Self {
                pointer,
                max_batch,
                size,
                stride,
                rect,
                confidence,
                pixels_cache: Vec::new(),
                output_cache: Vec::new(),
                workspace_growths: 0,
                pending: HashMap::new(),
                nv12_owners: HashMap::new(),
            })
        }
    }

    pub fn activation_bytes(&self) -> usize {
        #[cfg(feature = "tensorrt")]
        {
            unsafe { ffi::ci_activation_bytes(self.pointer.as_ptr()) }
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            0
        }
    }

    pub fn predict(&mut self, images: &[Image<'_>]) -> Result<Vec<Vec<[f32; 6]>>, Error> {
        let prepared = self.prepare_images(images)?;
        self.run_prepared(prepared)
    }

    /// Each callback must fill its own frame, including its numerical fallback.
    /// A failed frame never discards work performed for another frame.
    pub fn prepare_surfaces_each<F>(
        &mut self,
        shapes: &[(usize, usize)],
        pool: Option<&rayon::ThreadPool>,
        fill: F,
    ) -> Result<(PreparedBatch, usize), Error>
    where
        F: Fn(usize, &crate::core::Layout, &mut [u8]) -> Result<usize, Error> + Sync,
    {
        let layouts = self.surface_layouts(shapes)?;
        let (h, w) = (layouts[0].h, layouts[0].w);
        let mut pixels = std::mem::take(&mut self.pixels_cache);
        let needed = h * w * 3 * shapes.len();
        if pixels.capacity() < needed {
            self.workspace_growths += 1;
        }
        pixels.resize(needed, 0);
        match fill_surfaces(&mut pixels, &layouts, pool, &fill) {
            Ok(count) => Ok((
                PreparedBatch {
                    pixels,
                    shapes: shapes.to_vec(),
                    h,
                    w,
                },
                count,
            )),
            Err(e) => {
                self.pixels_cache = pixels;
                Err(e)
            }
        }
    }
    fn surface_layouts(
        &self,
        shapes: &[(usize, usize)],
    ) -> Result<Vec<crate::core::Layout>, Error> {
        if shapes.is_empty() || shapes.len() > self.max_batch {
            return Err(Error::Invalid("Invalid surface batch".into()));
        }
        for &(h, w) in shapes {
            crate::core::validate_dimensions(
                h,
                w,
                h.checked_mul(w).and_then(|n| n.checked_mul(3)).unwrap_or(0),
            )?;
        }
        let rect = self.rect && shapes.iter().all(|s| *s == shapes[0]);
        Ok(shapes
            .iter()
            .map(|&(h, w)| layout(h, w, self.size, self.stride, rect))
            .collect())
    }
    /// Fill a validated frame using the same reference as ordinary CPU input.
    pub fn fill_letterbox(
        image: &Image<'_>,
        l: &crate::core::Layout,
        target: &mut [u8],
    ) -> Result<(), Error> {
        crate::core::validate_dimensions(image.h, image.w, image.data.len())?;
        if l.h.checked_mul(l.w).and_then(|n| n.checked_mul(3)) != Some(target.len())
            || l.top.checked_add(l.rh).is_none_or(|n| n > l.h)
            || l.left.checked_add(l.rw).is_none_or(|n| n > l.w)
        {
            return Err(Error::Invalid("Invalid letterbox destination".into()));
        }
        if crate::resize::letterbox(image, l, target) {
            return Ok(());
        }
        #[cfg(feature = "tensorrt")]
        {
            ffi::status(unsafe {
                ffi::ci_letterbox(
                    image.data.as_ptr(),
                    image.h as i32,
                    image.w as i32,
                    target.as_mut_ptr(),
                    l.rh as i32,
                    l.rw as i32,
                    l.h as i32,
                    l.w as i32,
                    l.top as i32,
                    l.left as i32,
                )
            })
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            Err(Error::Unsupported("Built without TensorRT".into()))
        }
    }

    /// This can run while a previous asynchronous ticket is on the GPU.
    pub fn prepare_images(&mut self, images: &[Image<'_>]) -> Result<PreparedBatch, Error> {
        validate(images, self.max_batch)?;
        let views: Vec<_> = images
            .iter()
            .map(|image| BufferImage {
                h: image.h,
                w: image.w,
                bytes: image.data.len(),
                pointer: image.data.as_ptr(),
            })
            .collect();
        // SAFETY: immutable Image slices remain alive until preparation ends.
        unsafe { self.prepare_buffers(&views) }
    }

    /// Keep the OpenCV-only path available as the numerical reference.
    pub fn prepare_images_parallel(
        &mut self,
        images: &[Image<'_>],
        pool: &rayon::ThreadPool,
    ) -> Result<PreparedBatch, Error> {
        if images.len() < 2 {
            return self.prepare_images(images);
        }
        self.prepare_on_pool(images, Some(pool), false)
            .map(|(batch, _)| batch)
    }
    /// Exact odd-integer sampling, with OpenCV fallback for every other scale.
    pub fn prepare_images_fast(
        &mut self,
        images: &[Image<'_>],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<(PreparedBatch, usize), Error> {
        self.prepare_on_pool(images, pool, true)
    }
    fn prepare_on_pool(
        &mut self,
        images: &[Image<'_>],
        pool: Option<&rayon::ThreadPool>,
        exact: bool,
    ) -> Result<(PreparedBatch, usize), Error> {
        validate(images, self.max_batch)?;
        if images.is_empty() {
            return self.prepare_images(images).map(|batch| (batch, 0));
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            let _ = (pool, exact);
            Err(Error::Unsupported("Built without TensorRT".into()))
        }
        #[cfg(feature = "tensorrt")]
        {
            use rayon::prelude::*;
            let rect = self.rect
                && images
                    .iter()
                    .all(|i| (i.h, i.w) == (images[0].h, images[0].w));
            let layouts: Vec<_> = images
                .iter()
                .map(|i| layout(i.h, i.w, self.size, self.stride, rect))
                .collect();
            let first = &layouts[0];
            let image_bytes = first.h * first.w * 3;
            let mut pixels = std::mem::take(&mut self.pixels_cache);
            let needed = image_bytes * images.len();
            if pixels.capacity() < needed {
                self.workspace_growths += 1;
            }
            pixels.resize(needed, 0);
            let process = |((target, image), l): (
                (&mut [u8], &Image<'_>),
                &crate::core::Layout,
            )|
             -> Result<usize, Error> {
                if exact && crate::resize::letterbox(image, l, target) {
                    return Ok(1);
                }
                // SAFETY: independent validated source/output slices live
                // through this call. No pointer escapes preparation.
                ffi::status(unsafe {
                    ffi::ci_letterbox(
                        image.data.as_ptr(),
                        image.h as i32,
                        image.w as i32,
                        target.as_mut_ptr(),
                        l.rh as i32,
                        l.rw as i32,
                        l.h as i32,
                        l.w as i32,
                        l.top as i32,
                        l.left as i32,
                    )
                })?;
                Ok(0)
            };
            let result = if let Some(pool) = pool {
                pool.install(|| {
                    pixels
                        .par_chunks_mut(image_bytes)
                        .zip(images.par_iter())
                        .zip(layouts.par_iter())
                        .map(process)
                        .try_reduce(|| 0, |a, b| Ok(a + b))
                })
            } else {
                pixels
                    .chunks_exact_mut(image_bytes)
                    .zip(images)
                    .zip(&layouts)
                    .map(process)
                    .try_fold(0, |sum, count| count.map(|n| sum + n))
            };
            let count = match result {
                Ok(count) => count,
                Err(error) => {
                    self.pixels_cache = pixels;
                    return Err(error);
                }
            };
            Ok((
                PreparedBatch {
                    pixels,
                    shapes: images.iter().map(|i| (i.h, i.w)).collect(),
                    h: first.h,
                    w: first.w,
                },
                count,
            ))
        }
    }

    /// # Safety
    /// Each pointer must remain valid and unchanged for `bytes` bytes until this
    /// call returns. No source pointer is stored in the returned owned batch.
    pub unsafe fn prepare_buffers(
        &mut self,
        images: &[BufferImage],
    ) -> Result<PreparedBatch, Error> {
        if images.len() > self.max_batch {
            return Err(Error::Invalid("Batch exceeds engine capacity".into()));
        }
        for image in images {
            validate_dimensions(image.h, image.w, image.bytes)?;
            if image.pointer.is_null() {
                return Err(Error::Invalid("Null image buffer".into()));
            }
        }
        if images.is_empty() {
            return Ok(PreparedBatch {
                pixels: Vec::new(),
                shapes: Vec::new(),
                h: 0,
                w: 0,
            });
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            Err(Error::Unsupported(
                "Wheel was built without TensorRT support".into(),
            ))
        }
        #[cfg(feature = "tensorrt")]
        {
            let rect = self.rect
                && images
                    .iter()
                    .all(|image| (image.h, image.w) == (images[0].h, images[0].w));
            let layouts: Vec<_> = images
                .iter()
                .map(|image| layout(image.h, image.w, self.size, self.stride, rect))
                .collect();
            let first = &layouts[0];
            let image_bytes = first.h * first.w * 3;
            let mut pixels = std::mem::take(&mut self.pixels_cache);
            let needed = image_bytes * images.len();
            if pixels.capacity() < needed {
                self.workspace_growths += 1;
            }
            pixels.resize(needed, 0);
            for (index, (image, l)) in images.iter().zip(&layouts).enumerate() {
                ffi::status(unsafe {
                    ffi::ci_letterbox(
                        image.pointer,
                        image.h as i32,
                        image.w as i32,
                        pixels[index * image_bytes..].as_mut_ptr(),
                        l.rh as i32,
                        l.rw as i32,
                        l.h as i32,
                        l.w as i32,
                        l.top as i32,
                        l.left as i32,
                    )
                })?;
            }
            Ok(PreparedBatch {
                pixels,
                shapes: images.iter().map(|image| (image.h, image.w)).collect(),
                h: first.h,
                w: first.w,
            })
        }
    }

    #[cfg(feature = "tensorrt")]
    pub fn submit_device(&mut self, images: &[crate::gpu::GpuFrame]) -> Result<u64, Error> {
        if images.is_empty() || images.len() > self.max_batch {
            return Err(Error::Invalid("Invalid device batch capacity".into()));
        }
        let rect = self.rect
            && images
                .iter()
                .all(|f| (f.height(), f.width()) == (images[0].height(), images[0].width()));
        let layouts: Vec<_> = images
            .iter()
            .map(|f| layout(f.height(), f.width(), self.size, self.stride, rect))
            .collect();
        let first = &layouts[0];
        let dims: Vec<i32> = layouts
            .iter()
            .flat_map(|l| [l.rh as i32, l.rw as i32, l.top as i32, l.left as i32])
            .collect();
        // Deduplicate repeated frames and acquire multiple frame locks in a
        // stable order, including when separate engines share input images.
        let unique: std::collections::BTreeMap<_, _> = images
            .iter()
            .map(|f| (std::sync::Arc::as_ptr(&f.core) as usize, &f.core))
            .collect();
        let guards: std::collections::BTreeMap<_, _> = unique
            .iter()
            .map(|(&key, f)| {
                f.handle
                    .lock()
                    .map(|guard| (key, guard))
                    .map_err(|_| Error::Runtime("GPU image lock poisoned".into()))
            })
            .collect::<Result<_, _>>()?;
        let pointers: Vec<_> = images
            .iter()
            .map(|f| {
                guards[&(std::sync::Arc::as_ptr(&f.core) as usize)]
                    .as_ref()
                    .map(|h| h.0.as_ptr())
                    .ok_or_else(|| Error::Runtime("GPU image is closed".into()))
            })
            .collect::<Result<_, _>>()?;
        let mut ticket = 0;
        let mut rows = 0;
        ffi::status(unsafe {
            ffi::ci_submit_device(
                self.pointer.as_ptr(),
                pointers.as_ptr(),
                dims.as_ptr(),
                images.len() as i32,
                first.h as i32,
                first.w as i32,
                &mut ticket,
                &mut rows,
            )
        })?;
        self.pending.insert(
            ticket,
            PendingBatch {
                shapes: images.iter().map(|f| (f.height(), f.width())).collect(),
                h: first.h,
                w: first.w,
                rows: rows as usize,
            },
        );
        Ok(ticket)
    }

    pub fn async_stats(&self) -> [u64; 3] {
        #[cfg(feature = "tensorrt")]
        {
            let mut values = [0; 3];
            unsafe { ffi::ci_engine_stats(self.pointer.as_ptr(), values.as_mut_ptr()) };
            values
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            [0; 3]
        }
    }
    pub fn pinned_bytes(&self) -> usize {
        #[cfg(feature = "tensorrt")]
        {
            unsafe { ffi::ci_pinned_bytes(self.pointer.as_ptr()) }
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            0
        }
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    /// Diagnostic D2H tensor copy; never called by the live pipeline.
    pub fn inspect_input(&self, ticket: u64) -> Result<Vec<f32>, Error> {
        let p = self
            .pending
            .get(&ticket)
            .ok_or_else(|| Error::Invalid("Unknown ticket".into()))?;
        let mut tensor = vec![0f32; p.shapes.len() * 3 * p.h * p.w];
        #[cfg(feature = "tensorrt")]
        ffi::status(unsafe {
            ffi::ci_debug_input(
                self.pointer.as_ptr(),
                ticket,
                tensor.as_mut_ptr(),
                tensor.len(),
            )
        })?;
        #[cfg(not(feature = "tensorrt"))]
        return Err(Error::Unsupported("Built without TensorRT".into()));
        #[allow(unreachable_code)]
        Ok(tensor)
    }
    /// Calibrated point-chroma NV12 preprocessing. Unsupported geometry or
    /// colour profiles return None before any device work is submitted.
    pub fn submit_nv12(&mut self, images: &[Nv12Image]) -> Result<Option<u64>, Error> {
        if images.is_empty() || images.len() > self.max_batch {
            return Err(Error::Invalid("Invalid NV12 batch size".into()));
        }
        for i in images {
            let p = i.planes;
            let bytes = (p.height as usize)
                .checked_mul(p.width as usize)
                .and_then(|n| n.checked_mul(3))
                .unwrap_or(0);
            validate_dimensions(p.height as usize, p.width as usize, bytes)?;
            if p.width % 2 != 0
                || p.height % 2 != 0
                || p.yp < p.width
                || p.uvp < p.width
                || p.y == 0
                || p.uv == 0
                || p.ready == 0
            {
                return Err(Error::Invalid("Invalid NV12 planes".into()));
            }
        }
        let rect = self.rect
            && images.iter().all(|i| {
                (i.planes.height, i.planes.width)
                    == (images[0].planes.height, images[0].planes.width)
            });
        let layouts: Vec<_> = images
            .iter()
            .map(|i| {
                layout(
                    i.planes.height as usize,
                    i.planes.width as usize,
                    self.size,
                    self.stride,
                    rect,
                )
            })
            .collect();
        if !images.iter().zip(&layouts).all(|(i, l)| {
            let p = i.planes;
            p.lut != 0
                && crate::resize::odd_scale(p.height as usize, p.width as usize, l).is_some()
                && matches!(p.colorspace, 1 | 2 | 5 | 6)
                && matches!(p.range, 0..=2)
        }) {
            return Ok(None);
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            Err(Error::Unsupported("Built without TensorRT".into()))
        }
        #[cfg(feature = "tensorrt")]
        {
            let planes: Vec<_> = images.iter().map(|i| i.planes).collect();
            let dims: Vec<i32> = layouts
                .iter()
                .flat_map(|l| [l.rh as i32, l.rw as i32, l.top as i32, l.left as i32])
                .collect();
            let first = &layouts[0];
            let (mut ticket, mut rows) = (0, 0);
            ffi::status(unsafe {
                ffi::ci_submit_nv12(
                    self.pointer.as_ptr(),
                    planes.as_ptr(),
                    dims.as_ptr(),
                    images.len() as i32,
                    first.h as i32,
                    first.w as i32,
                    &mut ticket,
                    &mut rows,
                )
            })?;
            self.nv12_owners
                .insert(ticket, images.iter().map(|i| i.owner.clone()).collect());
            self.pending.insert(
                ticket,
                PendingBatch {
                    shapes: images
                        .iter()
                        .map(|i| (i.planes.height as usize, i.planes.width as usize))
                        .collect(),
                    h: first.h,
                    w: first.w,
                    rows: rows as usize,
                },
            );
            Ok(Some(ticket))
        }
    }
    /// Return None for unsupported geometry without submitting any GPU work.
    /// Supported scales are identity and isotropic odd integer reductions.
    pub fn submit_images_exact(&mut self, images: &[Image<'_>]) -> Result<Option<u64>, Error> {
        validate(images, self.max_batch)?;
        if images.is_empty() {
            return Err(Error::Invalid("Empty BGR batch".into()));
        }
        let rect = self.rect
            && images
                .iter()
                .all(|i| (i.h, i.w) == (images[0].h, images[0].w));
        let layouts: Vec<_> = images
            .iter()
            .map(|i| layout(i.h, i.w, self.size, self.stride, rect))
            .collect();
        if !images.iter().zip(&layouts).all(|(i, l)| {
            i.h % l.rh == 0 && i.w % l.rw == 0 && i.h / l.rh == i.w / l.rw && (i.w / l.rw) % 2 == 1
        }) {
            return Ok(None);
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            Err(Error::Unsupported("Built without TensorRT".into()))
        }
        #[cfg(feature = "tensorrt")]
        {
            let sources: Vec<_> = images.iter().map(|i| i.data.as_ptr()).collect();
            let dims: Vec<i32> = images
                .iter()
                .zip(&layouts)
                .flat_map(|(i, l)| {
                    [
                        i.h as i32,
                        i.w as i32,
                        l.rh as i32,
                        l.rw as i32,
                        l.top as i32,
                        l.left as i32,
                    ]
                })
                .collect();
            let first = &layouts[0];
            let (mut ticket, mut rows) = (0, 0);
            // SAFETY: validated immutable inputs outlive the synchronous copy
            // into engine-owned pinned buffers. No borrowed pointer is kept.
            ffi::status(unsafe {
                ffi::ci_submit_bgr(
                    self.pointer.as_ptr(),
                    sources.as_ptr(),
                    dims.as_ptr(),
                    images.len() as i32,
                    first.h as i32,
                    first.w as i32,
                    &mut ticket,
                    &mut rows,
                )
            })?;
            self.pending.insert(
                ticket,
                PendingBatch {
                    shapes: images.iter().map(|i| (i.h, i.w)).collect(),
                    h: first.h,
                    w: first.w,
                    rows: rows as usize,
                },
            );
            Ok(Some(ticket))
        }
    }

    pub fn submit_images(&mut self, images: &[Image<'_>]) -> Result<u64, Error> {
        let prepared = self.prepare_images(images)?;
        self.submit_prepared(prepared)
    }
    pub fn submit_prepared(&mut self, batch: PreparedBatch) -> Result<u64, Error> {
        if batch.shapes.is_empty() {
            return Err(Error::Invalid("Cannot submit an empty batch".into()));
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            Err(Error::Unsupported(
                "Wheel was built without TensorRT support".into(),
            ))
        }
        #[cfg(feature = "tensorrt")]
        {
            let mut ticket = 0;
            let mut rows = 0;
            let result = ffi::status(unsafe {
                ffi::ci_submit(
                    self.pointer.as_ptr(),
                    batch.pixels.as_ptr(),
                    batch.shapes.len() as i32,
                    batch.h as i32,
                    batch.w as i32,
                    &mut ticket,
                    &mut rows,
                )
            });
            if batch.pixels.capacity() > self.pixels_cache.capacity() {
                self.pixels_cache = batch.pixels;
            }
            result?;
            self.pending.insert(
                ticket,
                PendingBatch {
                    shapes: batch.shapes,
                    h: batch.h,
                    w: batch.w,
                    rows: rows as usize,
                },
            );
            Ok(ticket)
        }
    }
    /// The engine exclusively owns this initialized pinned slot. The callback's
    /// slice cannot escape; the reservation survives parallel filling and the
    /// GPU owns the slot until collect. Drop also handles callback errors/panics.
    pub fn submit_surfaces_pinned<F>(
        &mut self,
        shapes: &[(usize, usize)],
        pool: Option<&rayon::ThreadPool>,
        fill: F,
    ) -> Result<(u64, usize), Error>
    where
        F: Fn(usize, &crate::core::Layout, &mut [u8]) -> Result<usize, Error> + Sync,
    {
        let layouts = self.surface_layouts(shapes)?;
        #[cfg(not(feature = "tensorrt"))]
        {
            let _ = (layouts, pool, fill);
            Err(Error::Unsupported("Built without TensorRT".into()))
        }
        #[cfg(feature = "tensorrt")]
        {
            struct Reservation {
                engine: *mut std::ffi::c_void,
                ticket: u64,
                armed: bool,
            }
            impl Drop for Reservation {
                fn drop(&mut self) {
                    if self.armed {
                        unsafe {
                            ffi::ci_abort_host(self.engine, self.ticket);
                        }
                    }
                }
            }
            let (h, w) = (layouts[0].h, layouts[0].w);
            let mut ticket = 0;
            let mut pointer = std::ptr::null_mut();
            let mut length = 0;
            ffi::status(unsafe {
                ffi::ci_reserve_host(
                    self.pointer.as_ptr(),
                    shapes.len() as i32,
                    h as i32,
                    w as i32,
                    &mut ticket,
                    &mut pointer,
                    &mut length,
                )
            })?;
            let mut guard = Reservation {
                engine: self.pointer.as_ptr(),
                ticket,
                armed: true,
            };
            if pointer.is_null() || length != shapes.len() * h * w * 3 {
                return Err(Error::Runtime("Invalid pinned reservation".into()));
            }
            // SAFETY: the C owner initialized this allocation and reserved it
            // exclusively. No resizing, submission or GPU use occurs while
            // fill_surfaces holds disjoint mutable chunks; the callback ends
            // before enqueue and the next Rust engine operation.
            let pixels = unsafe { std::slice::from_raw_parts_mut(pointer, length) };
            let count = fill_surfaces(pixels, &layouts, pool, &fill)?;
            let mut rows = 0;
            ffi::status(unsafe { ffi::ci_submit_host(self.pointer.as_ptr(), ticket, &mut rows) })?;
            guard.armed = false;
            self.pending.insert(
                ticket,
                PendingBatch {
                    shapes: shapes.to_vec(),
                    h,
                    w,
                    rows: rows as usize,
                },
            );
            Ok((ticket, count))
        }
    }
    pub fn collect(&mut self, ticket: u64) -> Result<Vec<Vec<[f32; 6]>>, Error> {
        let pending = self
            .pending
            .get(&ticket)
            .ok_or_else(|| Error::Invalid("Unknown or collected asynchronous ticket".into()))?;
        #[cfg(not(feature = "tensorrt"))]
        {
            let _ = pending;
            Err(Error::Unsupported(
                "Wheel was built without TensorRT support".into(),
            ))
        }
        #[cfg(feature = "tensorrt")]
        {
            let needed = pending.shapes.len() * pending.rows * 6;
            if self.output_cache.capacity() < needed {
                self.workspace_growths += 1;
            }
            self.output_cache.resize(needed, 0.0);
            ffi::status(unsafe {
                ffi::ci_collect(
                    self.pointer.as_ptr(),
                    ticket,
                    self.output_cache.as_mut_ptr(),
                )
            })?;
            let pending = self.pending.remove(&ticket).unwrap();
            self.nv12_owners.remove(&ticket);
            pending
                .shapes
                .iter()
                .zip(self.output_cache.chunks_exact(pending.rows * 6))
                .map(|(&(h, w), raw)| decode(raw, h, w, pending.h, pending.w, self.confidence))
                .collect()
        }
    }

    pub fn workspace_bytes(&self) -> usize {
        self.pixels_cache.capacity() + self.output_cache.capacity() * std::mem::size_of::<f32>()
    }

    pub fn run_prepared(&mut self, batch: PreparedBatch) -> Result<Vec<Vec<[f32; 6]>>, Error> {
        let result = self.run_pixels(&batch);
        if batch.pixels.capacity() > self.pixels_cache.capacity() {
            self.pixels_cache = batch.pixels;
        }
        result
    }

    fn run_pixels(&mut self, batch: &PreparedBatch) -> Result<Vec<Vec<[f32; 6]>>, Error> {
        if batch.shapes.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            Err(Error::Unsupported(
                "Wheel was built without TensorRT support".into(),
            ))
        }
        #[cfg(feature = "tensorrt")]
        {
            let n = batch.shapes.len();
            let mut rows = 0;
            ffi::status(unsafe {
                ffi::ci_rows(
                    self.pointer.as_ptr(),
                    n as i32,
                    batch.h as i32,
                    batch.w as i32,
                    &mut rows,
                )
            })?;
            let needed = n * rows as usize * 6;
            if self.output_cache.capacity() < needed {
                self.workspace_growths += 1;
            }
            self.output_cache.resize(needed, 0.0);
            let output = &mut self.output_cache;
            ffi::status(unsafe {
                ffi::ci_predict(
                    self.pointer.as_ptr(),
                    batch.pixels.as_ptr(),
                    output.as_mut_ptr(),
                    n as i32,
                    batch.h as i32,
                    batch.w as i32,
                )
            })?;
            batch
                .shapes
                .iter()
                .zip(output.chunks_exact(rows as usize * 6))
                .map(|(&(h, w), raw)| decode(raw, h, w, batch.h, batch.w, self.confidence))
                .collect()
        }
    }
}

#[cfg(feature = "tensorrt")]
impl Drop for NativeEngine {
    fn drop(&mut self) {
        unsafe { ffi::ci_destroy(self.pointer.as_ptr()) };
    }
}

pub fn header_version() -> i32 {
    #[cfg(feature = "tensorrt")]
    {
        unsafe { ffi::ci_header_version() }
    }
    #[cfg(not(feature = "tensorrt"))]
    {
        0
    }
}

fn plan(bytes: &[u8]) -> Result<(&[u8], usize), Error> {
    if bytes.len() < 4 {
        return Err(Error::Invalid("Truncated engine".into()));
    }
    let length = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    if length > 1024 * 1024 {
        return Ok((bytes, 32));
    } // Native TensorRT plan, no Ultralytics JSON header.
    if length == 0 || length >= bytes.len() - 4 {
        return Err(Error::Invalid("Truncated engine metadata".into()));
    }
    let metadata: serde_json::Value = serde_json::from_slice(&bytes[4..4 + length])
        .map_err(|e| Error::Invalid(format!("Invalid engine metadata: {e}")))?;
    if metadata
        .get("task")
        .and_then(|v| v.as_str())
        .is_some_and(|v| v != "detect")
    {
        return Err(Error::Unsupported("Rust supports detection engines".into()));
    }
    let stride = metadata
        .get("stride")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    if stride == 0 || stride > 256 {
        return Err(Error::Invalid("Invalid engine stride".into()));
    }
    Ok((&bytes[4 + length..], stride))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_metadata_and_preserves_plan_bytes() {
        let metadata = br#"{"task":"detect","stride":32}"#;
        let mut bytes = (metadata.len() as u32).to_le_bytes().to_vec();
        bytes.extend_from_slice(metadata);
        bytes.extend_from_slice(b"plan");
        assert_eq!(plan(&bytes).unwrap(), (b"plan".as_slice(), 32));
        assert!(plan(&[0, 0, 0, 0]).is_err());
    }
}

#[cfg(all(test, feature = "tensorrt"))]
mod parallel_preprocess_tests {
    use super::*;
    #[test]
    #[ignore = "Requires a compatible local TensorRT engine and GPU runtime"]
    fn exact_gpu_preprocessing_matches_cpu_tensor_and_falls_back() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/int8_engine/best_yolo26n.engine");
        let mut engine =
            NativeEngine::new(path.to_str().unwrap(), 8, 640, 0, false, 0.25, true).unwrap();
        for shapes in [
            vec![(1080, 1920); 4],
            vec![(352, 640); 4],
            vec![(1080, 1920), (352, 640), (1080, 1920), (352, 640)],
            vec![(640, 640); 2],
            vec![(1920, 1080); 2],
        ] {
            let storage: Vec<Vec<u8>> = shapes
                .iter()
                .map(|&(h, w)| {
                    (0..h * w * 3)
                        .map(|i| ((i * 37 + i / 71) % 256) as u8)
                        .collect()
                })
                .collect();
            let images: Vec<_> = shapes
                .iter()
                .zip(&storage)
                .map(|(&(h, w), data)| Image { h, w, data })
                .collect();
            let cpu = engine.prepare_images(&images).unwrap();
            let ticket = engine
                .submit_images_exact(&images)
                .unwrap()
                .expect("exact geometry");
            let elements = cpu.shapes.len() * cpu.h * cpu.w * 3;
            let mut actual = vec![0f32; elements];
            ffi::status(unsafe {
                ffi::ci_debug_input(
                    engine.pointer.as_ptr(),
                    ticket,
                    actual.as_mut_ptr(),
                    elements,
                )
            })
            .unwrap();
            let pixels = cpu.h * cpu.w;
            for n in 0..images.len() {
                for ch in 0..3 {
                    for p in 0..pixels {
                        let expected =
                            cpu.pixels[(n * pixels + p) * 3 + 2 - ch] as f32 * (1.0f32 / 255.0f32);
                        assert_eq!(
                            actual[(n * 3 + ch) * pixels + p].to_bits(),
                            expected.to_bits(),
                            "tensor differs at image {n}, channel {ch}, pixel {p}"
                        );
                    }
                }
            }
            engine.collect(ticket).unwrap();
            assert_eq!(engine.pending_count(), 0);
        }
        for (h, w) in [(720, 1280), (540, 960), (100, 100)] {
            let source = vec![113u8; h * w * 3];
            let images = [Image {
                h,
                w,
                data: &source,
            }];
            assert!(engine.submit_images_exact(&images).unwrap().is_none());
            assert_eq!(engine.pending_count(), 0);
            let prepared = engine.prepare_images(&images).unwrap();
            let ticket = engine.submit_prepared(prepared).unwrap();
            engine.collect(ticket).unwrap();
        }
    }
    #[test]
    #[ignore = "Requires a compatible local TensorRT engine and GPU runtime"]
    fn full_hd_parallel_preprocess_is_byte_exact_at_every_pool_size() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/int8_engine/best_yolo26n.engine");
        for rect in [false, true] {
            let mut engine =
                NativeEngine::new(path.to_str().unwrap(), 8, 640, 0, rect, 0.25, false).unwrap();
            for shapes in [
                vec![(1080, 1920); 4],
                vec![(1080, 1920), (720, 480), (1, 1), (352, 640)],
            ] {
                let storage: Vec<Vec<u8>> = shapes
                    .iter()
                    .map(|&(h, w)| {
                        (0..h * w * 3)
                            .map(|i| ((i * 37 + i / 71) % 256) as u8)
                            .collect()
                    })
                    .collect();
                let images: Vec<_> = shapes
                    .iter()
                    .zip(&storage)
                    .map(|(&(h, w), data)| Image { h, w, data })
                    .collect();
                let serial = engine.prepare_images(&images).unwrap();
                for size in [2, 4, 8] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(size)
                        .build()
                        .unwrap();
                    let parallel = engine.prepare_images_parallel(&images, &pool).unwrap();
                    assert_eq!(serial.pixels, parallel.pixels);
                    assert_eq!(serial.shapes, parallel.shapes);
                    assert_eq!((serial.h, serial.w), (parallel.h, parallel.w));
                    let (fast, count) = engine.prepare_images_fast(&images, Some(&pool)).unwrap();
                    assert_eq!(serial.pixels, fast.pixels);
                    assert_eq!(serial.shapes, fast.shapes);
                    let expected = images
                        .iter()
                        .filter(|i| {
                            crate::resize::odd_scale(
                                i.h,
                                i.w,
                                &layout(
                                    i.h,
                                    i.w,
                                    640,
                                    32,
                                    rect && images
                                        .iter()
                                        .all(|v| (v.h, v.w) == (images[0].h, images[0].w)),
                                ),
                            )
                            .is_some()
                        })
                        .count();
                    assert_eq!(count, expected);
                    let (sequential, _) = engine.prepare_images_fast(&images, None).unwrap();
                    assert_eq!(serial.pixels, sequential.pixels);
                }
            }
        }
    }
}

fn fill_surfaces<F>(
    pixels: &mut [u8],
    layouts: &[crate::core::Layout],
    pool: Option<&rayon::ThreadPool>,
    fill: &F,
) -> Result<usize, Error>
where
    F: Fn(usize, &crate::core::Layout, &mut [u8]) -> Result<usize, Error> + Sync,
{
    use rayon::prelude::*;
    let bytes = layouts[0].h * layouts[0].w * 3;
    if let Some(pool) = pool {
        pool.install(|| {
            pixels
                .par_chunks_mut(bytes)
                .zip(layouts)
                .enumerate()
                .map(|(i, (target, l))| fill(i, l, target))
                .try_reduce(|| 0, |a, b| Ok(a + b))
        })
    } else {
        pixels
            .chunks_exact_mut(bytes)
            .zip(layouts)
            .enumerate()
            .try_fold(0, |sum, (i, (target, l))| {
                fill(i, l, target).map(|n| sum + n)
            })
    }
}

//! CUDA frame and encoder ownership for the TensorRT bridge.

use crate::{core::Error, engine::ffi};
use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

const MAX_FRAME_PIXELS: usize = 64 * 1024 * 1024;
const NV12_LOOKUP_BYTES: usize = 256 * 256 * 256 * 3;

struct FactoryHandle(NonNull<c_void>);
// The factory's GPU table and kernel module are immutable. Bridge operations
// select the correct device, and the shared owner outlives every image.
unsafe impl Send for FactoryHandle {}
unsafe impl Sync for FactoryHandle {}
impl Drop for FactoryHandle {
    fn drop(&mut self) {
        unsafe { ffi::ci_factory_destroy(self.0.as_ptr()) };
    }
}

struct FactoryCore {
    handle: FactoryHandle,
    device: i32,
}

pub(crate) struct ImageHandle(pub(crate) NonNull<c_void>);
// Access is serialised by FrameCore's mutex. Inference keeps its own C++
// shared image reference before any image lock is released.
unsafe impl Send for ImageHandle {}
impl Drop for ImageHandle {
    fn drop(&mut self) {
        unsafe { ffi::ci_frame_destroy(self.0.as_ptr()) };
    }
}

pub(crate) struct FrameCore {
    pub(crate) handle: Mutex<Option<ImageHandle>>,
    factory: Arc<FactoryCore>,
    width: usize,
    height: usize,
}

#[derive(Clone)]
pub struct GpuFrame {
    pub(crate) core: Arc<FrameCore>,
}

pub struct GpuFrameFactory {
    core: Arc<FactoryCore>,
}

/// CUDA addresses and row pitches for one NV12 image.
pub struct DeviceNv12Planes {
    pub y: *const u8,
    pub uv: *const u8,
    pub y_pitch: usize,
    pub uv_pitch: usize,
}

pub fn validate_nv12_dimensions(width: usize, height: usize) -> Result<(), Error> {
    if !(2..=16384).contains(&width)
        || !(2..=16384).contains(&height)
        || width % 2 != 0
        || height % 2 != 0
        || width * height > MAX_FRAME_PIXELS
    {
        return Err(Error::Invalid(
            "Expected even NV12 image at most 64M pixels".into(),
        ));
    }
    Ok(())
}

fn checked_handle(ptr: *mut c_void) -> Result<NonNull<c_void>, Error> {
    if let Some(handle) = NonNull::new(ptr) {
        return Ok(handle);
    }
    ffi::status(unsafe { ffi::ci_error_code() })?;
    Err(Error::Runtime("GPU frame allocation failed".into()))
}

impl GpuFrameFactory {
    pub fn new(lut: &[u8], device: i32) -> Result<Self, Error> {
        if device < 0 || lut.len() != NV12_LOOKUP_BYTES {
            return Err(Error::Invalid(
                "Expected complete uint8 NV12 lookup and CUDA device".into(),
            ));
        }

        let handle =
            checked_handle(unsafe { ffi::ci_factory_create(lut.as_ptr(), lut.len(), device) })?;
        Ok(Self {
            core: Arc::new(FactoryCore {
                handle: FactoryHandle(handle),
                device,
            }),
        })
    }

    pub fn device(&self) -> i32 {
        self.core.device
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        ffi::status(unsafe { ffi::ci_gpu_sync(self.core.device) })
    }

    pub fn stats(&self) -> [u64; 3] {
        let mut values = [0; 3];
        unsafe { ffi::ci_factory_stats(self.core.handle.0.as_ptr(), values.as_mut_ptr()) };
        values
    }

    /// Copy readable NV12 planes into a frame owned by this factory.
    ///
    /// # Safety
    ///
    /// The planes must reside on this factory's CUDA device, have readable
    /// spans for the specified dimensions and pitches, and remain valid for
    /// the duration of this call. Any writes to them must already be complete.
    pub unsafe fn import_nv12(
        &self,
        planes: DeviceNv12Planes,
        width: usize,
        height: usize,
    ) -> Result<GpuFrame, Error> {
        validate_nv12_dimensions(width, height)?;
        if planes.y.is_null()
            || planes.uv.is_null()
            || !(width..=i32::MAX as usize).contains(&planes.y_pitch)
            || !(width..=i32::MAX as usize).contains(&planes.uv_pitch)
        {
            return Err(Error::Invalid("Invalid NV12 plane address or pitch".into()));
        }
        let handle = checked_handle(unsafe {
            ffi::ci_frame_import(
                self.core.handle.0.as_ptr(),
                planes.y,
                planes.uv,
                planes.y_pitch as i32,
                planes.uv_pitch as i32,
                width as i32,
                height as i32,
            )
        })?;
        Ok(GpuFrame {
            core: Arc::new(FrameCore {
                handle: Mutex::new(Some(ImageHandle(handle))),
                factory: Arc::clone(&self.core),
                width,
                height,
            }),
        })
    }
}

impl GpuFrame {
    pub fn width(&self) -> usize {
        self.core.width
    }

    pub fn height(&self) -> usize {
        self.core.height
    }

    pub fn device(&self) -> i32 {
        self.core.factory.device
    }

    pub fn copy(&self) -> Result<Self, Error> {
        let guard = self
            .core
            .handle
            .lock()
            .map_err(|_| Error::Runtime("GPU image lock poisoned".into()))?;
        let handle = guard
            .as_ref()
            .ok_or_else(|| Error::Runtime("GPU image is closed".into()))?;
        let copy = checked_handle(unsafe { ffi::ci_frame_copy(handle.0.as_ptr()) })?;
        Ok(Self {
            core: Arc::new(FrameCore {
                handle: Mutex::new(Some(ImageHandle(copy))),
                factory: Arc::clone(&self.core.factory),
                width: self.width(),
                height: self.height(),
            }),
        })
    }

    pub fn download_bgr(&self) -> Result<Vec<u8>, Error> {
        let guard = self
            .core
            .handle
            .lock()
            .map_err(|_| Error::Runtime("GPU image lock poisoned".into()))?;
        let handle = guard
            .as_ref()
            .ok_or_else(|| Error::Runtime("GPU image is closed".into()))?;
        let mut data = vec![0; self.height() * self.width() * 3];
        ffi::status(unsafe { ffi::ci_frame_download(handle.0.as_ptr(), data.as_mut_ptr()) })?;
        Ok(data)
    }

    pub fn blend_bgra(
        &self,
        bytes: &[u8],
        width: usize,
        height: usize,
        x: i32,
        y: i32,
    ) -> Result<(), Error> {
        if width == 0
            || height == 0
            || width
                .checked_mul(height)
                .is_none_or(|n| n > MAX_FRAME_PIXELS || n * 4 != bytes.len())
        {
            return Err(Error::Invalid("Invalid BGRA overlay size".into()));
        }
        let guard = self
            .core
            .handle
            .lock()
            .map_err(|_| Error::Runtime("GPU image lock poisoned".into()))?;
        let handle = guard
            .as_ref()
            .ok_or_else(|| Error::Runtime("GPU image is closed".into()))?;
        ffi::status(unsafe {
            ffi::ci_frame_blend(
                handle.0.as_ptr(),
                bytes.as_ptr(),
                width as i32,
                height as i32,
                x,
                y,
            )
        })
    }

    pub fn close(&self) -> Result<(), Error> {
        self.core
            .handle
            .lock()
            .map_err(|_| Error::Runtime("GPU image lock poisoned".into()))?
            .take();
        Ok(())
    }
}

struct EncoderHandle(NonNull<c_void>);
unsafe impl Send for EncoderHandle {}
impl Drop for EncoderHandle {
    fn drop(&mut self) {
        unsafe { ffi::ci_encoder_destroy(self.0.as_ptr()) };
    }
}

pub struct GpuEncoder {
    handle: Mutex<Option<EncoderHandle>>,
}

impl GpuEncoder {
    pub fn new(width: i32, height: i32, fps: i32, device: i32) -> Result<Self, Error> {
        let handle = checked_handle(unsafe { ffi::ci_encoder_create(width, height, fps, device) })?;
        Ok(Self {
            handle: Mutex::new(Some(EncoderHandle(handle))),
        })
    }

    pub fn encode(&self, frame: Option<&GpuFrame>) -> Result<Vec<u8>, Error> {
        let encoder = self
            .handle
            .lock()
            .map_err(|_| Error::Runtime("GPU encoder lock poisoned".into()))?;
        let encoder = encoder
            .as_ref()
            .ok_or_else(|| Error::Runtime("GPU encoder is closed".into()))?;
        let image = frame
            .map(|frame| {
                frame
                    .core
                    .handle
                    .lock()
                    .map_err(|_| Error::Runtime("GPU image lock poisoned".into()))
            })
            .transpose()?;
        let pointer = match image.as_ref() {
            Some(guard) => guard
                .as_ref()
                .ok_or_else(|| Error::Runtime("GPU image is closed".into()))?
                .0
                .as_ptr(),
            None => std::ptr::null_mut(),
        };
        let mut output = std::ptr::null();
        let mut length = 0;
        ffi::status(unsafe {
            ffi::ci_encoder_encode(encoder.0.as_ptr(), pointer, &mut output, &mut length)
        })?;
        // The output belongs to the encoder and is read while its lock is held.
        Ok(if length == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(output, length) }.to_vec()
        })
    }

    pub fn close(&self) -> Result<(), Error> {
        let encoder = self
            .handle
            .lock()
            .map_err(|_| Error::Runtime("GPU encoder lock poisoned".into()))?
            .take();
        drop(encoder);
        Ok(())
    }
}

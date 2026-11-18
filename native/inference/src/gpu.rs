//! DLPack input borrows are pinned until device-to-device import completes.
use crate::{core::Error, engine::ffi, python_error};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    ffi as pyffi,
    prelude::*,
    types::PyBytes,
};
use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

#[repr(C)]
struct Device {
    kind: i32,
    index: i32,
}
#[repr(C)]
struct Dtype {
    code: u8,
    bits: u8,
    lanes: u16,
}
#[repr(C)]
struct Tensor {
    data: *mut c_void,
    device: Device,
    ndim: i32,
    dtype: Dtype,
    shape: *mut i64,
    strides: *mut i64,
    offset: u64,
}
#[repr(C)]
struct ManagedTensor {
    tensor: Tensor,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut ManagedTensor)>,
}

struct FactoryHandle(NonNull<c_void>);
// The factory's GPU table and kernel module are immutable. Bridge operations
// select the correct device; Arc keeps resources alive through every image.
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

pub(crate) struct ImageHandle(pub NonNull<c_void>);
// Access to a handle is serialized by FrameCore's mutex. CUDA work finishes or
// owns a C++ shared image reference before that mutex is released.
unsafe impl Send for ImageHandle {}
impl Drop for ImageHandle {
    fn drop(&mut self) {
        unsafe { ffi::ci_frame_destroy(self.0.as_ptr()) };
    }
}
pub(crate) struct FrameCore {
    pub handle: Mutex<Option<ImageHandle>>,
    factory: Arc<FactoryCore>,
    pub h: usize,
    pub w: usize,
}
#[pyclass(module = "construction_inference", frozen)]
pub struct GpuFrame {
    pub(crate) core: Arc<FrameCore>,
}
#[pyclass(module = "construction_inference", frozen)]
pub struct GpuFrameFactory {
    core: Arc<FactoryCore>,
}

fn pointer_result(ptr: *mut c_void) -> PyResult<NonNull<c_void>> {
    if let Some(p) = NonNull::new(ptr) {
        return Ok(p);
    }
    ffi::status(unsafe { ffi::ci_error_code() }).map_err(python_error)?;
    Err(PyRuntimeError::new_err("GPU frame allocation failed"))
}
fn plane(
    obj: &Bound<'_, PyAny>,
    expected: &[usize],
    device: i32,
) -> PyResult<(Py<PyAny>, usize, usize)> {
    let capsule = obj.call_method0("__dlpack__")?;
    if unsafe { pyffi::PyCapsule_IsValid(capsule.as_ptr(), c"dltensor".as_ptr()) } != 1 {
        return Err(PyValueError::new_err(
            "Expected a legacy DLPack tensor capsule",
        ));
    }
    let managed = unsafe { pyffi::PyCapsule_GetPointer(capsule.as_ptr(), c"dltensor".as_ptr()) }
        .cast::<ManagedTensor>();
    // A conforming DLPack exporter owns valid descriptor arrays until capsule
    // destruction. We borrow the capsule, never rename or manually delete it.
    let tensor = unsafe { &(*managed).tensor };
    if tensor.device.kind != 2
        || tensor.device.index != device
        || tensor.dtype.code != 1
        || tensor.dtype.bits != 8
        || tensor.dtype.lanes != 1
        || tensor.ndim != expected.len() as i32
        || tensor.shape.is_null()
        || tensor.data.is_null()
    {
        return Err(PyValueError::new_err(
            "Expected uint8 NV12 planes on the selected CUDA device",
        ));
    }
    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, expected.len()) };
    if shape
        .iter()
        .zip(expected)
        .any(|(&a, &b)| a < 0 || a as usize != b)
    {
        return Err(PyValueError::new_err("NV12 plane shape mismatch"));
    }
    let pitch = if tensor.strides.is_null() {
        expected.iter().skip(1).product()
    } else {
        let strides = unsafe { std::slice::from_raw_parts(tensor.strides, expected.len()) };
        let valid = if expected.len() == 2 {
            strides[1] == 1
        } else {
            strides[1] == 2 && strides[2] == 1
        };
        if !valid || strides[0] < 0 {
            return Err(PyValueError::new_err("Invalid NV12 plane strides"));
        }
        strides[0] as usize
    };
    if pitch < expected.iter().skip(1).product()
        || pitch > i32::MAX as usize
        || tensor.offset > usize::MAX as u64
    {
        return Err(PyValueError::new_err("Invalid NV12 pitch or offset"));
    }
    let pointer = (tensor.data as usize)
        .checked_add(tensor.offset as usize)
        .ok_or_else(|| PyValueError::new_err("Device pointer overflow"))?;
    Ok((capsule.unbind(), pointer, pitch))
}
#[pymethods]
impl GpuFrameFactory {
    #[new]
    #[pyo3(signature=(lut,device=0))]
    fn new(py: Python<'_>, lut: Bound<'_, PyBytes>, device: i32) -> PyResult<Self> {
        if device < 0 || lut.as_bytes().len() != 256 * 256 * 256 * 3 {
            return Err(PyValueError::new_err(
                "Expected complete uint8 NV12 lookup and CUDA device",
            ));
        }
        let data = lut.as_bytes();
        let handle =
            py.detach(
                || unsafe { ffi::ci_factory_create(data.as_ptr(), data.len(), device) } as usize,
            );
        let handle = pointer_result(handle as *mut c_void)?;
        Ok(Self {
            core: Arc::new(FactoryCore {
                handle: FactoryHandle(handle),
                device,
            }),
        })
    }
    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| ffi::status(unsafe { ffi::ci_gpu_sync(self.core.device) }))
            .map_err(python_error)
    }
    #[getter]
    fn stats(&self) -> [u64; 3] {
        let mut values = [0; 3];
        unsafe { ffi::ci_factory_stats(self.core.handle.0.as_ptr(), values.as_mut_ptr()) };
        values
    }
    fn import_nv12(
        &self,
        py: Python<'_>,
        y: Bound<'_, PyAny>,
        uv: Bound<'_, PyAny>,
        width: usize,
        height: usize,
    ) -> PyResult<GpuFrame> {
        if width < 2
            || height < 2
            || width > 16384
            || height > 16384
            || width % 2 != 0
            || height % 2 != 0
            || width
                .checked_mul(height)
                .is_none_or(|n| n > 64 * 1024 * 1024)
        {
            return Err(PyValueError::new_err(
                "Expected even NV12 image at most 64M pixels",
            ));
        }
        // PyAV 17.1 exports CUDA planes without a producer-stream handshake.
        // Synchronize before export; no image pixels cross host memory.
        py.detach(|| ffi::status(unsafe { ffi::ci_gpu_sync(self.core.device) }))
            .map_err(python_error)?;
        let (_y, yp, ys) = plane(&y, &[height, width], self.core.device)?;
        let (_uv, up, us) = plane(&uv, &[height / 2, width / 2, 2], self.core.device)?;
        let factory = self.core.handle.0.as_ptr() as usize;
        let address = py.detach(|| unsafe {
            ffi::ci_frame_import(
                factory as *mut c_void,
                yp as *const u8,
                up as *const u8,
                ys as i32,
                us as i32,
                width as i32,
                height as i32,
            )
        } as usize);
        let handle = pointer_result(address as *mut c_void)?;
        Ok(GpuFrame {
            core: Arc::new(FrameCore {
                handle: Mutex::new(Some(ImageHandle(handle))),
                factory: self.core.clone(),
                h: height,
                w: width,
            }),
        })
    }
}
#[pymethods]
impl GpuFrame {
    #[getter]
    fn width(&self) -> usize {
        self.core.w
    }
    #[getter]
    fn height(&self) -> usize {
        self.core.h
    }
    #[getter]
    fn device(&self) -> i32 {
        self.core.factory.device
    }
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let address = py.detach(|| {
            let guard = self
                .core
                .handle
                .lock()
                .map_err(|_| PyRuntimeError::new_err("GPU image lock poisoned"))?;
            let handle = guard
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("GPU image is closed"))?;
            Ok::<_, PyErr>(unsafe { ffi::ci_frame_copy(handle.0.as_ptr()) } as usize)
        })?;
        let handle = pointer_result(address as *mut c_void)?;
        Ok(Self {
            core: Arc::new(FrameCore {
                handle: Mutex::new(Some(ImageHandle(handle))),
                factory: self.core.factory.clone(),
                h: self.core.h,
                w: self.core.w,
            }),
        })
    }
    fn download_bgr(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let data = py
            .detach(|| {
                let guard = self
                    .core
                    .handle
                    .lock()
                    .map_err(|_| Error::Runtime("GPU image lock poisoned".into()))?;
                let handle = guard
                    .as_ref()
                    .ok_or_else(|| Error::Runtime("GPU image is closed".into()))?;
                let mut data = vec![0; self.core.h * self.core.w * 3];
                ffi::status(unsafe {
                    ffi::ci_frame_download(handle.0.as_ptr(), data.as_mut_ptr())
                })?;
                Ok::<_, Error>(data)
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &data).unbind())
    }
    fn blend_bgra(
        &self,
        py: Python<'_>,
        data: Bound<'_, PyBytes>,
        width: usize,
        height: usize,
        x: i32,
        y: i32,
    ) -> PyResult<()> {
        let bytes = data.as_bytes();
        if width == 0
            || height == 0
            || width
                .checked_mul(height)
                .is_none_or(|n| n > 64 * 1024 * 1024)
            || width.checked_mul(height).and_then(|n| n.checked_mul(4)) != Some(bytes.len())
        {
            return Err(PyValueError::new_err("Invalid BGRA overlay size"));
        }
        py.detach(|| {
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
        })
        .map_err(python_error)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.core
                .handle
                .lock()
                .map_err(|_| PyRuntimeError::new_err("GPU image lock poisoned"))?
                .take();
            Ok(())
        })
    }
}

struct EncoderHandle(NonNull<c_void>);
unsafe impl Send for EncoderHandle {}
impl Drop for EncoderHandle {
    fn drop(&mut self) {
        unsafe { ffi::ci_encoder_destroy(self.0.as_ptr()) };
    }
}
#[pyclass(module = "construction_inference")]
pub struct GpuEncoder {
    handle: Mutex<Option<EncoderHandle>>,
    lease: Mutex<Option<Py<PyAny>>>,
}
#[pymethods]
impl GpuEncoder {
    #[new]
    #[pyo3(signature=(width,height,fps=15,device=0,lease=None))]
    fn new(
        py: Python<'_>,
        width: i32,
        height: i32,
        fps: i32,
        device: i32,
        lease: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let address =
            py.detach(|| unsafe { ffi::ci_encoder_create(width, height, fps, device) } as usize);
        Ok(Self {
            handle: Mutex::new(Some(EncoderHandle(pointer_result(address as *mut c_void)?))),
            lease: Mutex::new(lease),
        })
    }
    #[pyo3(signature=(frame=None))]
    fn encode(&self, py: Python<'_>, frame: Option<Py<GpuFrame>>) -> PyResult<Py<PyBytes>> {
        let owner = frame.as_ref().map(|f| f.get().core.clone());
        let data = py
            .detach(|| {
                let encoder = self
                    .handle
                    .lock()
                    .map_err(|_| Error::Runtime("GPU encoder lock poisoned".into()))?;
                let encoder = encoder
                    .as_ref()
                    .ok_or_else(|| Error::Runtime("GPU encoder is closed".into()))?;
                let image = owner
                    .as_ref()
                    .map(|f| {
                        f.handle
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
                // Bitstream bytes, not an uncompressed frame, cross to the host.
                Ok::<_, Error>(if length == 0 {
                    Vec::new()
                } else {
                    unsafe { std::slice::from_raw_parts(output, length) }.to_vec()
                })
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &data).unbind())
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let encoder = self
                .handle
                .lock()
                .map_err(|_| PyRuntimeError::new_err("GPU encoder lock poisoned"))?
                .take();
            drop(encoder);
            self.lease
                .lock()
                .map_err(|_| PyRuntimeError::new_err("GPU encoder lease lock poisoned"))?
                .take();
            Ok(())
        })
    }
}

//! DLPack input borrows are pinned until device-to-device import completes.

use crate::python_error;
use native_inference::gpu as native_gpu;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    ffi as pyffi,
    prelude::*,
    types::PyBytes,
};
use std::{ffi::c_void, sync::Mutex};

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

#[pyclass(module = "construction_inference", frozen)]
pub(crate) struct GpuFrame {
    pub(crate) frame: native_gpu::GpuFrame,
}

#[pyclass(module = "construction_inference", frozen)]
pub(crate) struct GpuFrameFactory {
    factory: native_gpu::GpuFrameFactory,
}

#[pymethods]
impl GpuFrameFactory {
    #[new]
    #[pyo3(signature=(lut,device=0))]
    fn new(py: Python<'_>, lut: Bound<'_, PyBytes>, device: i32) -> PyResult<Self> {
        let data = lut.as_bytes();
        let factory = py
            .detach(|| native_gpu::GpuFrameFactory::new(data, device))
            .map_err(python_error)?;
        Ok(Self { factory })
    }

    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.factory.synchronize())
            .map_err(python_error)
    }

    #[getter]
    fn stats(&self) -> [u64; 3] {
        self.factory.stats()
    }

    fn import_nv12(
        &self,
        py: Python<'_>,
        y: Bound<'_, PyAny>,
        uv: Bound<'_, PyAny>,
        width: usize,
        height: usize,
    ) -> PyResult<GpuFrame> {
        native_gpu::validate_nv12_dimensions(width, height).map_err(python_error)?;
        // PyAV 17.1 exports CUDA planes without a producer-stream handshake.
        self.synchronize(py)?;
        let device = self.factory.device();
        let (_y, y_addr, y_pitch) = plane(&y, &[height, width], device)?;
        let (_uv, uv_addr, uv_pitch) = plane(&uv, &[height / 2, width / 2, 2], device)?;
        let frame = py
            .detach(|| {
                // The capsule owners above keep the validated CUDA planes
                // alive until the native copy returns.
                unsafe {
                    self.factory.import_nv12(
                        native_gpu::DeviceNv12Planes {
                            y: y_addr as *const u8,
                            uv: uv_addr as *const u8,
                            y_pitch,
                            uv_pitch,
                        },
                        width,
                        height,
                    )
                }
            })
            .map_err(python_error)?;
        Ok(GpuFrame { frame })
    }
}

#[pymethods]
impl GpuFrame {
    #[getter]
    fn width(&self) -> usize {
        self.frame.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.frame.height()
    }

    #[getter]
    fn device(&self) -> i32 {
        self.frame.device()
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let frame = py.detach(|| self.frame.copy()).map_err(python_error)?;
        Ok(Self { frame })
    }

    fn download_bgr(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let data = py
            .detach(|| self.frame.download_bgr())
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
        py.detach(|| self.frame.blend_bgra(bytes, width, height, x, y))
            .map_err(python_error)
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.frame.close()).map_err(python_error)
    }
}

#[pyclass(module = "construction_inference")]
pub(crate) struct GpuEncoder {
    encoder: native_gpu::GpuEncoder,
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
        let encoder = py
            .detach(|| native_gpu::GpuEncoder::new(width, height, fps, device))
            .map_err(python_error)?;
        Ok(Self {
            encoder,
            lease: Mutex::new(lease),
        })
    }

    #[pyo3(signature=(frame=None))]
    fn encode(&self, py: Python<'_>, frame: Option<Py<GpuFrame>>) -> PyResult<Py<PyBytes>> {
        let frame = frame.as_ref().map(|frame| frame.get().frame.clone());
        let data = py
            .detach(|| self.encoder.encode(frame.as_ref()))
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &data).unbind())
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.encoder.close().map_err(python_error)?;
            self.lease
                .lock()
                .map_err(|_| PyRuntimeError::new_err("GPU encoder lease lock poisoned"))?
                .take();
            Ok(())
        })
    }
}

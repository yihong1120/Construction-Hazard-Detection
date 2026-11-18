use crate::core::{Error, Image};
use crate::engine::{self, BufferImage, NativeEngine};
#[cfg(feature = "tensorrt")]
use crate::gpu;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::sync::{Mutex, MutexGuard};

pyo3::create_exception!(construction_inference, UnsupportedEngineError, PyValueError);

pub(crate) fn python_error(error: Error) -> PyErr {
    match error {
        Error::Invalid(message) => PyValueError::new_err(message),
        Error::Unsupported(message) => UnsupportedEngineError::new_err(message),
        Error::Runtime(message) => PyRuntimeError::new_err(message),
    }
}

#[pyclass(module = "construction_inference")]
struct TensorRTEngine {
    engine: Mutex<Option<NativeEngine>>,
}

#[pymethods]
impl TensorRTEngine {
    #[new]
    #[pyo3(signature=(path,max_batch=8,imgsz=640,device=0,rect=false,confidence=0.25,asynchronous=false))]
    fn new(
        py: Python<'_>,
        path: String,
        max_batch: usize,
        imgsz: usize,
        device: i32,
        rect: bool,
        confidence: f32,
        asynchronous: bool,
    ) -> PyResult<Self> {
        let engine = py
            .detach(|| {
                NativeEngine::new(
                    &path,
                    max_batch,
                    imgsz,
                    device,
                    rect,
                    confidence,
                    asynchronous,
                )
            })
            .map_err(python_error)?;
        Ok(Self {
            engine: Mutex::new(Some(engine)),
        })
    }
    #[getter]
    fn max_batch(&self) -> PyResult<usize> {
        Ok(self
            .lock()?
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
            .max_batch)
    }
    #[getter]
    fn activation_bytes(&self) -> PyResult<usize> {
        Ok(self
            .lock()?
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
            .activation_bytes())
    }

    #[getter]
    fn workspace_bytes(&self) -> PyResult<usize> {
        Ok(self
            .lock()?
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
            .workspace_bytes())
    }
    #[getter]
    fn workspace_growths(&self) -> PyResult<usize> {
        Ok(self
            .lock()?
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
            .workspace_growths)
    }

    fn predict(
        &self,
        py: Python<'_>,
        frames: Vec<(usize, usize, Bound<'_, PyBytes>)>,
    ) -> PyResult<Py<PyList>> {
        let images: Vec<_> = frames
            .iter()
            .map(|(h, w, data)| Image {
                h: *h,
                w: *w,
                data: data.as_bytes(),
            })
            .collect();
        let results = py
            .detach(|| {
                let mut guard = self
                    .engine
                    .lock()
                    .map_err(|_| Error::Runtime("Native engine lock poisoned".into()))?;
                let engine = guard
                    .as_mut()
                    .ok_or_else(|| Error::Runtime("Engine is closed".into()))?;
                engine.predict(&images)
            })
            .map_err(python_error)?;
        encode(py, results)
    }

    /// For exclusively leased worker frames. Read-only views do not prevent
    /// another process from writing: callers must retain the shared-ring lease.
    fn predict_buffers(
        &self,
        py: Python<'_>,
        frames: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyList>> {
        let prepared = self.prepare_buffers(&frames)?;
        let results = py
            .detach(|| {
                let mut guard = self
                    .engine
                    .lock()
                    .map_err(|_| Error::Runtime("Native engine lock poisoned".into()))?;
                guard
                    .as_mut()
                    .ok_or_else(|| Error::Runtime("Engine is closed".into()))?
                    .run_prepared(prepared)
            })
            .map_err(python_error)?;
        encode(py, results)
    }
    #[cfg(feature = "tensorrt")]
    fn submit_device_frames(
        &self,
        py: Python<'_>,
        frames: Vec<Py<gpu::GpuFrame>>,
    ) -> PyResult<u64> {
        let owners: Vec<_> = frames.iter().map(|f| f.get().core.clone()).collect();
        py.detach(|| {
            self.lock()?
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
                .submit_device(&owners)
                .map_err(python_error)
        })
    }
    #[cfg(feature = "tensorrt")]
    fn predict_device_frames(
        &self,
        py: Python<'_>,
        frames: Vec<Py<gpu::GpuFrame>>,
    ) -> PyResult<Py<PyList>> {
        let owners: Vec<_> = frames.iter().map(|f| f.get().core.clone()).collect();
        let rows = py.detach(|| {
            let mut guard = self.lock()?;
            let engine = guard
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?;
            let ticket = engine.submit_device(&owners).map_err(python_error)?;
            engine.collect(ticket).map_err(python_error)
        })?;
        encode(py, rows)
    }
    fn submit(
        &self,
        py: Python<'_>,
        frames: Vec<(usize, usize, Bound<'_, PyBytes>)>,
    ) -> PyResult<u64> {
        let images: Vec<_> = frames
            .iter()
            .map(|(h, w, data)| Image {
                h: *h,
                w: *w,
                data: data.as_bytes(),
            })
            .collect();
        py.detach(|| {
            self.lock()?
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
                .submit_images(&images)
                .map_err(python_error)
        })
    }
    fn submit_buffers(&self, py: Python<'_>, frames: Vec<Bound<'_, PyAny>>) -> PyResult<u64> {
        let prepared = self.prepare_buffers(&frames)?;
        py.detach(|| {
            self.lock()?
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
                .submit_prepared(prepared)
                .map_err(python_error)
        })
    }
    fn collect(&self, py: Python<'_>, ticket: u64) -> PyResult<Py<PyList>> {
        let rows = py.detach(|| {
            self.lock()?
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
                .collect(ticket)
                .map_err(python_error)
        })?;
        encode(py, rows)
    }
    #[getter]
    fn pending_count(&self) -> PyResult<usize> {
        Ok(self
            .lock()?
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
            .pending_count())
    }
    #[getter]
    fn pinned_bytes(&self) -> PyResult<usize> {
        Ok(self
            .lock()?
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
            .pinned_bytes())
    }
    #[getter]
    fn async_stats(&self) -> PyResult<[u64; 3]> {
        Ok(self
            .lock()?
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?
            .async_stats())
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let engine = self
                .engine
                .lock()
                .map_err(|_| Error::Runtime("Native engine lock poisoned".into()))?
                .take();
            drop(engine);
            Ok(())
        })
        .map_err(python_error)
    }
}

fn encode(py: Python<'_>, results: Vec<Vec<[f32; 6]>>) -> PyResult<Py<PyList>> {
    let output = PyList::empty(py);
    for rows in results {
        let batch = PyList::empty(py);
        for row in rows {
            let detection = PyList::new(py, &row[..5])?;
            detection.append(row[5] as u32)?;
            batch.append(detection)?;
        }
        output.append(batch)?;
    }
    Ok(output.unbind())
}
impl TensorRTEngine {
    fn prepare_buffers(&self, frames: &[Bound<'_, PyAny>]) -> PyResult<engine::PreparedBatch> {
        let buffers: Vec<_> = frames
            .iter()
            .map(PyBuffer::<u8>::get)
            .collect::<PyResult<_>>()?;
        let mut views = Vec::with_capacity(buffers.len());
        for buffer in &buffers {
            if !buffer.readonly()
                || !buffer.is_c_contiguous()
                || buffer.dimensions() != 3
                || buffer.shape()[2] != 3
                || buffer.suboffsets().is_some()
            {
                return Err(PyValueError::new_err(
                    "Expected read-only contiguous uint8 HWC BGR buffers",
                ));
            }
            views.push(BufferImage {
                h: buffer.shape()[0],
                w: buffer.shape()[1],
                bytes: buffer.len_bytes(),
                pointer: buffer.buf_ptr().cast(),
            });
        }
        let prepared = {
            let mut guard = self.lock()?;
            let engine = guard
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Engine is closed"))?;
            // SAFETY: exports pin storage; the GIL remains held while OpenCV
            // reads it. The worker owns the producer lease until its reply.
            unsafe { engine.prepare_buffers(&views) }.map_err(python_error)?
        };
        drop(views);
        drop(buffers);
        Ok(prepared)
    }
    fn lock(&self) -> PyResult<MutexGuard<'_, Option<NativeEngine>>> {
        self.engine
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Native engine lock poisoned"))
    }
}

#[pymodule]
fn construction_inference(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ABI_VERSION", 1)?;
    module.add("BUFFER_API_VERSION", 1)?;
    module.add("ASYNC_API_VERSION", 1)?;
    module.add("HAS_GPU_ENCODER", cfg!(feature = "gpu-media"))?;
    module.add("HAS_TENSORRT", cfg!(feature = "tensorrt"))?;
    module.add("TENSORRT_HEADER_VERSION", engine::header_version())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add(
        "UnsupportedEngineError",
        module.py().get_type::<UnsupportedEngineError>(),
    )?;
    module.add_class::<TensorRTEngine>()?;
    #[cfg(feature = "tensorrt")]
    {
        module.add_class::<gpu::GpuFrameFactory>()?;
        module.add_class::<gpu::GpuFrame>()?;
        module.add_class::<gpu::GpuEncoder>()?;
    }
    module.add(
        "GPU_FRAME_API_VERSION",
        i32::from(cfg!(feature = "tensorrt")),
    )?;
    Ok(())
}

//! Safe Python bindings; numeric work runs detached from the interpreter.
mod capture;
mod core;
mod kernels;
mod resources;

use core::{Detection, SafetyRow, Tracker, Warnings};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass(module = "construction_postprocess")]
#[derive(Default)]
struct CentroidTracker {
    core: Tracker,
}

#[pymethods]
impl CentroidTracker {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn track(
        &mut self,
        py: Python<'_>,
        detections: Vec<Detection>,
        frame_count: u64,
        movement_thr_sq: f64,
        max_id_keep: u64,
    ) -> PyResult<Py<PyList>> {
        if !movement_thr_sq.is_finite() || movement_thr_sq < 0.0 {
            return Err(PyValueError::new_err(
                "movement_thr_sq must be finite and non-negative",
            ));
        }
        if detections.iter().flatten().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "detections must contain finite numbers",
            ));
        }
        let tracks = py.detach(|| {
            self.core
                .track(&detections, frame_count, movement_thr_sq, max_id_keep)
        });
        let result = PyList::empty(py);
        for track in tracks {
            let row = PyList::new(py, track.detection)?;
            row.append(track.id)?;
            row.append(track.moving)?;
            result.append(row)?;
        }
        Ok(result.unbind())
    }
}

fn read_rows(datas: &Bound<'_, PyList>, proximity: bool) -> PyResult<Vec<SafetyRow>> {
    let mut rows = Vec::with_capacity(datas.len());
    for row in datas.iter() {
        if row.len()? < 6 {
            return Err(PyValueError::new_err("each row needs at least six values"));
        }
        let mut bbox = [0.0f64; 4];
        for (index, value) in bbox.iter_mut().enumerate() {
            *value = row.get_item(index)?.extract()?;
        }
        let class: f64 = row.get_item(5)?.extract()?;
        if !class.is_finite() || bbox.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "coordinates and labels must be finite",
            ));
        }
        if bbox[0] > bbox[2] || bbox[1] > bbox[3] {
            return Err(PyValueError::new_err(
                "safety_warnings expects normalized boxes",
            ));
        }
        let track_id = if proximity && class == 5.0 && row.len()? >= 7 {
            let value = row.get_item(6)?;
            if value.is_none() || value.eq("")? || value.eq(-1)? {
                None
            } else {
                Some(value.str()?.to_str()?.to_owned())
            }
        } else {
            None
        };
        rows.push(SafetyRow {
            bbox,
            class,
            track_id,
        });
    }
    Ok(rows)
}

fn encode_warnings(py: Python<'_>, rows: &[SafetyRow], warnings: Warnings) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    for (label, count) in [
        ("warning_no_hardhat", warnings.no_hardhat),
        ("warning_no_safety_vest", warnings.no_vest),
    ] {
        if count > 0 {
            let warning = PyDict::new(py);
            warning.set_item("count", count)?;
            result.set_item(label, warning)?;
        }
    }
    for (label, indices) in [
        ("warning_close_to_machinery", warnings.machinery),
        ("warning_close_to_vehicle", warnings.vehicle),
    ] {
        if indices.is_empty() {
            continue;
        }
        let warning = PyDict::new(py);
        let boxes = PyList::empty(py);
        let ids = PyList::empty(py);
        for &index in &indices {
            boxes.append(PyList::new(py, rows[index].bbox)?)?;
            if let Some(id) = &rows[index].track_id {
                ids.append(id)?;
            }
        }
        warning.set_item("count", indices.len())?;
        warning.set_item("person_bboxes", boxes)?;
        warning.set_item("person_track_ids", ids)?;
        result.set_item(label, warning)?;
    }
    Ok(result.unbind())
}

#[pyfunction]
fn safety_warnings(
    py: Python<'_>,
    datas: &Bound<'_, PyList>,
    detect_safety: bool,
    detect_proximity: bool,
) -> PyResult<Py<PyDict>> {
    if !detect_safety && !detect_proximity {
        return Ok(PyDict::new(py).unbind());
    }
    let rows = read_rows(datas, detect_proximity)?;
    let warnings = py.detach(|| core::safety(&rows, detect_safety, detect_proximity));
    encode_warnings(py, &rows, warnings)
}

#[pymodule]
fn construction_postprocess(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CentroidTracker>()?;
    module.add_function(wrap_pyfunction!(safety_warnings, module)?)?;
    module.add_function(wrap_pyfunction!(kernels::mst_edges, module)?)?;
    module.add_function(wrap_pyfunction!(kernels::blend_bgra, module)?)?;
    module.add_function(wrap_pyfunction!(kernels::blend_mask_lut, module)?)?;
    module.add("GEOMETRY_API_VERSION", 1)?;
    module.add("OVERLAY_API_VERSION", 1)?;
    module.add_class::<capture::LatestCapture>()?;
    module.add_class::<capture::CapturedFrame>()?;
    module.add("CAPTURE_API_VERSION", 1)?;
    module.add_class::<resources::ResourceLease>()?;
    module.add_function(wrap_pyfunction!(resources::acquire_resource, module)?)?;
    module.add("RESOURCE_API_VERSION", 1)?;
    module.add("ABI_VERSION", 1)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

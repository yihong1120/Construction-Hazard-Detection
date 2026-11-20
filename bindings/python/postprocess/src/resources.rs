//! File-description locks survive parent exit when passed to an encoder child.
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::Path,
    sync::Mutex,
};

#[pyclass(module = "construction_postprocess")]
pub struct ResourceLease {
    files: Mutex<Vec<File>>,
}
#[pymethods]
impl ResourceLease {
    fn filenos(&self) -> PyResult<Vec<i32>> {
        Ok(self
            .files
            .lock()
            .map_err(error)?
            .iter()
            .map(AsRawFd::as_raw_fd)
            .collect())
    }
    fn close(&self) -> PyResult<()> {
        self.files.lock().map_err(error)?.clear();
        Ok(())
    }
}
fn error(value: impl ToString) -> PyErr {
    PyRuntimeError::new_err(value.to_string())
}
fn open(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
}
fn try_lock(file: &File) -> std::io::Result<bool> {
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let err = std::io::Error::last_os_error();
    if err.kind() == std::io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(err)
    }
}
#[pyfunction]
pub fn acquire_resource(
    py: Python<'_>,
    directory: String,
    key: String,
    capacity: usize,
    units: usize,
) -> PyResult<Option<ResourceLease>> {
    if key.is_empty()
        || !key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        || units == 0
        || units > capacity
        || capacity > 16384
    {
        return Err(PyValueError::new_err(
            "Invalid GPU resource capacity, key or units",
        ));
    }
    let _ = py;
    (|| {
        let root = Path::new(&directory);
        std::fs::create_dir_all(root).map_err(error)?;
        let mut control = open(&root.join(format!("{key}.control"))).map_err(error)?;
        if unsafe { libc::flock(control.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(error(std::io::Error::last_os_error()));
        }
        let mut text = String::new();
        Read::by_ref(&mut control)
            .take(32)
            .read_to_string(&mut text)
            .map_err(error)?;
        let old = if text.trim().is_empty() {
            capacity
        } else {
            text.trim().parse::<usize>().map_err(error)?
        };
        if old == 0 || old > 16384 {
            return Err(error("Corrupt GPU resource capacity"));
        }
        let mut files = Vec::new();
        if old != capacity {
            for index in 0..old {
                let file = open(&root.join(format!("{key}.{index}"))).map_err(error)?;
                if !try_lock(&file).map_err(error)? {
                    return Err(error("GPU budget differs from active processes"));
                }
                files.push(file);
            }
            files.clear();
        }
        control.seek(SeekFrom::Start(0)).map_err(error)?;
        control.set_len(0).map_err(error)?;
        write!(control, "{capacity}").map_err(error)?;
        for index in 0..capacity {
            let file = open(&root.join(format!("{key}.{index}"))).map_err(error)?;
            if try_lock(&file).map_err(error)? {
                files.push(file);
                if files.len() == units {
                    return Ok(Some(ResourceLease {
                        files: Mutex::new(files),
                    }));
                }
            }
        }
        Ok(None)
    })()
}

//! FFmpeg process and one-frame mailbox. Exported frame owners are immutable.
use pyo3::{
    exceptions::{PyBufferError, PyRuntimeError, PyValueError},
    ffi,
    prelude::*,
};
use std::{
    ffi::c_int,
    io::Read,
    process::{Child, Command, Stdio},
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Default)]
struct State {
    latest: Option<Arc<Vec<u8>>>,
    received: u64,
    dropped: u64,
    ended: bool,
}

#[pyclass(module = "construction_postprocess", frozen)]
pub struct CapturedFrame {
    pixels: Arc<Vec<u8>>,
}
#[pymethods]
impl CapturedFrame {
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() || flags & ffi::PyBUF_WRITABLE != 0 {
            return Err(PyBufferError::new_err("Captured frame is read-only"));
        }
        unsafe {
            (*view).buf = slf.pixels.as_ptr() as *mut _;
            (*view).len = slf.pixels.len() as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).format = c"B".as_ptr() as *mut _;
            (*view).ndim = 1;
            (*view).shape = &mut (*view).len;
            (*view).strides = &mut (*view).itemsize;
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            (*view).obj = slf.into_ptr();
        }
        Ok(())
    }
}

#[pyclass(module = "construction_postprocess")]
pub struct LatestCapture {
    child: Mutex<Option<Child>>,
    reader: Mutex<Option<JoinHandle<()>>>,
    state: Arc<(Mutex<State>, Condvar)>,
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
    #[pyo3(get)]
    fps: f64,
}
fn failure(message: impl ToString) -> PyErr {
    PyRuntimeError::new_err(message.to_string())
}

fn probe(binary: &str, url: &str, timeout: f64) -> PyResult<(usize, usize, f64)> {
    let timeout_option = if url.starts_with("rtsp://") || url.starts_with("rtsps://") {
        "-timeout"
    } else {
        "-rw_timeout"
    };
    let mut command = Command::new(binary);
    command.args([
        "-v",
        "error",
        timeout_option,
        &format!("{}", (timeout * 1e6) as u64),
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=p=0",
    ]);
    if url.starts_with("rtsp://") || url.starts_with("rtsps://") {
        command.args(["-rtsp_transport", "tcp"]);
    }
    let mut child = command
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| failure("Cannot start ffprobe"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| failure("Missing ffprobe output"))?;
    // Read concurrently so a malformed probe response cannot block on its pipe.
    let output = thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout.take(4096).read_to_string(&mut text);
        text
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status.success()),
            Ok(None) if started.elapsed().as_secs_f64() < timeout => {
                thread::sleep(Duration::from_millis(10))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(failure("ffprobe failed or timed out"));
            }
        }
    };
    let text = output
        .join()
        .map_err(|_| failure("ffprobe reader failed"))?;
    if !status? {
        return Err(failure("ffprobe could not inspect source"));
    }
    let fields: Vec<_> = text
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .split(',')
        .collect();
    let invalid = || failure("Invalid source dimensions or frame rate");
    let w = fields
        .first()
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(invalid)?;
    let h = fields
        .get(1)
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(invalid)?;
    let fps = fields
        .get(2)
        .and_then(|v| v.split_once('/'))
        .and_then(|(n, d)| Some(n.parse::<f64>().ok()? / d.parse::<f64>().ok()?))
        .unwrap_or(0.0);
    if w == 0
        || h == 0
        || w.checked_mul(h)
            .is_none_or(|pixels| pixels > 64 * 1024 * 1024)
    {
        return Err(invalid());
    }
    Ok((
        w,
        h,
        if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            0.0
        },
    ))
}

#[pymethods]
impl LatestCapture {
    #[new]
    #[pyo3(signature=(url,ffmpeg="ffmpeg",ffprobe="ffprobe",timeout=5.0,cuda=false,realtime=false))]
    fn new(
        py: Python<'_>,
        url: String,
        ffmpeg: &str,
        ffprobe: &str,
        timeout: f64,
        cuda: bool,
        realtime: bool,
    ) -> PyResult<Self> {
        if !timeout.is_finite() || timeout <= 0.0 || timeout > 300.0 || url.is_empty() {
            return Err(PyValueError::new_err("Invalid capture source or timeout"));
        }
        let (width, height, fps) = py.detach(|| probe(ffprobe, &url, timeout))?;
        let timeout_option = if url.starts_with("rtsp://") || url.starts_with("rtsps://") {
            "-timeout"
        } else {
            "-rw_timeout"
        };
        let mut command = Command::new(ffmpeg);
        command.args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            timeout_option,
            &format!("{}", (timeout * 1e6) as u64),
        ]);
        if realtime {
            command.arg("-re");
        }
        if url.starts_with("rtsp://") || url.starts_with("rtsps://") {
            command.args(["-rtsp_transport", "tcp"]);
        }
        if cuda {
            command.args(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]);
        }
        command.args(["-i", &url, "-map", "0:v:0", "-an", "-sn", "-dn"]);
        let filter = if cuda {
            format!("hwdownload,format=nv12,scale={width}:{height}")
        } else {
            format!("scale={width}:{height}")
        };
        command.args(["-vf", &filter, "-vsync", "0"]);
        command.args([
            "-threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        ]);
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| failure("Cannot start capture ffmpeg"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| failure("Missing decoder output"))?;
        let state = Arc::new((Mutex::new(State::default()), Condvar::new()));
        let producer = state.clone();
        let reader = thread::spawn(move || {
            let mut pixels = vec![0; width * height * 3];
            loop {
                if stdout.read_exact(&mut pixels).is_err() {
                    break;
                }
                let (lock, ready) = &*producer;
                let Ok(mut state) = lock.lock() else {
                    return;
                };
                if state.ended {
                    break;
                }
                let previous = state.latest.replace(Arc::new(pixels));
                state.received += 1;
                if previous.is_some() {
                    state.dropped += 1;
                }
                ready.notify_one();
                drop(state);
                pixels = previous
                    .and_then(|value| Arc::try_unwrap(value).ok())
                    .unwrap_or_else(|| vec![0; width * height * 3]);
            }
            if let Ok(mut state) = producer.0.lock() {
                state.ended = true;
                producer.1.notify_all();
            }
        });
        Ok(Self {
            child: Mutex::new(Some(child)),
            reader: Mutex::new(Some(reader)),
            state,
            width,
            height,
            fps,
        })
    }
    fn read(&self, py: Python<'_>, timeout: f64) -> PyResult<Option<CapturedFrame>> {
        if !timeout.is_finite() || timeout <= 0.0 || timeout > 300.0 {
            return Err(PyValueError::new_err("Invalid read timeout"));
        }
        let value = py.detach(|| {
            let state = self.state.0.lock().map_err(failure)?;
            let (mut state, _) = self
                .state
                .1
                .wait_timeout_while(state, Duration::from_secs_f64(timeout), |s| {
                    s.latest.is_none() && !s.ended
                })
                .map_err(failure)?;
            Ok::<_, PyErr>(state.latest.take())
        })?;
        Ok(value.map(|pixels| CapturedFrame { pixels }))
    }
    #[getter]
    fn counters(&self) -> PyResult<(u64, u64)> {
        let state = self.state.0.lock().map_err(failure)?;
        Ok((state.received, state.dropped))
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.shutdown())
    }
}
impl LatestCapture {
    fn shutdown(&self) -> PyResult<()> {
        {
            let mut state = self.state.0.lock().map_err(failure)?;
            state.ended = true;
            state.latest = None;
            self.state.1.notify_all();
        }
        if let Some(mut child) = self.child.lock().map_err(failure)?.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(reader) = self.reader.lock().map_err(failure)?.take() {
            let _ = reader.join();
        }
        Ok(())
    }
}
impl Drop for LatestCapture {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

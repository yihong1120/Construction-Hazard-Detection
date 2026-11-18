//! Rust worker prototype. TensorRT and OpenCV operations use a small C++ ABI;
//! Rust owns batching, letterbox dimensions, result filtering and measurement.
use serde::Deserialize;
use serde_json::{Value, json};
use std::ffi::{CStr, c_char, c_void};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

unsafe extern "C" {
    fn chd_error() -> *const c_char;
    fn chd_create(data: *const u8, size: usize, lut: *const f32, capacity: usize) -> *mut c_void;
    fn chd_destroy(engine: *mut c_void);
    fn chd_load(engine: *mut c_void, tensor: *const f32, batch: i32, h: i32, w: i32) -> i32;
    fn chd_run(engine: *mut c_void) -> i32;
    fn chd_detect(
        engine: *mut c_void,
        bytes: *const u8,
        result: *mut f32,
        batch: i32,
        h: i32,
        w: i32,
    ) -> i32;
    fn chd_input(engine: *mut c_void, result: *mut f32, elements: usize) -> i32;
    fn chd_letterbox(
        source: *const u8,
        h: i32,
        w: i32,
        output: *mut u8,
        rh: i32,
        rw: i32,
        th: i32,
        tw: i32,
        top: i32,
        left: i32,
    ) -> i32;
}

fn status(code: i32) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        // chd_error returns a thread-local, null-terminated message.
        Err(unsafe { CStr::from_ptr(chd_error()) }
            .to_string_lossy()
            .into_owned())
    }
}
struct Engine(*mut c_void);
impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { chd_destroy(self.0) };
    }
}

#[derive(Deserialize)]
struct FrameInfo {
    file: String,
    height: usize,
    width: usize,
}
#[derive(Deserialize)]
struct Manifest {
    frames: Vec<FrameInfo>,
    imgsz: usize,
    stride: usize,
}
struct Frame {
    bytes: Vec<u8>,
    height: usize,
    width: usize,
}
#[derive(Clone)]
struct Layout {
    h: usize,
    w: usize,
    rh: usize,
    rw: usize,
    top: usize,
    left: usize,
}
struct Batch {
    start: usize,
    stop: usize,
    layouts: Vec<Layout>,
}

fn layout(frame: &Frame, size: usize, stride: usize, auto: bool) -> Layout {
    let gain = (size as f64 / frame.height as f64).min(size as f64 / frame.width as f64);
    let rh = (frame.height as f64 * gain).round_ties_even() as usize;
    let rw = (frame.width as f64 * gain).round_ties_even() as usize;
    let dh = if auto {
        (size - rh) % stride
    } else {
        size - rh
    };
    let dw = if auto {
        (size - rw) % stride
    } else {
        size - rw
    };
    Layout {
        h: rh + dh,
        w: rw + dw,
        rh,
        rw,
        top: (dh as f64 * 0.5 - 0.1).round_ties_even() as usize,
        left: (dw as f64 * 0.5 - 0.1).round_ties_even() as usize,
    }
}

fn floats(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() % 4 != 0 {
        return Err("Invalid float32 file".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

struct Worker {
    engine: Engine,
    frames: Vec<Frame>,
    batches: Vec<Batch>,
    data: PathBuf,
}
impl Worker {
    fn new(manifest: &Path, model: &Path, batch_size: usize) -> Result<Self, String> {
        if !(1..=16).contains(&batch_size) {
            return Err("Batch must be 1..16".into());
        }
        let metadata: Manifest =
            serde_json::from_slice(&std::fs::read(manifest).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let data = manifest.parent().unwrap().to_path_buf();
        let lut = floats(&data.join("normalize.f32"))?;
        if lut.len() != 256 {
            return Err("Normalization LUT requires 256 values".into());
        }
        let mut plan = std::fs::read(model).map_err(|e| e.to_string())?;
        // Ultralytics prepends [u32 JSON length][metadata] to the TensorRT plan.
        if plan.len() < 4 {
            return Err("Truncated engine".into());
        }
        let size = u32::from_le_bytes(plan[..4].try_into().unwrap()) as usize;
        if size > plan.len() - 4 {
            return Err("Invalid metadata length".into());
        }
        let _: Value = serde_json::from_slice(&plan[4..4 + size]).map_err(|e| e.to_string())?;
        let pointer = unsafe {
            chd_create(
                plan[4 + size..].as_ptr(),
                plan.len() - 4 - size,
                lut.as_ptr(),
                batch_size * 3 * metadata.imgsz * metadata.imgsz,
            )
        };
        if pointer.is_null() {
            status(-1)?;
        }
        let engine = Engine(pointer);
        plan.clear();
        let mut frames = Vec::new();
        for info in metadata.frames {
            let bytes = std::fs::read(data.join(info.file)).map_err(|e| e.to_string())?;
            if info.width == 0 || info.height == 0 || bytes.len() != info.width * info.height * 3 {
                return Err("Invalid BGR frame".into());
            }
            frames.push(Frame {
                bytes,
                height: info.height,
                width: info.width,
            });
        }
        if frames.is_empty() {
            return Err("Empty corpus".into());
        }
        let mut batches = Vec::new();
        for start in (0..frames.len()).step_by(batch_size) {
            let stop = (start + batch_size).min(frames.len());
            let auto = frames[start..stop]
                .iter()
                .all(|f| (f.height, f.width) == (frames[start].height, frames[start].width));
            let layouts: Vec<_> = frames[start..stop]
                .iter()
                .map(|f| layout(f, metadata.imgsz, metadata.stride, auto))
                .collect();
            if layouts
                .iter()
                .any(|l| (l.h, l.w) != (layouts[0].h, layouts[0].w))
            {
                return Err("Inconsistent batch shape".into());
            }
            batches.push(Batch {
                start,
                stop,
                layouts,
            });
        }
        Ok(Self {
            engine,
            frames,
            batches,
            data,
        })
    }

    fn detect(&self, batch: &Batch, dump: Option<usize>) -> Result<Vec<Vec<[f32; 6]>>, String> {
        let first = &batch.layouts[0];
        let image_bytes = first.h * first.w * 3;
        let n = batch.stop - batch.start;
        let mut pixels = vec![0u8; image_bytes * n];
        for (i, (frame, l)) in self.frames[batch.start..batch.stop]
            .iter()
            .zip(&batch.layouts)
            .enumerate()
        {
            // All slices are validated against the manifest and calculated layouts.
            status(unsafe {
                chd_letterbox(
                    frame.bytes.as_ptr(),
                    frame.height as i32,
                    frame.width as i32,
                    pixels[i * image_bytes..].as_mut_ptr(),
                    l.rh as i32,
                    l.rw as i32,
                    l.h as i32,
                    l.w as i32,
                    l.top as i32,
                    l.left as i32,
                )
            })?;
        }
        let mut output = vec![0f32; n * 300 * 6];
        status(unsafe {
            chd_detect(
                self.engine.0,
                pixels.as_ptr(),
                output.as_mut_ptr(),
                n as i32,
                first.h as i32,
                first.w as i32,
            )
        })?;
        if let Some(index) = dump {
            let mut input = vec![0f32; image_bytes * n];
            status(unsafe { chd_input(self.engine.0, input.as_mut_ptr(), input.len()) })?;
            let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(self.data.join(format!("rust.input.{index}.f32")), bytes)
                .map_err(|e| e.to_string())?;
            let raw: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(self.data.join(format!("rust.output.{index}.f32")), raw)
                .map_err(|e| e.to_string())?;
        }
        let mut results = Vec::new();
        for (i, frame) in self.frames[batch.start..batch.stop].iter().enumerate() {
            let gain =
                (first.h as f64 / frame.height as f64).min(first.w as f64 / frame.width as f64);
            let new_h = (frame.height as f64 * gain).round_ties_even().max(1.0);
            let new_w = (frame.width as f64 * gain).round_ties_even().max(1.0);
            let gx = (new_w / frame.width as f64) as f32;
            let gy = (new_h / frame.height as f64) as f32;
            let px = ((first.w as f64 - new_w) * 0.5 - 0.1).round_ties_even() as f32;
            let py = ((first.h as f64 - new_h) * 0.5 - 0.1).round_ties_even() as f32;
            let mut rows = Vec::new();
            for detection in output[i * 1800..(i + 1) * 1800].chunks_exact(6) {
                if detection[4] <= 0.25 {
                    continue;
                }
                let mut row: [f32; 6] = detection.try_into().unwrap();
                for coordinate in [0, 2] {
                    row[coordinate] = ((row[coordinate] - px) / gx).clamp(0., frame.width as f32);
                }
                for coordinate in [1, 3] {
                    row[coordinate] = ((row[coordinate] - py) / gy).clamp(0., frame.height as f32);
                }
                rows.push(row);
                if rows.len() == 300 {
                    break;
                }
            }
            results.push(rows);
        }
        Ok(results)
    }

    fn execute(&self, command: &Value) -> Result<Value, String> {
        let mode = command["mode"].as_str().ok_or("Missing mode")?;
        let repeats = command["repeats"].as_u64().unwrap_or(1) as usize;
        if mode == "verify" {
            let mut all = Vec::new();
            for (index, batch) in self.batches.iter().enumerate() {
                all.extend(self.detect(batch, Some(index))?);
            }
            return Ok(json!({"detections":all}));
        }
        let mut times = Vec::new();
        let mut checksum = 0.;
        if mode == "infer" {
            let batch = &self.batches[0];
            let l = &batch.layouts[0];
            let tensor = floats(&self.data.join("python.input.0.f32"))?;
            if tensor.len() != (batch.stop - batch.start) * 3 * l.h * l.w {
                return Err("Tensor shape mismatch".into());
            }
            status(unsafe {
                chd_load(
                    self.engine.0,
                    tensor.as_ptr(),
                    (batch.stop - batch.start) as i32,
                    l.h as i32,
                    l.w as i32,
                )
            })?;
            for _ in 0..20 {
                status(unsafe { chd_run(self.engine.0) })?;
            }
            for _ in 0..repeats {
                let start = Instant::now();
                status(unsafe { chd_run(self.engine.0) })?;
                times.push(start.elapsed().as_secs_f64() * 1000.);
            }
        } else if mode == "detect" {
            for _ in 0..repeats {
                for batch in &self.batches {
                    let start = Instant::now();
                    let result = self.detect(batch, None)?;
                    std::hint::black_box(&result);
                    checksum += result.iter().flatten().map(|r| r[4] as f64).sum::<f64>();
                    times.push(start.elapsed().as_secs_f64() * 1000.);
                }
            }
        } else {
            return Err("Unknown benchmark mode".into());
        }
        Ok(json!({"times_ms":times,"checksum":checksum,"images":
            if mode=="infer" {repeats * (self.batches[0].stop-self.batches[0].start)} else {repeats*self.frames.len()}}))
    }
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("Usage: binary manifest.json model.engine batch".into());
    }
    let worker = Worker::new(
        Path::new(&args[1]),
        Path::new(&args[2]),
        args[3].parse().map_err(|e| format!("{e}"))?,
    )?;
    println!("{}", json!({"ready":true}));
    io::stdout().flush().map_err(|e| e.to_string())?;
    for line in io::stdin().lock().lines() {
        let value: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let response = match worker.execute(&value) {
            Ok(result) => result,
            Err(error) => json!({"error":error}),
        };
        println!("{response}");
        io::stdout().flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

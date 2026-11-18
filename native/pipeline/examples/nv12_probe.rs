//! Diagnostic colour/tensor comparison; downloads are deliberately outside
//! every performance experiment. No production service or Python dependency.
use anyhow::{Result, ensure};
use construction_inference::{
    core::Image,
    engine::{NativeEngine, Nv12Image, Nv12Planes},
};
use construction_media::{Decoder, GpuMetrics};
use serde_json::json;
use std::{
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use tokio_util::sync::CancellationToken;

fn image(frame: &Arc<construction_media::Nv12Frame>) -> Nv12Image {
    let p = frame.info();
    // SAFETY: the retained immutable AVFrame owns both planes and the event.
    unsafe {
        Nv12Image::new(
            Nv12Planes {
                y: p.y,
                uv: p.uv,
                ready: p.ready,
                lut: p.lut,
                width: p.width,
                height: p.height,
                yp: p.yp,
                uvp: p.uvp,
                colorspace: p.colorspace,
                range: p.range,
                device: p.device,
            },
            frame.clone(),
        )
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 4, "Usage: nv12_probe ENGINE VIDEO FRAMES");
    let count: usize = args[3].parse()?;
    let metrics = Arc::new(GpuMetrics::default());
    let mut gpu = Some(Decoder::open_device(
        &args[2],
        0,
        CancellationToken::new(),
        Duration::from_secs(10),
        metrics.clone(),
    )?);
    let mut cpu = Decoder::open(&args[2], CancellationToken::new(), Duration::from_secs(10))?;
    let mut engine = NativeEngine::new(&args[1], 8, 640, 0, false, 0.25, true)?;
    let mut frames = vec![];
    for index in 0..count {
        ensure!(
            gpu.as_mut().unwrap().next_frame()?.is_some() && cpu.next_frame()?.is_some(),
            "Short source"
        );
        let frame = gpu.as_mut().unwrap().retain_nv12()?;
        let p = frame.info();
        let mut pixels = vec![0; p.width as usize * p.height as usize * 3 + 64];
        cpu.bgr(p.width as u32, p.height as u32, &mut pixels)?;
        let gpu_bgr = frame.bgr_padded()?;
        let decoded_max_error = pixels
            .iter()
            .zip(gpu_bgr)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let input = image(&frame);
        let weak = Arc::downgrade(&frame);
        if index + 1 == count {
            drop(gpu.take());
        }
        let ticket = engine
            .submit_nv12(&[input])?
            .expect("Supported geometry/profile");
        drop(frame);
        ensure!(
            weak.upgrade().is_some(),
            "GPU owner released before collect"
        );
        let actual = engine.inspect_input(ticket)?;
        let actual_rows = engine.collect(ticket)?;
        ensure!(
            weak.upgrade().is_none(),
            "GPU frame owner leaked after collect"
        );
        let ticket = engine.submit_images(&[Image {
            h: p.height as usize,
            w: p.width as usize,
            data: &pixels[..pixels.len() - 64],
        }])?;
        let expected = engine.inspect_input(ticket)?;
        let expected_rows = engine.collect(ticket)?;
        let mut changed = 0;
        let mut max_error = 0f32;
        let mut error_sum = 0f64;
        for (a, b) in actual.iter().zip(&expected) {
            changed += usize::from(a.to_bits() != b.to_bits());
            let error = (a - b).abs() * 255.;
            max_error = max_error.max(error);
            error_sum += f64::from(error);
        }
        frames.push(json!({"index":index,"source_colorspace":p.colorspace,"source_range":p.range,
            "width":p.width,"height":p.height,"decoded_bgr_max_error":decoded_max_error,
            "tensor_elements":actual.len(),"tensor_changed_elements":changed,
            "tensor_max_byte_error":max_error,"tensor_mean_byte_error":error_sum/actual.len() as f64,
            "gpu_detections":actual_rows[0],"cpu_detections":expected_rows[0]}));
    }
    // Also destroy an engine with an uncollected ticket. Its GPU work must
    // finish before the Rust ticket owners (and native AVFrame) are released.
    let mut decoder = Decoder::open_device(
        &args[2],
        0,
        CancellationToken::new(),
        Duration::from_secs(10),
        metrics.clone(),
    )?;
    ensure!(decoder.next_frame()?.is_some(), "Empty lifetime fixture");
    let frame = decoder.retain_nv12()?;
    let weak = Arc::downgrade(&frame);
    engine
        .submit_nv12(&[image(&frame)])?
        .expect("Supported lifetime fixture");
    drop(frame);
    drop(decoder);
    ensure!(weak.upgrade().is_some(), "Pending owner missing");
    drop(engine);
    ensure!(weak.upgrade().is_none(), "Engine drop leaked pending owner");
    let mut decoder = Decoder::open_device(
        &args[2],
        0,
        CancellationToken::new(),
        Duration::from_secs(10),
        metrics.clone(),
    )?;
    ensure!(decoder.next_frame()?.is_some(), "Empty thread-move fixture");
    let frame = decoder.retain_nv12()?;
    let input = image(&frame);
    let mut moved = NativeEngine::new(&args[1], 8, 640, 0, false, 0.25, true)?;
    drop(frame);
    drop(decoder);
    std::thread::spawn(move || -> Result<()> {
        let ticket = moved.submit_nv12(&[input])?.expect("Thread-move geometry");
        moved.collect(ticket)?;
        Ok(())
    })
    .join()
    .expect("GPU thread panicked")?;
    println!(
        "{}",
        json!({"frames":frames,"ownership_checks_passed":true,
        "diagnostic_materializations":metrics.materialized.load(Ordering::Relaxed),
        "diagnostic_download_payload_bytes":metrics.download_payload_bytes.load(Ordering::Relaxed)})
    );
    Ok(())
}

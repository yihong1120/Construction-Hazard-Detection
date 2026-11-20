//! CPU YUV ownership and per-image input parity against the regular CPU path.
use anyhow::{Result, ensure};
use construction_inference::{
    core::{Image, layout},
    resize,
};
use construction_media::{Decoder, GpuMetrics};
use sha2::{Digest, Sha256};
use std::{
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 5, "host_probe SOURCE FRAMES WIDTH HEIGHT");
    let (count, w, h): (usize, u32, u32) = (a[2].parse()?, a[3].parse()?, a[4].parse()?);
    let stats = Arc::new(GpuMetrics::default());
    let mut decoder = Decoder::open_device(
        &a[1],
        -1,
        CancellationToken::new(),
        Duration::from_secs(10),
        stats.clone(),
    )?;
    let l = layout(h as usize, w as usize, 640, 32, false);
    let mut reference = vec![0; w as usize * h as usize * 3 + 64];
    let mut expected = vec![0; l.h * l.w * 3];
    let mut actual = expected.clone();
    let mut rows = vec![];
    let mut retained = None;
    for i in 0..count {
        ensure!(decoder.next_frame()?.is_some(), "Short source");
        let host = decoder.retain_host(w, h)?;
        actual.fill(23);
        let fused_start = Instant::now();
        let fused = host.letterbox(&mut actual, l.h, l.w, l.rh, l.rw, l.top, l.left);
        let fused_ms = fused_start.elapsed().as_secs_f64() * 1000.0;
        let materialized_before = stats.host_materialized.load(Ordering::Relaxed);
        let reference_start = Instant::now();
        let bgr = host.bgr_padded()?;
        let image = Image {
            h: h as usize,
            w: w as usize,
            data: &bgr[..bgr.len() - 64],
        };
        let supported = resize::letterbox(&image, &l, &mut expected);
        #[cfg(feature = "tensorrt")]
        let supported = if !supported {
            construction_inference::engine::NativeEngine::fill_letterbox(
                &image,
                &l,
                &mut expected,
            )?;
            true
        } else {
            supported
        };
        let reference_ms = reference_start.elapsed().as_secs_f64() * 1000.0;
        if fused && supported {
            ensure!(expected == actual, "Model input differs on frame {i}");
        } else if !fused {
            ensure!(
                actual.iter().all(|byte| *byte == 23),
                "Ineligible sampling modified the output"
            );
        }
        ensure!(
            stats.host_materialized.load(Ordering::Relaxed) == materialized_before + 1,
            "Fused sampling must leave full BGR unmaterialized"
        );
        decoder.bgr(w, h, &mut reference)?;
        ensure!(
            bgr[..reference.len() - 64] == reference[..reference.len() - 64],
            "Full BGR differs"
        );
        rows.push(serde_json::json!({
            "frame": i,
            "fused": fused,
            "parity_checked": fused && supported,
            "lookup_bytes": host.lookup_bytes(),
            "fused_ms": fused_ms,
            "reference_ms": reference_ms,
            "sha256": fused.then(|| hex::encode(Sha256::digest(&actual))),
        }));
        if retained.is_none() {
            retained = Some((
                decoder.retain_host(w, h)?,
                reference.clone(),
                if fused && supported {
                    Some(expected.clone())
                } else {
                    None
                },
            ));
        }
    }
    drop(decoder);
    let (owner, first, model) = retained.unwrap();
    if let Some(model) = model {
        let mut fresh = vec![0; model.len()];
        ensure!(owner.letterbox(&mut fresh, l.h, l.w, l.rh, l.rw, l.top, l.left));
        ensure!(
            fresh == model,
            "Decoder Drop invalidated retained YUV planes"
        );
    }
    ensure!(
        owner.bgr_padded()? == first,
        "Decoder Drop invalidated retained frame"
    );
    println!(
        "{}",
        serde_json::json!({"frames":rows,"host_retained":stats.host_retained.load(Ordering::Relaxed),"host_sampled":stats.host_sampled.load(Ordering::Relaxed)})
    );
    Ok(())
}
